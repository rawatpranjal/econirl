"""Maximum Margin Planning (Ratliff, Bagnell & Zinkevich 2006).

This module implements the Maximum Margin Planning (MMP) algorithm, which learns
reward functions from demonstrated behavior using structured large-margin methods.

Key differences from Max Margin IRL (Abbeel & Ng 2004):
- Uses **subgradient descent** instead of QP solving
- Incorporates **loss-augmented inference** (adds loss to MDP reward during training)
- More **scalable** to large state spaces
- Uses **L2 regularization** instead of norm constraint

Algorithm:
    Objective: min_θ (λ/2)||θ||² + (1/N) Σᵢ max_π [Δ(π*, π) + θᵀ(φ(π) - φ(π*))]

    For t = 1, ..., max_iter:
        1. Loss-augmented inference: π̂ = argmax [θᵀφ(π) + Δ(π*, π)]
           - Solve MDP with reward R_aug(s,a) = θᵀφ(s,a) + scale * Δ(s,a)
        2. Compute features: μ̂ = φ(π̂), μ* = φ(π_expert)
        3. Subgradient: g = λθ + (μ̂ - μ*)
        4. Update: θ_{t+1} = θ_t - η_t * g

Reference:
    Ratliff, N., Bagnell, J.A., & Zinkevich, M. (2006). "Maximum Margin Planning."
    In Proceedings of the 23rd International Conference on Machine Learning (ICML).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class MMPConfig:
    """Configuration for Maximum Margin Planning estimation.

    Attributes:
        learning_rate: Base learning rate for subgradient descent.
        learning_rate_schedule: Schedule for learning rate decay.
            - "constant": η_t = η_0
            - "1/t": η_t = η_0 / t
            - "1/sqrt(t)": η_t = η_0 / sqrt(t)
        max_iterations: Maximum number of subgradient iterations.
        convergence_tol: Convergence tolerance on gradient norm.
        regularization_lambda: L2 regularization coefficient λ.

        loss_type: Type of loss function Δ(π*, π).
            - "policy_kl": Δ(s,a) = -log(π*(a|s) + ε), high loss for rare expert actions
            - "trajectory_hamming": Δ(s,a) = 1 - π*(a|s), soft 0-1 loss
        loss_scale: Scale factor for loss in loss-augmented inference.

        inner_tol: Tolerance for inner value iteration solver.
        inner_max_iter: Maximum iterations for inner solver.
        inner_solver: Solver type for inner loop ("value" or "hybrid").

        compute_se: Whether to compute standard errors.
        n_bootstrap: Number of bootstrap samples for standard errors.
        verbose: Whether to print progress messages.
    """

    # Optimization
    learning_rate: float = 0.1
    learning_rate_schedule: Literal["constant", "1/t", "1/sqrt(t)"] = "1/sqrt(t)"
    max_iterations: int = 200
    convergence_tol: float = 1e-5
    regularization_lambda: float = 0.01

    # Loss function
    loss_type: Literal["policy_kl", "trajectory_hamming"] = "policy_kl"
    loss_scale: float = 1.0

    # Inner solver
    inner_tol: float = 1e-8
    inner_max_iter: int = 5000
    inner_solver: Literal["value", "hybrid"] = "hybrid"

    # Inference
    compute_se: bool = True
    n_bootstrap: int = 100
    verbose: bool = False


class MaxMarginPlanningEstimator(BaseEstimator):
    """Maximum Margin Planning Estimator (Ratliff, Bagnell & Zinkevich 2006).

    Learns reward function parameters from expert demonstrations using
    structured large-margin methods. The key innovation is loss-augmented
    inference, which modifies the MDP reward during training to find the
    most violating policy.

    Unlike standard Max Margin IRL (Abbeel & Ng 2004), MMP:
    - Uses subgradient descent instead of QP solving
    - Scales to larger state spaces
    - Uses L2 regularization instead of norm constraints
    - Incorporates task-specific loss functions

    Parameters
    ----------
    config : MMPConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Attributes
    ----------
    config : MMPConfig
        Configuration object.

    Examples
    --------
    >>> from econirl.estimation.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
    >>> from econirl.preferences.action_reward import ActionDependentReward
    >>>
    >>> # Create estimator with custom config
    >>> config = MMPConfig(verbose=True, max_iterations=100)
    >>> estimator = MaxMarginPlanningEstimator(config=config)
    >>>
    >>> # Estimate reward from demonstrations
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: MMPConfig | None = None,
        **kwargs,
    ):
        """Initialize the Maximum Margin Planning estimator.

        Args:
            config: Configuration object. If None, uses default config.
            **kwargs: Override individual config parameters.
        """
        # Build config from defaults + overrides
        if config is None:
            config = MMPConfig(**kwargs)
        else:
            # Apply any kwargs as overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method="bootstrap" if config.compute_se else "asymptotic",
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "MMP (Ratliff, Bagnell & Zinkevich 2006)"

    def _estimate_expert_policy(
        self,
        panel: Panel,
        problem: DDCProblem,
    ) -> torch.Tensor:
        """Estimate expert policy from demonstration data using CCP estimation.

        Computes the empirical conditional choice probabilities:
            π̂*(a|s) = count(s,a) / count(s)

        Args:
            panel: Panel data with expert demonstrations.
            problem: Problem specification.

        Returns:
            Expert policy tensor of shape (num_states, num_actions).
        """
        n_states = problem.num_states
        n_actions = problem.num_actions

        # Count state-action pairs
        state_action_counts = torch.zeros((n_states, n_actions), dtype=torch.float32)
        state_counts = torch.zeros(n_states, dtype=torch.float32)

        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                state_action_counts[s, a] += 1
                state_counts[s] += 1

        # Convert to probabilities with smoothing for unvisited states
        expert_policy = torch.zeros((n_states, n_actions), dtype=torch.float32)

        for s in range(n_states):
            if state_counts[s] > 0:
                expert_policy[s, :] = state_action_counts[s, :] / state_counts[s]
            else:
                # Uniform distribution for unvisited states
                expert_policy[s, :] = 1.0 / n_actions

        return expert_policy

    def _compute_loss_matrix(
        self,
        expert_policy: torch.Tensor,
        n_states: int,
        n_actions: int,
    ) -> torch.Tensor:
        """Compute loss matrix Δ(s,a) based on expert policy.

        The loss encourages the learned policy to match the expert.
        Higher loss for actions the expert rarely takes.

        Args:
            expert_policy: Expert policy π*(a|s), shape (n_states, n_actions).
            n_states: Number of states.
            n_actions: Number of actions.

        Returns:
            Loss matrix of shape (n_states, n_actions).
        """
        if self.config.loss_type == "policy_kl":
            # KL-based loss: Δ(s,a) = -log(π*(a|s) + ε)
            # High loss for actions the expert rarely takes
            eps = 1e-10
            loss_matrix = -torch.log(expert_policy + eps)
        elif self.config.loss_type == "trajectory_hamming":
            # Soft 0-1 loss: Δ(s,a) = 1 - π*(a|s)
            # 0 for actions expert always takes, 1 for actions never taken
            loss_matrix = 1.0 - expert_policy
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        return loss_matrix

    def _loss_augmented_value_iteration(
        self,
        reward_matrix: torch.Tensor,
        loss_matrix: torch.Tensor,
        operator: SoftBellmanOperator,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Solve MDP with loss-augmented reward.

        The key MMP innovation: during training, we solve the MDP with
        R_aug(s,a) = R(s,a) + loss_scale * Δ(s,a)

        This finds the most violating policy - the one that maximizes
        the structured hinge loss.

        Args:
            reward_matrix: Reward matrix R(s,a), shape (n_states, n_actions).
            loss_matrix: Loss matrix Δ(s,a), shape (n_states, n_actions).
            operator: Bellman operator.

        Returns:
            Tuple of (value function, policy, converged).
        """
        # Augmented reward = reward + loss
        augmented_reward = reward_matrix + self.config.loss_scale * loss_matrix

        if self.config.inner_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                augmented_reward,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )
        else:
            result = value_iteration(
                operator,
                augmented_reward,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )

        return result.V, result.policy, result.converged

    def _compute_expert_features(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
    ) -> torch.Tensor:
        """Compute expert feature expectations from demonstration data.

        The feature expectation is the average feature over all state-action
        pairs visited by the expert:
            μ* = (1/N) Σ_{i,t} φ(s_{i,t}, a_{i,t})

        Args:
            panel: Panel data with expert demonstrations.
            reward_fn: Reward function with feature matrix.

        Returns:
            Expert feature expectations, shape (n_features,).
        """
        # Get feature matrix
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
            is_action_dependent = True
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
            is_action_dependent = False
        else:
            raise TypeError(
                f"MaxMarginPlanningEstimator requires LinearReward or "
                f"ActionDependentReward, got {type(reward_fn)}"
            )

        n_features = reward_fn.num_parameters
        feature_sum = torch.zeros(n_features, dtype=torch.float32)
        total_count = 0

        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                if is_action_dependent:
                    a = traj.actions[t].item()
                    feature_sum += feature_matrix[s, a, :]
                else:
                    feature_sum += feature_matrix[s, :]
                total_count += 1

        if total_count > 0:
            return feature_sum / total_count
        return feature_sum

    def _compute_policy_features(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        initial_distribution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute feature expectations under a given policy.

        Uses the stationary distribution under the policy to compute:
            μ_π = Σ_s d_π(s) Σ_a π(a|s) φ(s,a)

        where d_π(s) is the discounted state visitation frequency.

        Args:
            policy: Policy π(a|s), shape (n_states, n_actions).
            transitions: Transition matrices P(s'|s,a).
            reward_fn: Reward function with feature matrix.
            problem: Problem specification.
            initial_distribution: Initial state distribution (optional).

        Returns:
            Policy feature expectations, shape (n_features,).
        """
        n_states = problem.num_states
        gamma = problem.discount_factor

        # Compute policy-induced transition matrix
        # P_π[s, s'] = Σ_a π(a|s) * P(s'|s,a)
        P_pi = torch.einsum("sa,ast->st", policy, transitions)

        # Compute discounted state visitation frequencies
        if initial_distribution is None:
            d_0 = torch.ones(n_states, dtype=torch.float32) / n_states
        else:
            d_0 = initial_distribution

        # Solve d = (1 - γ)(I - γ P_π^T)^{-1} d_0
        I = torch.eye(n_states, dtype=torch.float32)
        try:
            inv_matrix = torch.linalg.solve(I - gamma * P_pi.T, d_0)
            d_pi = (1 - gamma) * inv_matrix
            d_pi = d_pi / d_pi.sum()  # Normalize
        except RuntimeError:
            # Fallback: iterative computation
            d_pi = d_0.clone()
            for _ in range(1000):
                d_new = d_0 * (1 - gamma) + gamma * (P_pi.T @ d_pi)
                if torch.abs(d_new - d_pi).max() < 1e-10:
                    break
                d_pi = d_new
            d_pi = d_pi / d_pi.sum()

        # Compute feature expectations based on reward type
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
            # μ_π = Σ_s d_π(s) Σ_a π(a|s) φ(s,a,k)
            policy_weighted_features = torch.einsum("sa,sak->sk", policy, feature_matrix)
            feature_expectations = torch.einsum("s,sk->k", d_pi, policy_weighted_features)
        elif isinstance(reward_fn, LinearReward):
            # μ_π = Σ_s d_π(s) φ(s)
            feature_expectations = torch.einsum("s,sk->k", d_pi, reward_fn.state_features)
        else:
            raise TypeError(
                f"MaxMarginPlanningEstimator requires LinearReward or "
                f"ActionDependentReward, got {type(reward_fn)}"
            )

        return feature_expectations

    def _compute_subgradient(
        self,
        theta: torch.Tensor,
        expert_features: torch.Tensor,
        policy_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute subgradient of the MMP objective.

        The subgradient is:
            g = λθ + (μ̂ - μ*)

        where:
        - λ is the regularization coefficient
        - θ are the current parameters
        - μ̂ are feature expectations under the loss-augmented policy
        - μ* are expert feature expectations

        Args:
            theta: Current parameter estimate.
            expert_features: Expert feature expectations μ*.
            policy_features: Loss-augmented policy feature expectations μ̂.

        Returns:
            Subgradient vector.
        """
        # Regularization term
        reg_term = self.config.regularization_lambda * theta

        # Feature difference term
        feature_diff = policy_features - expert_features

        return reg_term + feature_diff

    def _get_learning_rate(self, iteration: int) -> float:
        """Get learning rate at given iteration.

        Args:
            iteration: Current iteration (1-indexed).

        Returns:
            Learning rate for this iteration.
        """
        base_lr = self.config.learning_rate

        if self.config.learning_rate_schedule == "constant":
            return base_lr
        elif self.config.learning_rate_schedule == "1/t":
            return base_lr / iteration
        elif self.config.learning_rate_schedule == "1/sqrt(t)":
            return base_lr / np.sqrt(iteration)
        else:
            raise ValueError(f"Unknown schedule: {self.config.learning_rate_schedule}")

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        true_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run Maximum Margin Planning optimization.

        Args:
            panel: Panel data with expert demonstrations.
            utility: Reward function specification.
            problem: Problem specification.
            transitions: Transition probability matrices.
            initial_params: Initial parameter values (optional).
            true_params: True parameters for debugging (optional).

        Returns:
            EstimationResult with optimized parameters.
        """
        start_time = time.time()

        # Verify utility is a supported type
        if not isinstance(utility, (LinearReward, ActionDependentReward)):
            raise TypeError(
                f"MaxMarginPlanningEstimator requires LinearReward or "
                f"ActionDependentReward, got {type(utility)}"
            )

        reward_fn = utility
        n_features = reward_fn.num_parameters
        n_states = problem.num_states
        n_actions = problem.num_actions

        # Create Bellman operator
        operator = SoftBellmanOperator(problem, transitions)

        # Initialize parameters
        if initial_params is None:
            theta = reward_fn.get_initial_parameters()
        else:
            theta = initial_params.clone()

        # Estimate expert policy from demonstrations
        expert_policy = self._estimate_expert_policy(panel, problem)
        self._log(f"Estimated expert policy from {panel.num_observations} observations")

        # Compute loss matrix
        loss_matrix = self._compute_loss_matrix(expert_policy, n_states, n_actions)
        self._log(f"Computed loss matrix (type={self.config.loss_type})")

        # Compute expert feature expectations (constant throughout optimization)
        expert_features = self._compute_expert_features(panel, reward_fn)
        self._log(f"Expert features: {expert_features}")

        # Compute initial state distribution from panel
        initial_dist = panel.compute_state_frequencies(n_states)

        # Tracking
        converged = False
        best_obj = float("inf")
        best_theta = theta.clone()
        inner_not_converged = 0

        # Progress bar
        pbar = tqdm(
            range(1, self.config.max_iterations + 1),
            desc="MMP",
            disable=not self.config.verbose,
            leave=True,
        )

        for iteration in pbar:
            # Step 1: Loss-augmented inference
            reward_matrix = reward_fn.compute(theta)
            V, policy, inner_converged = self._loss_augmented_value_iteration(
                reward_matrix, loss_matrix, operator
            )
            if not inner_converged:
                inner_not_converged += 1

            # Step 2: Compute policy feature expectations
            policy_features = self._compute_policy_features(
                policy, transitions, reward_fn, problem, initial_dist
            )

            # Step 3: Compute subgradient
            subgradient = self._compute_subgradient(theta, expert_features, policy_features)
            grad_norm = torch.norm(subgradient).item()

            # Objective value (for tracking)
            # L(θ) = (λ/2)||θ||² + structured hinge loss
            reg_loss = 0.5 * self.config.regularization_lambda * torch.sum(theta ** 2)
            hinge_loss = torch.dot(theta, policy_features - expert_features)
            # Add the actual loss value: Σ Δ(π*, π̂)
            loss_value = torch.sum(loss_matrix * policy)
            obj = (reg_loss + hinge_loss + self.config.loss_scale * loss_value).item()

            # Track best
            if obj < best_obj:
                best_obj = obj
                best_theta = theta.clone()

            # Update progress bar
            postfix = {
                "obj": f"{obj:.4f}",
                "||g||": f"{grad_norm:.4f}",
            }
            if true_params is not None:
                rmse = torch.sqrt(torch.mean((theta - true_params) ** 2)).item()
                postfix["RMSE"] = f"{rmse:.4f}"
            pbar.set_postfix(postfix)

            # Check convergence
            if grad_norm < self.config.convergence_tol:
                converged = True
                self._log(f"Converged at iteration {iteration}")
                break

            # Step 4: Subgradient update
            lr = self._get_learning_rate(iteration)
            theta = theta - lr * subgradient

        pbar.close()

        # Use best parameters found
        final_theta = best_theta
        n_iterations = iteration

        # Compute final policy (without loss augmentation)
        final_reward = reward_fn.compute(final_theta)
        if self.config.inner_solver == "hybrid":
            final_result = hybrid_iteration(
                operator, final_reward,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )
        else:
            final_result = value_iteration(
                operator, final_reward,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )

        final_V = final_result.V
        final_policy = final_result.policy

        # Compute log-likelihood on expert data
        log_probs = operator.compute_log_choice_probabilities(final_reward, final_V)
        ll = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ll += log_probs[s, a].item()

        # Feature matching quality
        final_policy_features = self._compute_policy_features(
            final_policy, transitions, reward_fn, problem, initial_dist
        )
        feature_diff = torch.norm(expert_features - final_policy_features).item()

        # Compute Hessian for inference
        hessian = None
        if self.config.compute_se:
            hessian = self._numerical_hessian(
                final_theta, panel, reward_fn, problem, transitions, operator
            )

        optimization_time = time.time() - start_time

        self._log(f"Optimization complete: feature_diff={feature_diff:.6f}, LL={ll:.2f}")
        if inner_not_converged > 0:
            self._log(f"Warning: Inner loop did not converge {inner_not_converged} times")

        return EstimationResult(
            parameters=final_theta,
            log_likelihood=ll,
            value_function=final_V,
            policy=final_policy,
            hessian=hessian,
            converged=converged,
            num_iterations=n_iterations,
            num_function_evals=n_iterations,
            message="Converged" if converged else "Max iterations reached",
            optimization_time=optimization_time,
            metadata={
                "expert_features": expert_features.tolist(),
                "final_policy_features": final_policy_features.tolist(),
                "feature_difference": feature_diff,
                "final_objective": best_obj,
                "regularization_lambda": self.config.regularization_lambda,
                "loss_type": self.config.loss_type,
                "loss_scale": self.config.loss_scale,
                "learning_rate_schedule": self.config.learning_rate_schedule,
                "inner_not_converged": inner_not_converged,
            },
        )

    def _numerical_hessian(
        self,
        params: torch.Tensor,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        operator: SoftBellmanOperator,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Compute numerical Hessian of the log-likelihood.

        Uses central differences for numerical stability.

        Args:
            params: Parameter vector.
            panel: Panel data.
            reward_fn: Reward function.
            problem: Problem specification.
            transitions: Transition matrices.
            operator: Bellman operator.
            eps: Finite difference step size.

        Returns:
            Hessian matrix of shape (n_params, n_params).
        """
        n_params = len(params)
        hessian = torch.zeros((n_params, n_params), dtype=params.dtype)

        def ll_at(p: torch.Tensor) -> float:
            """Compute log-likelihood at parameters p."""
            reward_matrix = reward_fn.compute(p)
            if self.config.inner_solver == "hybrid":
                result = hybrid_iteration(
                    operator, reward_matrix,
                    tol=self.config.inner_tol,
                    max_iter=self.config.inner_max_iter,
                )
            else:
                result = value_iteration(
                    operator, reward_matrix,
                    tol=self.config.inner_tol,
                    max_iter=self.config.inner_max_iter,
                )

            log_probs = operator.compute_log_choice_probabilities(reward_matrix, result.V)

            ll = 0.0
            for traj in panel.trajectories:
                for t in range(len(traj)):
                    s = traj.states[t].item()
                    a = traj.actions[t].item()
                    ll += log_probs[s, a].item()
            return ll

        # Compute Hessian using central differences
        for i in range(n_params):
            h_i = max(eps, min(abs(params[i].item()) * eps, 0.1))

            for j in range(i, n_params):
                h_j = max(eps, min(abs(params[j].item()) * eps, 0.1))

                if i == j:
                    # Diagonal: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h²
                    p_plus = params.clone()
                    p_plus[i] += h_i
                    p_minus = params.clone()
                    p_minus[i] -= h_i

                    ll_plus = ll_at(p_plus)
                    ll_0 = ll_at(params)
                    ll_minus = ll_at(p_minus)

                    hessian[i, i] = (ll_plus - 2 * ll_0 + ll_minus) / (h_i * h_i)
                else:
                    # Off-diagonal: 4-point formula
                    p_pp = params.clone()
                    p_pp[i] += h_i
                    p_pp[j] += h_j

                    p_pm = params.clone()
                    p_pm[i] += h_i
                    p_pm[j] -= h_j

                    p_mp = params.clone()
                    p_mp[i] -= h_i
                    p_mp[j] += h_j

                    p_mm = params.clone()
                    p_mm[i] -= h_i
                    p_mm[j] -= h_j

                    h_ij = (ll_at(p_pp) - ll_at(p_pm) - ll_at(p_mp) + ll_at(p_mm)) / (4 * h_i * h_j)
                    hessian[i, j] = h_ij
                    hessian[j, i] = h_ij

        # Ensure Hessian is negative semi-definite at maximum
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        if (eigenvalues > 0).any():
            self._log("Warning: Hessian not negative semi-definite, projecting")
            eigenvalues_clamped = torch.clamp(eigenvalues, max=-1e-8)
            hessian = eigenvectors @ torch.diag(eigenvalues_clamped) @ eigenvectors.T

        return hessian

    def compute_margin(
        self,
        theta: torch.Tensor,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
    ) -> float:
        """Compute the structured margin for given parameters.

        The margin measures how much better the expert policy is than
        the best alternative under the current reward.

        Args:
            theta: Reward parameters.
            panel: Expert demonstrations.
            reward_fn: Reward function.
            problem: Problem specification.
            transitions: Transition matrices.

        Returns:
            Margin value.
        """
        operator = SoftBellmanOperator(problem, transitions)

        # Expert features
        expert_features = self._compute_expert_features(panel, reward_fn)

        # Find optimal policy under theta
        reward_matrix = reward_fn.compute(theta)
        if self.config.inner_solver == "hybrid":
            result = hybrid_iteration(
                operator, reward_matrix,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )
        else:
            result = value_iteration(
                operator, reward_matrix,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )

        # Policy features
        initial_dist = panel.compute_state_frequencies(problem.num_states)
        policy_features = self._compute_policy_features(
            result.policy, transitions, reward_fn, problem, initial_dist
        )

        # Margin = θ' (μ* - μ_π)
        margin = torch.dot(theta, expert_features - policy_features).item()

        return margin
