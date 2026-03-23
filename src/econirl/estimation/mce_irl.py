"""Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL).

This module implements Maximum Causal Entropy IRL from Ziebart's 2010 thesis,
which properly accounts for the causal structure in sequential decision-making.

The key difference from standard MaxEnt IRL is the backward pass uses soft
value iteration which respects that agents don't know future randomness
at decision time.

Algorithm (following Ziebart 2010):
    BACKWARD PASS:
    1. Initialize Q(s,a) and V(s) at terminal states
    2. Propagate backwards using soft Bellman:
       Q(s,a) = R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')
       V(s) = softmax(Q(s,·))

    FORWARD PASS:
    3. Compute state visitation frequencies:
       D(s) = ρ₀(s) + Σ_{s',a} D(s') π(a|s') P(s|s',a)

    GRADIENT:
    4. Δθ = E_expert[∇R] - E_policy[∇R]
          = empirical_features - Σ_s D(s) ∇R(s)

References:
    Ziebart, B. D. (2010). Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy. PhD thesis, CMU.

    Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008).
        Maximum entropy inverse reinforcement learning. AAAI.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy import optimize
from tqdm import tqdm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class MCEIRLConfig:
    """Configuration for MCE IRL estimation."""

    # Optimization
    optimizer: Literal["L-BFGS-B", "BFGS", "gradient"] = "L-BFGS-B"
    learning_rate: float = 0.02  # Lower than typical SGD; works well with Adam
    outer_tol: float = 1e-6
    outer_max_iter: int = 200
    gradient_clip: float = 1.0  # Max gradient norm (prevents divergence)
    use_adam: bool = True  # Use Adam optimizer (adaptive learning rate)
    adam_beta1: float = 0.9  # Adam first moment decay
    adam_beta2: float = 0.999  # Adam second moment decay
    adam_eps: float = 1e-8  # Adam numerical stability

    # Inner solver (soft value iteration)
    inner_solver: Literal["value", "hybrid"] = "hybrid"  # hybrid is faster
    inner_tol: float = 1e-8
    inner_max_iter: int = 10000  # Higher for high discount factors
    switch_tol: float = 1e-3  # For hybrid: switch to NK when error < this

    # State visitation computation
    svf_tol: float = 1e-8
    svf_max_iter: int = 1000

    # Inference
    compute_se: bool = True
    se_method: Literal["bootstrap", "asymptotic", "hessian"] = "bootstrap"
    n_bootstrap: int = 100

    # Verbosity
    verbose: bool = False


class MCEIRLEstimator(BaseEstimator):
    """Maximum Causal Entropy IRL Estimator.

    Recovers reward function parameters from demonstrated behavior using
    the maximum causal entropy principle (Ziebart 2010).

    This differs from standard MaxEnt IRL in that it properly accounts for
    the causal structure of sequential decisions - agents act before observing
    future randomness.

    Parameters
    ----------
    config : MCEIRLConfig, optional
        Configuration object with algorithm parameters.
    **kwargs
        Override individual config parameters.

    Attributes
    ----------
    config : MCEIRLConfig
        Configuration object.

    Examples
    --------
    >>> from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    >>> from econirl.preferences.reward import LinearReward
    >>>
    >>> # Create estimator with custom config
    >>> config = MCEIRLConfig(verbose=True, n_bootstrap=200)
    >>> estimator = MCEIRLEstimator(config=config)
    >>>
    >>> # Estimate reward from demonstrations
    >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
    >>> print(result.summary())
    """

    def __init__(
        self,
        config: MCEIRLConfig | None = None,
        **kwargs,
    ):
        # Build config from defaults + overrides
        if config is None:
            config = MCEIRLConfig(**kwargs)
        else:
            # Apply any kwargs as overrides
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Map "hessian" to "asymptotic" for compatibility with standard_errors module
        # (they are semantically equivalent: both use inverse Hessian for variance)
        if config.compute_se:
            if config.se_method == "hessian":
                effective_se_method = "asymptotic"
            else:
                effective_se_method = config.se_method
        else:
            effective_se_method = "asymptotic"

        super().__init__(
            se_method=effective_se_method,
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )
        self.config = config

    @property
    def name(self) -> str:
        return "MCE IRL (Ziebart 2010)"

    def _soft_value_iteration(
        self,
        operator: SoftBellmanOperator,
        reward_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Run soft value iteration (backward pass).

        Computes the soft value function and policy using the soft Bellman
        equation, which is the backward pass in MCE IRL.

        Uses either pure value iteration or hybrid (contraction + Newton-Kantorovich)
        based on config.inner_solver setting. Hybrid is typically 5-7x faster.

        Parameters
        ----------
        operator : SoftBellmanOperator
            Bellman operator with problem and transitions.
        reward_matrix : torch.Tensor
            Reward matrix R(s,a), shape (n_states, n_actions).

        Returns
        -------
        V : torch.Tensor
            Soft value function, shape (n_states,).
        policy : torch.Tensor
            Soft policy π(a|s), shape (n_states, n_actions).
        converged : bool
            Whether the iteration converged.
        """
        if self.config.inner_solver == "hybrid":
            result = hybrid_iteration(
                operator,
                reward_matrix,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
                switch_tol=self.config.switch_tol,
            )
            return result.V, result.policy, result.converged
        else:
            # Pure value iteration (original implementation)
            result = value_iteration(
                operator,
                reward_matrix,
                tol=self.config.inner_tol,
                max_iter=self.config.inner_max_iter,
            )
            return result.V, result.policy, result.converged

    def _compute_state_visitation(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        problem: DDCProblem,
        initial_dist: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute expected state visitation frequencies (forward pass).

        Uses forward message passing to compute the expected number of times
        each state is visited under the policy.

        Parameters
        ----------
        policy : torch.Tensor
            Policy π(a|s), shape (n_states, n_actions).
        transitions : torch.Tensor
            Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
        problem : DDCProblem
            Problem specification.
        initial_dist : torch.Tensor, optional
            Initial state distribution ρ₀. If None, uses uniform.

        Returns
        -------
        D : torch.Tensor
            State visitation frequencies, shape (n_states,).
        """
        n_states = problem.num_states
        gamma = problem.discount_factor

        if initial_dist is None:
            D = torch.ones(n_states, dtype=policy.dtype) / n_states
        else:
            D = initial_dist.clone()

        # Compute policy-weighted transition: P_π(s'|s) = Σ_a π(a|s) P(s'|s,a)
        # transitions: (n_actions, n_states, n_states) = [a, from_s, to_s]
        # policy: (n_states, n_actions) = [s, a]
        P_pi = torch.einsum("sa,ast->st", policy, transitions)

        # Fixed point iteration: D = ρ₀ + γ P_π^T D
        # Equivalently: D = (I - γ P_π^T)^{-1} ρ₀
        # But we use iteration for numerical stability

        rho0 = D.clone()
        for i in range(self.config.svf_max_iter):
            D_new = rho0 + gamma * (P_pi.T @ D)

            delta = torch.abs(D_new - D).max().item()
            D = D_new

            if delta < self.config.svf_tol:
                break

        # Normalize to probability distribution
        D = D / D.sum()

        return D

    def _compute_empirical_features(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
    ) -> torch.Tensor:
        """Compute empirical feature expectations from demonstrations.

        Computes the average state-action features visited by the demonstrator:
            μ_D = (1/N) Σ_{i,t} φ(s_{i,t}, a_{i,t})

        For state-only features: μ_D = (1/N) Σ_{i,t} φ(s_{i,t})

        Handles both:
        - LinearReward (state-only): 2D feature matrix (n_states, n_features)
        - ActionDependentReward: 3D feature matrix (n_states, n_actions, n_features)
        """
        # Get feature matrix - handle both 2D and 3D cases
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
        else:
            # Try to get feature_matrix, fall back to state_features
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        total_obs = sum(len(traj) for traj in panel.trajectories)

        if feature_matrix.ndim == 3:
            # Action-dependent features: (n_states, n_actions, n_features)
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            feature_sum = feature_matrix[all_states, all_actions, :].sum(dim=0)
        else:
            # State-only features: (n_states, n_features)
            all_states = panel.get_all_states()
            feature_sum = feature_matrix[all_states, :].sum(dim=0)

        if total_obs > 0:
            return feature_sum / total_obs
        return feature_sum

    def _compute_expected_features(
        self,
        panel: Panel,
        policy: torch.Tensor,
        reward_fn: BaseUtilityFunction,
        transitions: torch.Tensor | None = None,
        initial_dist: torch.Tensor | None = None,
        discount: float = 0.9,
    ) -> torch.Tensor:
        """Compute expected feature expectations under the learned policy.

        For action-dependent features (3D), computes expectations over the
        EMPIRICAL state sequence, weighting by the current policy's action
        probabilities:

            μ_π = (1/N) Σ_{i,t} Σ_a π(a|s_{i,t}) φ(s_{i,t}, a)

        For state-only features (2D), computes expectations using the state
        visitation distribution under the current policy:

            μ_π = Σ_s d_π(s) φ(s)

        Parameters
        ----------
        panel : Panel
            Panel data containing the empirical state sequence.
        policy : torch.Tensor
            Policy probabilities π(a|s), shape (n_states, n_actions).
        reward_fn : BaseUtilityFunction
            Reward function with feature matrix.
        transitions : torch.Tensor, optional
            Transition matrices, shape (n_actions, n_states, n_states).
            Required for state-only features.
        initial_dist : torch.Tensor, optional
            Initial state distribution. Required for state-only features.
        discount : float
            Discount factor for state visitation computation.

        Returns
        -------
        torch.Tensor
            Expected features, shape (n_features,).
        """
        # Get feature matrix - handle both 2D and 3D cases
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
        else:
            # Try to get feature_matrix, fall back to state_features
            if hasattr(reward_fn, 'feature_matrix'):
                feature_matrix = reward_fn.feature_matrix
            elif hasattr(reward_fn, 'state_features'):
                feature_matrix = reward_fn.state_features
            else:
                raise ValueError(f"Unsupported reward function type: {type(reward_fn)}")

        total_obs = sum(len(traj) for traj in panel.trajectories)

        if feature_matrix.ndim == 3:
            # Action-dependent: (n_states, n_actions, n_features)
            # E[φ] = (1/N) Σ_{i,t} Σ_a π(a|s_{i,t}) * φ(s_{i,t}, a, k)
            # Vectorized: gather all states, then batch multiply
            all_states = torch.cat([traj.states for traj in panel.trajectories])
            # policy[all_states] → (N, n_actions), feature_matrix[all_states] → (N, n_actions, n_features)
            # Sum over actions and observations
            feature_sum = (policy[all_states].unsqueeze(-1) * feature_matrix[all_states]).sum(dim=(0, 1))

            if total_obs > 0:
                return feature_sum / total_obs
            return feature_sum
        else:
            # State-only: (n_states, n_features)
            # Use state visitation frequencies under the current policy
            n_states = feature_matrix.shape[0]

            if transitions is not None and initial_dist is not None:
                # Create a temporary DDCProblem for the existing _compute_state_visitation
                problem = DDCProblem(
                    num_states=n_states,
                    num_actions=policy.shape[1],
                    discount_factor=discount,
                )
                d = self._compute_state_visitation(
                    policy, transitions, problem, initial_dist
                )
                # E_π[φ] = Σ_s d_π(s) * φ(s)
                return d @ feature_matrix
            else:
                # Fallback: iterate over empirical states
                all_states = panel.get_all_states()
                feature_sum = feature_matrix[all_states, :].sum(dim=0)
                if total_obs > 0:
                    return feature_sum / total_obs
                return feature_sum

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> torch.Tensor:
        """Compute initial state distribution from data."""
        counts = torch.zeros(n_states, dtype=torch.float32)
        init_states = torch.tensor(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=torch.long,
        )
        counts.scatter_add_(0, init_states, torch.ones_like(init_states, dtype=torch.float32))

        if counts.sum() > 0:
            return counts / counts.sum()
        return torch.ones(n_states) / n_states

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
        """Run MCE IRL optimization.

        Parameters
        ----------
        true_params : torch.Tensor, optional
            True parameters for debugging. If provided, RMSE is shown in progress bar.
        """
        start_time = time.time()

        reward_fn = utility
        operator = SoftBellmanOperator(problem, transitions)

        # Initialize parameters
        if initial_params is None:
            params = reward_fn.get_initial_parameters()
        else:
            params = initial_params.clone()

        # Compute empirical features (constant)
        empirical_features = self._compute_empirical_features(panel, reward_fn)
        initial_dist = self._compute_initial_distribution(panel, problem.num_states)

        self._log(f"Empirical features: {empirical_features}")
        self._log(f"Initial distribution entropy: {-(initial_dist * torch.log(initial_dist + 1e-10)).sum():.3f}")

        # Tracking
        n_function_evals = 0
        inner_not_converged = 0

        # Adam optimizer state
        if self.config.use_adam:
            m = torch.zeros_like(params)  # First moment
            v = torch.zeros_like(params)  # Second moment
            optimizer_name = "Adam"
        else:
            m = v = None
            optimizer_name = "SGD"

        # Run optimization using gradient ascent on log-likelihood
        # The gradient is: ∇L = μ_D - μ_π (for maximization)
        self._log(f"Starting MCE IRL with {optimizer_name} (lr={self.config.learning_rate})")

        best_obj = float('inf')
        best_params = params.clone()
        patience_counter = 0
        max_patience = 20

        # Use tqdm for progress tracking
        pbar = tqdm(
            range(self.config.outer_max_iter),
            desc="MCE IRL",
            disable=not self.config.verbose,
            leave=True,
        )

        for i in pbar:
            # Forward and backward passes
            reward_matrix = reward_fn.compute(params)
            V, policy, inner_converged = self._soft_value_iteration(operator, reward_matrix)
            if not inner_converged:
                inner_not_converged += 1

            expected_features = self._compute_expected_features(
                panel, policy, reward_fn,
                transitions=transitions,
                initial_dist=initial_dist,
                discount=problem.discount_factor,
            )

            # Feature matching gradient: μ_D - μ_π
            # We want to INCREASE params in direction where μ_D > μ_π
            gradient = empirical_features - expected_features  # Gradient ascent direction

            # Gradient clipping (prevents divergence when rewards approach zero)
            grad_norm = torch.norm(gradient).item()
            if self.config.gradient_clip > 0 and grad_norm > self.config.gradient_clip:
                gradient = gradient * (self.config.gradient_clip / grad_norm)
                grad_norm = self.config.gradient_clip

            # Objective: ||μ_D - μ_π||^2
            obj = 0.5 * torch.sum((empirical_features - expected_features) ** 2).item()

            # Track best
            if obj < best_obj:
                best_obj = obj
                best_params = params.clone()
                patience_counter = 0
            else:
                patience_counter += 1

            # Update progress bar with metrics
            postfix = {
                "obj": f"{obj:.6f}",
                "||grad||": f"{grad_norm:.4f}",
            }
            if true_params is not None:
                rmse = torch.sqrt(torch.mean((params - true_params) ** 2)).item()
                postfix["RMSE"] = f"{rmse:.6f}"
            pbar.set_postfix(postfix)

            # Check convergence
            if grad_norm < self.config.outer_tol:
                converged = True
                break

            if patience_counter > max_patience:
                pbar.set_description("MCE IRL (early stop)")
                break

            # Gradient step with Adam or SGD
            if self.config.use_adam:
                # Adam update (Kingma & Ba, 2014)
                t = i + 1  # Timestep (1-indexed)
                m = self.config.adam_beta1 * m + (1 - self.config.adam_beta1) * gradient
                v = self.config.adam_beta2 * v + (1 - self.config.adam_beta2) * (gradient ** 2)
                # Bias correction
                m_hat = m / (1 - self.config.adam_beta1 ** t)
                v_hat = v / (1 - self.config.adam_beta2 ** t)
                # Update
                params = params + self.config.learning_rate * m_hat / (torch.sqrt(v_hat) + self.config.adam_eps)
            else:
                # Simple SGD
                params = params + self.config.learning_rate * gradient

            n_function_evals += 1

        pbar.close()
        final_params = best_params
        converged = grad_norm < self.config.outer_tol if grad_norm else False

        # Final solution
        reward_matrix = reward_fn.compute(final_params)
        V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
        D = self._compute_state_visitation(policy, transitions, problem, initial_dist)

        # Feature difference for diagnostics
        final_expected = self._compute_expected_features(panel, policy, reward_fn)
        feature_diff = torch.norm(empirical_features - final_expected).item()

        # Log-likelihood
        log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
        ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()

        # Inference
        hessian = None
        gradient_contributions = None
        standard_errors = None

        if self.config.compute_se:
            self._log(f"Computing standard errors via {self.config.se_method}")

            if self.config.se_method == "bootstrap":
                standard_errors = self._bootstrap_inference(
                    panel, reward_fn, problem, transitions, final_params, initial_dist
                )
            else:
                # Numerical Hessian
                hessian = self._numerical_hessian(
                    final_params, panel, reward_fn, problem, transitions, initial_dist
                )

        optimization_time = time.time() - start_time

        self._log(f"Optimization complete: feature_diff={feature_diff:.6f}, LL={ll:.2f}")
        if inner_not_converged > 0:
            self._log(f"Warning: Inner loop did not converge {inner_not_converged} times")

        return EstimationResult(
            parameters=final_params,
            log_likelihood=ll,
            value_function=V,
            policy=policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=n_function_evals,
            num_function_evals=n_function_evals,
            num_inner_iterations=0,
            message="",
            optimization_time=optimization_time,
            metadata={
                "optimizer": self.config.optimizer,
                "empirical_features": empirical_features.tolist(),
                "final_expected_features": final_expected.tolist(),
                "feature_difference": feature_diff,
                "state_visitation": D.tolist(),
                "inner_not_converged": inner_not_converged,
                "standard_errors": standard_errors.tolist() if standard_errors is not None else None,
            },
        )

    def _bootstrap_inference(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        point_estimate: torch.Tensor,
        initial_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Compute standard errors via bootstrap.

        Resamples trajectories and re-estimates parameters to get
        sampling distribution.
        """
        n_params = len(point_estimate)
        bootstrap_estimates = torch.zeros((self.config.n_bootstrap, n_params))

        trajectories = panel.trajectories
        n_traj = len(trajectories)

        operator = SoftBellmanOperator(problem, transitions)

        for b in range(self.config.n_bootstrap):
            # Resample trajectories with replacement
            indices = np.random.choice(n_traj, size=n_traj, replace=True)
            boot_trajectories = [trajectories[i] for i in indices]
            boot_panel = Panel(trajectories=boot_trajectories)

            # Compute empirical features for bootstrap sample
            empirical_features = self._compute_empirical_features(boot_panel, reward_fn)
            boot_initial = self._compute_initial_distribution(boot_panel, problem.num_states)

            # Quick optimization from point estimate
            params = point_estimate.clone()

            for _ in range(50):  # Fewer iterations for bootstrap
                reward_matrix = reward_fn.compute(params)
                V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
                expected_features = self._compute_expected_features(boot_panel, policy, reward_fn)

                gradient = expected_features - empirical_features
                params = params - 0.1 * gradient

                if torch.norm(gradient) < 0.01:
                    break

            bootstrap_estimates[b] = params

            if self.config.verbose and (b + 1) % 20 == 0:
                self._log(f"Bootstrap {b + 1}/{self.config.n_bootstrap}")

        # Standard errors = std of bootstrap estimates
        standard_errors = bootstrap_estimates.std(dim=0)

        return standard_errors

    def _numerical_hessian(
        self,
        params: torch.Tensor,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_dist: torch.Tensor,
        eps: float = 1e-3,
    ) -> torch.Tensor:
        """Compute numerical Hessian of the log-likelihood.

        Uses central differences with adaptive step size for numerical stability.
        The Hessian at a maximum should be negative semi-definite, so we ensure
        the returned Hessian has this property by projecting onto the negative
        semi-definite cone if needed.

        Parameters
        ----------
        params : torch.Tensor
            Parameter vector at which to compute Hessian.
        panel : Panel
            Panel data for computing log-likelihood.
        reward_fn : LinearReward
            Reward function specification.
        problem : DDCProblem
            Problem specification.
        transitions : torch.Tensor
            Transition matrices.
        initial_dist : torch.Tensor
            Initial state distribution (unused but kept for API consistency).
        eps : float
            Step size for finite differences. Default 1e-3 for stability.

        Returns
        -------
        torch.Tensor
            Hessian matrix, shape (n_params, n_params).
            Guaranteed to be negative semi-definite for valid inference.

        Notes
        -----
        Step size selection follows Gill, Murray, and Wright (1981) guidance:
        - Use adaptive step h_i = max(eps, min(|params[i]| * eps, 0.1))
        - Lower bound (eps) prevents division by zero for zero-valued parameters
        - Upper bound (0.1) prevents excessively large steps for large parameters
          that would introduce truncation error and numerical instability
        - The default eps=1e-3 balances truncation error (favors larger h)
          against rounding error (favors smaller h) for float32 precision
        """
        operator = SoftBellmanOperator(problem, transitions)
        n_params = len(params)
        hessian = torch.zeros((n_params, n_params), dtype=params.dtype)

        def ll_at(p):
            reward_matrix = reward_fn.compute(p)
            V, policy, _ = self._soft_value_iteration(operator, reward_matrix)
            log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)

            ll = log_probs[panel.get_all_states(), panel.get_all_actions()].sum().item()
            return ll

        # Compute Hessian using central differences
        # Use adaptive step size based on parameter magnitude with bounds
        for i in range(n_params):
            # Adaptive step: larger for larger params, bounded between eps and 0.1
            h_i = max(eps, min(abs(params[i].item()) * eps, 0.1))

            for j in range(i, n_params):
                h_j = max(eps, min(abs(params[j].item()) * eps, 0.1))

                if i == j:
                    # Diagonal: use standard 2nd derivative formula
                    # f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h^2
                    p_plus = params.clone()
                    p_plus[i] += h_i
                    p_minus = params.clone()
                    p_minus[i] -= h_i

                    ll_plus = ll_at(p_plus)
                    ll_0 = ll_at(params)
                    ll_minus = ll_at(p_minus)

                    h_ii = (ll_plus - 2 * ll_0 + ll_minus) / (h_i * h_i)
                    hessian[i, i] = h_ii
                else:
                    # Off-diagonal: use 4-point formula for mixed partial
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

        # Ensure Hessian is negative semi-definite (required at a maximum)
        # If not, project onto negative semi-definite cone
        eigenvalues, eigenvectors = torch.linalg.eigh(hessian)
        if (eigenvalues > 0).any():
            # Some eigenvalues are positive - not at a proper maximum
            # Project to negative semi-definite by clamping positive eigenvalues
            self._log("Warning: Hessian not negative semi-definite, projecting")
            eigenvalues_clamped = torch.clamp(eigenvalues, max=-1e-8)
            hessian = eigenvectors @ torch.diag(eigenvalues_clamped) @ eigenvectors.T

        return hessian
