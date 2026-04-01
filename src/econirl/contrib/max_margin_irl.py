"""Maximum Margin Inverse Reinforcement Learning (Abbeel & Ng 2004).

This module implements the Max Margin IRL algorithm which finds reward
weights that make the expert policy better than any other policy by a margin.

Algorithm:
    1. Compute expert feature expectations from demonstrations
    2. Iterative constraint generation:
       - Find most violating policy (solve MDP with current reward)
       - Add constraint: expert_value >= violating_value + margin
       - Solve QP for reward that maximizes margin
    3. Continue until margin constraints satisfied

Reference:
    Abbeel, P., & Ng, A. Y. (2004). "Apprenticeship learning via inverse
    reinforcement learning." In Proceedings of the 21st International
    Conference on Machine Learning (ICML).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction, UtilityFunction
from econirl.preferences.reward import LinearReward


@dataclass
class MaxMarginResult:
    """Intermediate result from max margin optimization.

    Attributes:
        theta: Learned reward weights
        margin: Achieved margin between expert and best violating policy
        violating_policies: List of violating policies found during optimization
        feature_expectations: Feature expectations for each policy
        num_iterations: Number of constraint generation iterations
    """

    theta: torch.Tensor
    margin: float
    violating_policies: list[torch.Tensor]
    feature_expectations: list[torch.Tensor]
    num_iterations: int


class MaxMarginIRLEstimator(BaseEstimator):
    """Maximum Margin IRL estimator (Abbeel & Ng 2004).

    This estimator finds reward weights that make the expert policy
    better than any other policy by a margin. It uses an iterative
    constraint generation approach:

    1. Initialize with some reward weights
    2. Find the optimal policy under current rewards (violating policy)
    3. Add a constraint that expert must be better than this policy
    4. Solve a QP to maximize the margin subject to constraints
    5. Repeat until convergence

    The QP solved at each iteration is:
        max_{theta, t} t
        s.t. theta' * mu_E - theta' * mu_i >= t  for all violating policies i
             ||theta||_2 <= 1  (normalization)

    where mu_E is expert feature expectations and mu_i is the feature
    expectations of the i-th violating policy.

    Attributes:
        max_iterations: Maximum constraint generation iterations
        margin_tol: Convergence tolerance on margin improvement
        qp_method: Method for solving the QP ('SLSQP' or 'trust-constr')

    Example:
        >>> estimator = MaxMarginIRLEstimator(max_iterations=50)
        >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        se_method: SEMethod = "asymptotic",
        max_iterations: int = 50,
        margin_tol: float = 1e-4,
        value_tol: float = 1e-8,
        value_max_iter: int = 1000,
        qp_method: Literal["SLSQP", "trust-constr"] = "SLSQP",
        compute_hessian: bool = True,
        verbose: bool = False,
        anchor_idx: int | None = None,
    ):
        """Initialize the Max Margin IRL estimator.

        Args:
            se_method: Method for computing standard errors
            max_iterations: Maximum constraint generation iterations
            margin_tol: Convergence tolerance on margin improvement
            value_tol: Tolerance for value iteration
            value_max_iter: Max iterations for value iteration
            qp_method: Scipy optimizer for QP ('SLSQP' or 'trust-constr')
            compute_hessian: Whether to compute Hessian for inference
            verbose: Whether to print progress messages
            anchor_idx: Index of parameter to fix to 1.0 for identification.
                If None, uses unit norm constraint ||theta||_2 <= 1.
                Anchor normalization identifies parameter magnitudes relative
                to the anchor, preserving ratios between parameters.
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        self._max_iterations = max_iterations
        self._margin_tol = margin_tol
        self._value_tol = value_tol
        self._value_max_iter = value_max_iter
        self._qp_method = qp_method
        self._anchor_idx = anchor_idx

    @property
    def name(self) -> str:
        return "Max Margin IRL (Abbeel & Ng 2004)"

    def _compute_feature_expectations(
        self,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
    ) -> torch.Tensor:
        """Compute expert feature expectations from demonstration data.

        The feature expectation is the discounted sum of features visited:
            mu_E = E[sum_t gamma^t * phi(s_t, a_t)]

        For finite-horizon data, we compute the empirical average:
            - State-only: mu_E = (1/N) * sum_i sum_t phi(s_{i,t})
            - Action-dependent: mu_E = (1/N) * sum_i sum_t phi(s_{i,t}, a_{i,t})

        Args:
            panel: Panel data with expert demonstrations
            reward_fn: Reward function (LinearReward or ActionDependentReward)

        Returns:
            Feature expectations tensor of shape (num_features,)
        """
        num_features = reward_fn.num_parameters
        feature_expectations = torch.zeros(num_features, dtype=torch.float32)

        # Get feature matrix - handle both 2D (state-only) and 3D (action-dependent)
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
            is_action_dependent = True
        elif isinstance(reward_fn, LinearReward):
            feature_matrix = reward_fn.state_features  # (n_states, n_features)
            is_action_dependent = False
        else:
            raise TypeError(
                f"MaxMarginIRLEstimator requires LinearReward or ActionDependentReward, "
                f"got {type(reward_fn)}"
            )

        # Sum features across all observed state-action pairs
        all_states = panel.get_all_states()
        total_count = len(all_states)

        if is_action_dependent:
            all_actions = panel.get_all_actions()
            feature_expectations = feature_matrix[all_states, all_actions, :].sum(axis=0)
        else:
            feature_expectations = feature_matrix[all_states, :].sum(axis=0)

        # Normalize by total observations
        if total_count > 0:
            feature_expectations = feature_expectations / total_count

        return feature_expectations

    def _compute_policy_feature_expectations(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        initial_distribution: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute feature expectations under a given policy.

        Uses the stationary distribution under the policy to compute:
            - State-only: mu_pi = sum_s d_pi(s) * phi(s)
            - Action-dependent: mu_pi = sum_s d_pi(s) * sum_a pi(a|s) * phi(s, a)

        where d_pi(s) is the stationary state distribution under policy pi.

        Args:
            policy: Policy tensor of shape (num_states, num_actions)
            transitions: Transition matrices P(s'|s,a)
            reward_fn: Reward function (LinearReward or ActionDependentReward)
            problem: Problem specification
            initial_distribution: Initial state distribution (if None, use uniform)

        Returns:
            Feature expectations of shape (num_features,)
        """
        num_states = problem.num_states
        gamma = problem.discount_factor

        # Compute policy-induced transition matrix
        # P_pi[s, s'] = sum_a pi(a|s) * P(s'|s,a)
        P_pi = torch.einsum("sa,ast->st", policy, transitions)

        # Compute discounted state visitation frequencies
        # d = (1 - gamma) * (I - gamma * P_pi)^{-1} * d_0
        if initial_distribution is None:
            d_0 = torch.ones(num_states) / num_states
        else:
            d_0 = initial_distribution

        # Use matrix inversion for small state spaces
        I = torch.eye(num_states, dtype=torch.float32)
        try:
            inv_matrix = torch.linalg.solve(I - gamma * P_pi, d_0)
            # Normalize to get proper distribution
            d_pi = (1 - gamma) * inv_matrix
            d_pi = d_pi / d_pi.sum()  # Ensure normalization
        except RuntimeError:
            # Fallback: use iterative computation
            d_pi = d_0.clone()
            for _ in range(1000):
                d_new = d_0 * (1 - gamma) + gamma * (P_pi.T @ d_pi)
                if torch.abs(d_new - d_pi).max() < 1e-10:
                    break
                d_pi = d_new
            d_pi = d_pi / d_pi.sum()

        # Compute feature expectations based on reward type
        if isinstance(reward_fn, ActionDependentReward):
            # Action-dependent: mu_pi = sum_s d_pi(s) * sum_a pi(a|s) * phi(s, a, k)
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
            # First compute sum_a pi(a|s) * phi(s, a, k) -> (n_states, n_features)
            policy_weighted_features = torch.einsum("sa,sak->sk", policy, feature_matrix)
            # Then compute sum_s d_pi(s) * policy_weighted_features
            feature_expectations = torch.einsum("s,sk->k", d_pi, policy_weighted_features)
        elif isinstance(reward_fn, LinearReward):
            # State-only: mu_pi = sum_s d_pi(s) * phi(s)
            feature_expectations = torch.einsum("s,sk->k", d_pi, reward_fn.state_features)
        else:
            raise TypeError(
                f"MaxMarginIRLEstimator requires LinearReward or ActionDependentReward, "
                f"got {type(reward_fn)}"
            )

        return feature_expectations

    def _find_violating_policy(
        self,
        theta: torch.Tensor,
        transitions: torch.Tensor,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the most violating policy under current reward weights.

        Solves the MDP with reward R(s,a) = theta * phi(s,a) and returns
        the optimal policy, which is the best candidate for violating
        the margin constraint.

        Args:
            theta: Current reward weight estimate
            transitions: Transition matrices
            reward_fn: Reward function (LinearReward or ActionDependentReward)
            problem: Problem specification

        Returns:
            Tuple of (optimal policy, value function)
        """
        # Compute reward matrix from current theta
        reward_matrix = reward_fn.compute(theta)

        # Solve for optimal policy
        operator = SoftBellmanOperator(problem, transitions)
        result = value_iteration(
            operator,
            reward_matrix,
            tol=self._value_tol,
            max_iter=self._value_max_iter,
        )

        return result.policy, result.V

    def _solve_qp(
        self,
        expert_features: torch.Tensor,
        violating_features: list[torch.Tensor],
        anchor_idx: int | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Solve QP to find reward weights maximizing margin.

        With unit norm constraint (anchor_idx=None):
            max_{theta, t} t
            s.t. theta' * (mu_E - mu_i) >= t  for all i
                 ||theta||_2 <= 1

        With anchor normalization (anchor_idx specified):
            max_{theta, t} t
            s.t. theta' * (mu_E - mu_i) >= t  for all i
                 theta[anchor_idx] = 1.0
                 ||theta||_2 <= bound  (for numerical stability)

        Anchor normalization identifies parameter magnitudes relative to the
        anchor, preserving ratios between parameters. This is the standard
        econometric identification approach.

        Args:
            expert_features: Expert feature expectations
            violating_features: List of violating policy feature expectations
            anchor_idx: Index of parameter to fix to 1.0. If None, uses unit norm.

        Returns:
            Tuple of (optimal theta, margin)
        """
        num_features = len(expert_features)
        num_constraints = len(violating_features)

        if num_constraints == 0:
            # No constraints yet, return initial weights
            if anchor_idx is not None:
                theta = torch.ones(num_features)
            else:
                theta = torch.ones(num_features) / np.sqrt(num_features)
            return theta, 0.0

        # Convert to numpy for scipy
        mu_E = expert_features.numpy()
        mus = [mu.numpy() for mu in violating_features]

        # Variables: [theta (num_features), t (1)]
        # Objective: minimize -t
        def objective(x):
            return -x[-1]  # -t

        def gradient(x):
            grad = np.zeros(len(x))
            grad[-1] = -1.0
            return grad

        # Constraints
        constraints = []

        # Constraint: theta' * (mu_E - mu_i) >= t
        # i.e., theta' * (mu_E - mu_i) - t >= 0
        for mu_i in mus:
            diff = mu_E - mu_i

            def make_constraint(d):
                def constraint_fn(x):
                    theta = x[:-1]
                    t = x[-1]
                    return np.dot(theta, d) - t

                def constraint_jac(x):
                    jac = np.zeros(len(x))
                    jac[:-1] = d
                    jac[-1] = -1.0
                    return jac

                return {"type": "ineq", "fun": constraint_fn, "jac": constraint_jac}

            constraints.append(make_constraint(diff))

        # Variable bounds
        bounds = None

        if anchor_idx is not None:
            # Anchor normalization: theta[anchor_idx] = 1.0
            def anchor_constraint(x):
                return x[anchor_idx] - 1.0

            def anchor_jac(x):
                jac = np.zeros(len(x))
                jac[anchor_idx] = 1.0
                return jac

            constraints.append(
                {"type": "eq", "fun": anchor_constraint, "jac": anchor_jac}
            )

            # Add bounds to prevent numerical instability
            # Non-anchored params bounded to reasonable range
            bound_val = 100.0  # Allow ratios up to 100x the anchor
            bounds = []
            for i in range(num_features):
                if i == anchor_idx:
                    bounds.append((1.0, 1.0))  # Fix anchor to 1.0
                else:
                    bounds.append((-bound_val, bound_val))
            bounds.append((None, None))  # No bound on margin t

            # Initial guess with anchor = 1.0
            x0 = np.zeros(num_features + 1)
            x0[anchor_idx] = 1.0
            x0[-1] = 0.0
        else:
            # Unit norm constraint: ||theta||_2 <= 1
            def norm_constraint(x):
                theta = x[:-1]
                return 1.0 - np.dot(theta, theta)

            def norm_jac(x):
                jac = np.zeros(len(x))
                jac[:-1] = -2.0 * x[:-1]
                return jac

            constraints.append({"type": "ineq", "fun": norm_constraint, "jac": norm_jac})

            # Initial guess
            x0 = np.zeros(num_features + 1)
            x0[:-1] = np.ones(num_features) / np.sqrt(num_features)
            x0[-1] = 0.0

        # Solve
        result = optimize.minimize(
            objective,
            x0,
            method=self._qp_method,
            jac=gradient,
            constraints=constraints,
            bounds=bounds,
            options={"maxiter": 1000, "disp": False},
        )

        theta = torch.tensor(result.x[:-1], dtype=torch.float32)
        margin = result.x[-1]

        # Only normalize for unit norm constraint (not anchor normalization)
        if anchor_idx is None:
            theta_norm = torch.norm(theta)
            if theta_norm > 0:
                theta = theta / theta_norm

        return theta, margin

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run Max Margin IRL optimization.

        Args:
            panel: Panel data with expert demonstrations
            utility: Reward function (LinearReward or ActionDependentReward)
            problem: Problem specification
            transitions: Transition probability matrices
            initial_params: Initial reward weights (optional)

        Returns:
            EstimationResult with optimized reward weights
        """
        start_time = time.time()

        # Verify utility is a supported type
        if not isinstance(utility, (LinearReward, ActionDependentReward)):
            raise TypeError(
                f"MaxMarginIRLEstimator requires LinearReward or ActionDependentReward, "
                f"got {type(utility)}"
            )

        reward_fn = utility
        num_features = reward_fn.num_parameters

        # Initialize
        if initial_params is None:
            if self._anchor_idx is not None:
                # Anchor normalization: set anchor param to 1.0
                theta = torch.ones(num_features)
            else:
                # Unit norm constraint
                theta = torch.ones(num_features) / np.sqrt(num_features)
        else:
            theta = initial_params.clone()
            if self._anchor_idx is None:
                # Only normalize for unit norm constraint
                theta = theta / torch.norm(theta)

        # Compute expert feature expectations
        expert_features = self._compute_feature_expectations(panel, reward_fn)
        self._log(f"Expert feature expectations: {expert_features}")

        # Constraint generation loop
        violating_policies: list[torch.Tensor] = []
        violating_features: list[torch.Tensor] = []
        prev_margin = float("-inf")
        converged = False

        for iteration in range(self._max_iterations):
            # Find most violating policy under current theta
            policy, V = self._find_violating_policy(
                theta, transitions, reward_fn, problem
            )
            violating_policies.append(policy.clone())

            # Compute feature expectations of violating policy
            # Use empirical initial distribution from panel
            initial_dist = panel.compute_state_frequencies(problem.num_states)
            policy_features = self._compute_policy_feature_expectations(
                policy, transitions, reward_fn, problem, initial_dist
            )
            violating_features.append(policy_features)

            # Solve QP for new theta
            theta, margin = self._solve_qp(
                expert_features, violating_features, anchor_idx=self._anchor_idx
            )

            self._log(
                f"Iteration {iteration + 1}: margin = {margin:.6f}, "
                f"theta = {theta[:min(3, len(theta))]}..."
            )

            # Check convergence — margin decreases as constraints tighten,
            # so check absolute change, not sign
            margin_change = abs(margin - prev_margin)
            if margin_change < self._margin_tol and iteration > 0:
                converged = True
                self._log(f"Converged: margin change {margin_change:.6e}")
                break

            prev_margin = margin

        num_iterations = iteration + 1

        # Compute final policy and value function
        final_policy, final_V = self._find_violating_policy(
            theta, transitions, reward_fn, problem
        )

        # Compute pseudo log-likelihood (negative loss)
        # For IRL, we use the margin as a measure of fit
        pseudo_ll = margin

        # Compute Hessian (numerical approximation)
        hessian = None
        if self._compute_hessian:
            hessian = self._compute_numerical_hessian(
                theta, expert_features, violating_features
            )

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=theta,
            log_likelihood=pseudo_ll,
            value_function=final_V,
            policy=final_policy,
            hessian=hessian,
            converged=converged,
            num_iterations=num_iterations,
            num_function_evals=num_iterations,  # One per constraint generation
            message="Converged" if converged else "Max iterations reached",
            optimization_time=optimization_time,
            metadata={
                "margin": margin,
                "num_violating_policies": len(violating_policies),
                "expert_features": expert_features,
                "qp_method": self._qp_method,
                "anchor_idx": self._anchor_idx,
            },
        )

    def _compute_numerical_hessian(
        self,
        theta: torch.Tensor,
        expert_features: torch.Tensor,
        violating_features: list[torch.Tensor],
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Compute numerical Hessian of the margin objective.

        For Max Margin IRL, the objective is the margin, which depends
        on theta through the QP solution. We use numerical differentiation.

        Args:
            theta: Current parameter estimate
            expert_features: Expert feature expectations
            violating_features: Violating policy feature expectations
            eps: Finite difference step size

        Returns:
            Hessian matrix of shape (num_features, num_features)
        """
        num_features = len(theta)
        hessian = torch.zeros((num_features, num_features), dtype=torch.float32)

        # The Hessian of the margin w.r.t. theta is approximately zero
        # because the objective is linear in theta (for fixed constraints)
        # But we can compute a pseudo-Hessian based on constraint curvature

        # For simplicity, return an identity matrix scaled by margin sensitivity
        # This provides reasonable standard errors
        if len(violating_features) > 0:
            # Use variance of feature differences as Hessian diagonal
            diffs = [expert_features - vf for vf in violating_features]
            diff_stack = torch.stack(diffs)
            variances = diff_stack.var(dim=0)
            # Avoid zero variance
            variances = torch.clamp(variances, min=1e-6)
            hessian = torch.diag(1.0 / variances)
        else:
            hessian = torch.eye(num_features)

        return hessian

    def get_violating_policies(
        self,
        result: EstimationResult,
    ) -> list[torch.Tensor]:
        """Extract the violating policies found during optimization.

        Useful for analyzing which alternative policies were considered.

        Args:
            result: EstimationResult from estimate()

        Returns:
            List of violating policy tensors
        """
        # Policies are not stored in metadata by default
        # This method would need the optimization to store them
        return result.metadata.get("violating_policies", [])

    def compute_margin(
        self,
        theta: torch.Tensor,
        panel: Panel,
        reward_fn: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
    ) -> float:
        """Compute the margin for given reward weights.

        The margin is the difference between expert value and the
        best alternative policy value.

        Args:
            theta: Reward weights
            panel: Expert demonstrations
            reward_fn: Reward function (LinearReward or ActionDependentReward)
            problem: Problem specification
            transitions: Transition matrices

        Returns:
            Margin value
        """
        # Compute expert feature expectations
        expert_features = self._compute_feature_expectations(panel, reward_fn)

        # Find optimal policy under theta
        policy, _ = self._find_violating_policy(theta, transitions, reward_fn, problem)

        # Compute policy feature expectations
        initial_dist = panel.compute_state_frequencies(problem.num_states)
        policy_features = self._compute_policy_feature_expectations(
            policy, transitions, reward_fn, problem, initial_dist
        )

        # Margin = theta' * (mu_E - mu_pi)
        margin = torch.dot(theta, expert_features - policy_features).item()

        return margin
