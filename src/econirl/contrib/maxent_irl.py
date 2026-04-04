"""Maximum Entropy Inverse Reinforcement Learning (MaxEnt IRL) estimator.

This module implements the MaxEnt IRL algorithm from Ziebart et al. (2008)
for recovering reward functions from observed behavior.

Algorithm:
    1. Compute empirical feature expectations from demonstrations
    2. Iteratively:
       - Solve soft Bellman for V under current reward
       - Compute policy from V
       - Compute expected features under policy
       - Gradient = empirical_features - expected_features
       - Update parameters via L-BFGS

The MaxEnt objective is:
    max_theta E_D[R(s; theta)] - log Z(theta)

where D is the demonstration distribution and Z is the partition function.
This is equivalent to matching feature expectations while maximizing
policy entropy.

Reference:
    Ziebart, B. D., Maas, A. L., Bagnell, J. A., & Dey, A. K. (2008).
    "Maximum entropy inverse reinforcement learning."
    AAAI Conference on Artificial Intelligence.
"""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import value_iteration, policy_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.base import BaseUtilityFunction, UtilityFunction
from econirl.preferences.reward import LinearReward


class MaxEntIRLEstimator(BaseEstimator):
    """Maximum Entropy Inverse Reinforcement Learning estimator.

    Recovers reward function parameters from demonstrated behavior using
    the maximum entropy principle. The algorithm finds reward parameters
    that make the demonstrated behavior have maximum entropy subject to
    matching empirical feature expectations.

    The key insight is that the gradient of the log-likelihood has a
    simple form:
        gradient = empirical_features - expected_features

    This allows efficient optimization via L-BFGS.

    Attributes:
        optimizer: Scipy optimizer to use
        inner_tol: Convergence tolerance for value iteration
        outer_tol: Convergence tolerance for outer optimization
        outer_max_iter: Maximum iterations for outer optimization

    Example:
        >>> # Create state features for IRL
        >>> features = jnp.array(np.random.randn(100, 5))
        >>> reward_fn = LinearReward(
        ...     state_features=features,
        ...     parameter_names=["f1", "f2", "f3", "f4", "f5"],
        ...     n_actions=2,
        ... )
        >>>
        >>> estimator = MaxEntIRLEstimator(verbose=True)
        >>> result = estimator.estimate(panel, reward_fn, problem, transitions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        se_method: SEMethod = "asymptotic",
        optimizer: Literal["L-BFGS-B", "BFGS"] = "L-BFGS-B",
        inner_solver: Literal["value", "policy"] = "policy",
        inner_tol: float = 1e-10,
        inner_max_iter: int = 1000,
        outer_tol: float = 1e-6,
        outer_max_iter: int = 200,
        learning_rate: float = 1.0,
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the MaxEnt IRL estimator.

        Args:
            se_method: Method for computing standard errors
            optimizer: Scipy optimizer for outer loop
            inner_solver: Solver for inner loop ("value" or "policy")
            inner_tol: Tolerance for inner loop convergence
            inner_max_iter: Max iterations for inner loop
            outer_tol: Tolerance for outer optimization convergence
            outer_max_iter: Max iterations for outer optimization
            learning_rate: Learning rate for gradient updates (used with L-BFGS)
            compute_hessian: Whether to compute Hessian for inference
            verbose: Whether to print progress messages
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        self._optimizer = optimizer
        self._inner_solver = inner_solver
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._outer_tol = outer_tol
        self._outer_max_iter = outer_max_iter
        self._learning_rate = learning_rate

    @property
    def name(self) -> str:
        return "MaxEnt IRL (Ziebart 2008)"

    def _solve_inner(
        self,
        operator: SoftBellmanOperator,
        reward_matrix: jnp.ndarray,
    ):
        """Solve the inner dynamic programming problem."""
        if self._inner_solver == "policy":
            return policy_iteration(
                operator,
                reward_matrix,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
                eval_method="matrix",
            )
        else:
            return value_iteration(
                operator,
                reward_matrix,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
            )

    def _compute_empirical_features(
        self,
        panel: Panel,
        reward_fn,
    ) -> jnp.ndarray:
        """Compute empirical feature expectations from demonstrations.

        Handles both state-only (2D) and action-dependent (3D) features.
        """
        if isinstance(reward_fn, ActionDependentReward):
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
            n_features = feature_matrix.shape[2]
            all_states = jnp.concatenate([traj.states for traj in panel.trajectories])
            all_actions = jnp.concatenate([traj.actions for traj in panel.trajectories])
            feature_sum = feature_matrix[all_states, all_actions, :].sum(axis=0)
            total_obs = len(all_states)
            return feature_sum / total_obs if total_obs > 0 else feature_sum
        else:
            state_features = reward_fn.state_features  # (num_states, num_features)
            all_states = jnp.concatenate([traj.states for traj in panel.trajectories])
            feature_sum = state_features[all_states].sum(axis=0)
            total_obs = len(all_states)
            return feature_sum / total_obs if total_obs > 0 else feature_sum

    def _compute_state_visitation_frequency(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        initial_distribution: jnp.ndarray | None = None,
        horizon: int = 100,
    ) -> jnp.ndarray:
        """Compute expected state visitation frequencies under policy.

        Uses forward message passing to compute:
            mu_t(s) = sum_{s'} mu_{t-1}(s') sum_a pi(a|s') P(s|s',a)

        Then averages over time: mu(s) = (1/T) sum_t mu_t(s)

        Args:
            policy: Choice probabilities, shape (num_states, num_actions)
            transitions: Transition matrices, shape (num_actions, num_states, num_states)
            problem: DDC problem specification
            initial_distribution: Initial state distribution. If None, uses uniform.
            horizon: Number of time steps for forward computation

        Returns:
            State visitation frequencies, shape (num_states,)
        """
        num_states = problem.num_states
        num_actions = problem.num_actions
        beta = problem.discount_factor

        # Initial state distribution
        if initial_distribution is None:
            mu = jnp.ones(num_states, dtype=policy.dtype) / num_states
        else:
            mu = jnp.array(initial_distribution)

        # Accumulate visitation frequencies
        visitation = jnp.array(mu)

        # Forward pass
        for t in range(1, horizon):
            # Compute policy-weighted transition matrix: P_pi(s'|s) = sum_a pi(a|s) P(s'|s,a)
            # transitions: (num_actions, num_states, num_states) = [a, from_s, to_s]
            # policy: (num_states, num_actions) = [s, a]
            P_pi = jnp.einsum("sa,ast->st", policy, transitions)

            # Update: mu_new(s') = sum_s mu(s) P_pi(s'|s)
            mu_new = jnp.einsum("s,st->t", mu, P_pi)

            # Discount and accumulate
            visitation = visitation + (beta ** t) * mu_new
            mu = mu_new

        # Normalize to get average visitation frequency
        visitation = visitation / visitation.sum()

        return visitation

    def _compute_expected_features(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        reward_fn,
        problem: DDCProblem,
        panel: Panel | None = None,
    ) -> jnp.ndarray:
        """Compute expected feature expectations under the current policy.

        Handles both state-only (2D) and action-dependent (3D) features.
        """
        # Compute initial state distribution from data if available
        initial_distribution = None
        if panel is not None:
            initial_counts = jnp.zeros(problem.num_states, dtype=jnp.float32)
            init_states = jnp.array(
                [int(traj.states[0]) for traj in panel.trajectories if len(traj) > 0],
                dtype=jnp.int64,
            )
            initial_counts = initial_counts.at[init_states].add(jnp.ones_like(init_states, dtype=jnp.float32))
            if initial_counts.sum() > 0:
                initial_distribution = initial_counts / initial_counts.sum()

        # Compute state visitation frequency
        visitation = self._compute_state_visitation_frequency(
            policy,
            transitions,
            problem,
            initial_distribution=initial_distribution,
        )

        if isinstance(reward_fn, ActionDependentReward):
            # 3D: E_π[φ] = Σ_s d(s) Σ_a π(a|s) φ(s,a)
            feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
            return jnp.einsum("s,sa,sak->k", visitation, policy, feature_matrix)
        else:
            # 2D: E_π[φ] = Σ_s d(s) φ(s)
            state_features = reward_fn.state_features
            return jnp.einsum("s,sk->k", visitation, state_features)

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run MaxEnt IRL optimization.

        The algorithm maximizes the log-likelihood:
            L(theta) = E_D[R(s; theta)] - log Z(theta)

        The gradient is:
            dL/dtheta = empirical_features - expected_features

        Args:
            panel: Panel data with observed demonstrations
            utility: Utility/reward function specification (should be LinearReward)
            problem: Problem specification
            transitions: Transition probability matrices
            initial_params: Starting values (defaults to zeros)

        Returns:
            EstimationResult with optimized reward parameters
        """
        start_time = time.time()

        # Support both LinearReward (state-only) and ActionDependentReward (3D)
        if not isinstance(utility, (LinearReward, ActionDependentReward)):
            raise TypeError(
                f"MaxEntIRLEstimator requires LinearReward or ActionDependentReward, "
                f"got {type(utility).__name__}."
            )

        reward_fn = utility
        self._use_action_features = isinstance(utility, ActionDependentReward)

        # Initialize parameters
        if initial_params is None:
            initial_params = reward_fn.get_initial_parameters()

        # Create Bellman operator
        operator = SoftBellmanOperator(problem, transitions)

        # Compute empirical feature expectations (constant)
        empirical_features = self._compute_empirical_features(panel, reward_fn)
        self._log(f"Empirical features: {empirical_features}")

        # Tracking variables
        total_inner_iterations = 0
        num_function_evals = 0

        # Use L-BFGS-B with a CONSISTENT (f, ∇f) pair:
        #   f = -LL(θ) = -Σ_{i,t} log π(a_{i,t}|s_{i,t}; θ)  (negative log-likelihood)
        #   ∇f = expected_features - empirical_features  (gradient of NLL)
        # These are consistent because the MaxEnt IRL gradient IS ∂(-LL)/∂θ.
        self._log(f"Starting MaxEnt IRL optimization with L-BFGS-B")

        # Pure JAX objective: returns scalar NLL so jaxopt can differentiate it.
        def jax_objective(params):
            nonlocal total_inner_iterations, num_function_evals
            num_function_evals += 1

            reward_matrix = reward_fn.compute(params)
            solver_result = self._solve_inner(operator, reward_matrix)
            total_inner_iterations += solver_result.num_iterations

            log_probs = operator.compute_log_choice_probabilities(
                reward_matrix, solver_result.V
            )
            nll = -log_probs[panel.get_all_states(), panel.get_all_actions()].sum()
            return nll

        lower, upper = reward_fn.get_parameter_bounds()
        lower_jax = jnp.asarray(lower, dtype=jnp.float64)
        upper_jax = jnp.asarray(upper, dtype=jnp.float64)

        result_opt = minimize_lbfgsb(
            jax_objective,
            jnp.array(initial_params, dtype=jnp.float64),
            bounds=(lower_jax, upper_jax),
            maxiter=self._outer_max_iter,
            tol=self._outer_tol,
            verbose=self._verbose,
            desc="MaxEnt-IRL L-BFGS-B",
        )

        final_params = jnp.array(result_opt.x, dtype=jnp.float32)
        converged = result_opt.success

        # Compute final value function and policy
        reward_matrix = reward_fn.compute(final_params)
        solver_result = self._solve_inner(operator, reward_matrix)

        # Compute final feature difference for log-likelihood proxy
        final_expected = self._compute_expected_features(
            solver_result.policy,
            transitions,
            reward_fn,
            problem,
            panel,
        )
        feature_diff = float(jnp.linalg.norm(empirical_features - final_expected))

        # Compute log-likelihood (using CCP likelihood as proxy)
        log_probs = operator.compute_log_choice_probabilities(
            reward_matrix, solver_result.V
        )

        ll = float(log_probs[panel.get_all_states(), panel.get_all_actions()].sum())

        # Compute Hessian for standard errors
        hessian = None
        gradient_contributions = None

        if self._compute_hessian:
            self._log("Computing Hessian for standard errors")

            def ll_fn(params):
                """Log-likelihood function for Hessian computation."""
                reward_mat = reward_fn.compute(params)
                solver_res = self._solve_inner(operator, reward_mat)
                log_p = operator.compute_log_choice_probabilities(
                    reward_mat, solver_res.V
                )

                total_ll = float(log_p[panel.get_all_states(), panel.get_all_actions()].sum())
                return jnp.array(total_ll)

            hessian = compute_numerical_hessian(final_params, ll_fn)

            # Compute per-observation gradients for robust SEs
            gradient_contributions = self._compute_gradient_contributions(
                final_params, panel, reward_fn, operator
            )

        optimization_time = time.time() - start_time

        self._log(f"Optimization complete: feature_diff={feature_diff:.6f}, LL={ll:.2f}")

        return EstimationResult(
            parameters=final_params,
            log_likelihood=ll,
            value_function=solver_result.V,
            policy=solver_result.policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=result_opt.nit,
            num_function_evals=num_function_evals,
            num_inner_iterations=total_inner_iterations,
            message=result_opt.message,
            optimization_time=optimization_time,
            metadata={
                "optimizer": self._optimizer,
                "inner_tol": self._inner_tol,
                "outer_tol": self._outer_tol,
                "empirical_features": empirical_features.tolist(),
                "final_expected_features": final_expected.tolist(),
                "feature_difference": feature_diff,
            },
        )

    def _compute_gradient_contributions(
        self,
        params: jnp.ndarray,
        panel: Panel,
        reward_fn: LinearReward,
        operator: SoftBellmanOperator,
        eps: float = 1e-5,
    ) -> jnp.ndarray:
        """Compute per-observation gradient contributions.

        These are needed for robust and clustered standard errors.
        Uses numerical differentiation.

        Args:
            params: Current parameter values
            panel: Panel data
            reward_fn: Reward function
            operator: Bellman operator
            eps: Finite difference step size

        Returns:
            Gradient contributions, shape (num_observations, num_parameters)
        """
        n_obs = panel.num_observations
        n_params = len(params)

        gradients = np.zeros((n_obs, n_params))

        # Pre-compute log probabilities at current params
        reward_matrix = reward_fn.compute(params)
        solver_result = self._solve_inner(operator, reward_matrix)
        log_probs_base = operator.compute_log_choice_probabilities(
            reward_matrix, solver_result.V
        )

        # Compute gradient for each parameter
        for k in range(n_params):
            params_plus = np.asarray(params, dtype=np.float32)
            params_plus[k] += eps
            params_plus = jnp.array(params_plus)

            reward_plus = reward_fn.compute(params_plus)
            solver_plus = self._solve_inner(operator, reward_plus)
            log_probs_plus = operator.compute_log_choice_probabilities(
                reward_plus, solver_plus.V
            )

            params_minus = np.asarray(params, dtype=np.float32)
            params_minus[k] -= eps
            params_minus = jnp.array(params_minus)

            reward_minus = reward_fn.compute(params_minus)
            solver_minus = self._solve_inner(operator, reward_minus)
            log_probs_minus = operator.compute_log_choice_probabilities(
                reward_minus, solver_minus.V
            )

            # Compute gradients for all observations at once
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            gradients[:, k] = np.asarray(
                (log_probs_plus[all_states, all_actions] - log_probs_minus[all_states, all_actions]) / (2 * eps)
            )

        return jnp.array(gradients)

    def compute_feature_expectations(
        self,
        params: jnp.ndarray,
        reward_fn: LinearReward,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        panel: Panel | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute both empirical and expected feature expectations.

        Useful for diagnostics and checking feature matching quality.

        Args:
            params: Reward parameters
            reward_fn: LinearReward function
            problem: Problem specification
            transitions: Transition matrices
            panel: Panel data (for empirical features and initial distribution)

        Returns:
            Tuple of (empirical_features, expected_features)
        """
        operator = SoftBellmanOperator(problem, transitions)

        # Compute policy under given reward
        reward_matrix = reward_fn.compute(params)
        solver_result = self._solve_inner(operator, reward_matrix)

        # Compute expected features
        expected = self._compute_expected_features(
            solver_result.policy,
            transitions,
            reward_fn,
            problem,
            panel,
        )

        # Compute empirical features if panel provided
        if panel is not None:
            empirical = self._compute_empirical_features(panel, reward_fn)
        else:
            empirical = jnp.zeros_like(expected)

        return empirical, expected
