"""Behavioral Cloning — supervised learning baseline.

Predicts P(a|s) directly from demonstration frequencies. No RL, no DP,
no reward recovery. This is equivalent to the first stage of Hotz-Miller
CCP estimation, returned as the final policy without any subsequent
structural estimation step.

Every IRL benchmark includes behavioral cloning as a lower bound on
imitation performance: any method that cannot beat BC is not learning
from the MDP structure.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import BaseUtilityFunction


class BehavioralCloningEstimator(BaseEstimator):
    """Behavioral Cloning estimator.

    Counts state-action frequencies from demonstrations to produce
    an empirical policy P(a|s). Optional Laplace smoothing prevents
    zero probabilities for unvisited state-action pairs.

    This estimator does not recover structural parameters or reward
    functions — it only imitates observed behavior.

    Attributes:
        smoothing: Laplace smoothing constant added to all counts.

    Example:
        >>> estimator = BehavioralCloningEstimator(smoothing=1.0)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.policy)  # P(a|s) from demonstrations
    """

    def __init__(
        self,
        smoothing: float = 0.0,
        verbose: bool = False,
    ):
        """Initialize the Behavioral Cloning estimator.

        Args:
            smoothing: Laplace smoothing constant. 0.0 = no smoothing
                (MLE), 1.0 = add-one smoothing.
            verbose: Whether to print progress messages.
        """
        super().__init__(
            se_method="asymptotic",
            compute_hessian=False,
            verbose=verbose,
        )
        self._smoothing = smoothing

    @property
    def name(self) -> str:
        return "Behavioral Cloning"

    def _optimize(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Count state-action frequencies from demonstrations.

        Args:
            panel: Panel data with expert demonstrations.
            utility: Utility function (ignored — BC is model-free).
            problem: Problem specification.
            transitions: Transition matrices (ignored).
            initial_params: Ignored.

        Returns:
            EstimationResult with empirical policy.
        """
        start = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions

        # Count state-action pairs
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        counts = jnp.zeros((n_states, n_actions), dtype=jnp.float32)
        counts = counts.at[all_states, all_actions].add(1.0)

        # Laplace smoothing
        counts = counts + self._smoothing

        # Normalize to P(a|s)
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums = jnp.clip(row_sums, a_min=1e-10)
        policy = counts / row_sums

        # Log-likelihood of observed data under the empirical policy
        ll = float(jnp.log(policy[all_states, all_actions] + 1e-10).sum())

        elapsed = time.time() - start
        self._log(f"Computed empirical policy in {elapsed:.3f}s, LL={ll:.2f}")

        return EstimationResult(
            parameters=policy.flatten(),
            log_likelihood=ll,
            value_function=jnp.zeros(n_states),
            policy=policy,
            hessian=None,
            converged=True,
            num_iterations=1,
            message="Direct estimation (no iteration)",
            optimization_time=elapsed,
        )

    def estimate(
        self,
        panel: Panel,
        utility: BaseUtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate empirical policy from panel data.

        Overrides BaseEstimator.estimate() to skip standard error
        computation and identification diagnostics, which are not
        meaningful for frequency counting.

        Args:
            panel: Panel data with observed choices.
            utility: Utility function (ignored).
            problem: Problem specification.
            transitions: Transition matrices (ignored).
            initial_params: Ignored.

        Returns:
            EstimationSummary with empirical policy.
        """
        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        n_params = problem.num_states * (problem.num_actions - 1)
        n_obs = panel.num_observations

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=result.log_likelihood,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * result.log_likelihood + 2 * n_params,
            bic=-2 * result.log_likelihood
            + n_params * float(jnp.log(jnp.array(n_obs))),
            prediction_accuracy=self._compute_prediction_accuracy(
                panel, result.policy
            ),
        )

        # Parameter names: one per (state, action) cell
        param_names = [
            f"P(a={a}|s={s})"
            for s in range(problem.num_states)
            for a in range(problem.num_actions)
        ]

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=param_names,
            standard_errors=jnp.full_like(result.parameters, float("nan")),
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=result.log_likelihood,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=True,
            num_iterations=1,
            convergence_message="Direct estimation (no iteration)",
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=result.optimization_time,
            metadata={},
        )
