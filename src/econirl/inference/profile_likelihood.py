"""Profile likelihood computation for structural DDC models.

The profile likelihood fixes one parameter at a grid of values and
re-optimizes the remaining parameters at each point. The resulting
curve reveals identification strength: a sharply peaked profile means
the parameter is well-identified, while a flat profile signals weak
or non-identification.

Profile-based confidence intervals invert the chi-squared criterion
LL(theta_k) > LL_max - chi2(1, alpha)/2, which is more reliable than
Wald intervals when the likelihood surface is non-quadratic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from econirl.core.types import DDCProblem, Panel
    from econirl.estimation.base import BaseEstimator
    from econirl.inference.results import EstimationSummary


def profile_likelihood(
    estimator: BaseEstimator,
    panel: Panel,
    utility,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    mle_result: EstimationSummary,
    param_index: int,
    grid: np.ndarray | None = None,
    n_points: int = 20,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Compute the profile likelihood for a single parameter.

    Fixes parameter param_index at each value in the grid and
    re-optimizes the remaining K-1 parameters. Uses warm-starting
    from the MLE estimate for speed.

    Args:
        estimator: A BaseEstimator instance.
        panel: Panel data for estimation.
        utility: Utility function specification.
        problem: DDCProblem specification.
        transitions: Transition probabilities.
        mle_result: EstimationSummary from full MLE.
        param_index: Index of the parameter to profile (0-based).
        grid: Explicit grid of values. If None, uses MLE +/- 3 SE
            with n_points equally spaced values.
        n_points: Number of grid points if grid is not provided.
        alpha: Significance level for profile CI (default 0.05).

    Returns:
        Dict with keys:
            grid_values (ndarray): Parameter values tested.
            profile_ll (ndarray): Profile log-likelihood at each value.
            mle_value (float): MLE estimate of the profiled parameter.
            mle_ll (float): Log-likelihood at the MLE.
            ci_lower (float): Lower bound of profile CI.
            ci_upper (float): Upper bound of profile CI.
            param_name (str): Name of the profiled parameter.
    """
    mle_params = np.asarray(mle_result.parameters)
    mle_se = np.asarray(mle_result.standard_errors)
    mle_value = float(mle_params[param_index])
    mle_ll = float(mle_result.log_likelihood)
    param_name = mle_result.parameter_names[param_index]

    if grid is None:
        se_k = float(mle_se[param_index])
        if se_k <= 0 or np.isnan(se_k):
            se_k = abs(mle_value) * 0.1 if mle_value != 0 else 0.1
        grid = np.linspace(mle_value - 3 * se_k, mle_value + 3 * se_k, n_points)

    profile_ll = np.full(len(grid), np.nan)
    wrapped_utility = _FixedParamUtility(utility, param_index, 0.0)

    for i, fixed_value in enumerate(grid):
        wrapped_utility.fixed_value = fixed_value

        # Initial params for reduced problem: remove the fixed parameter
        init_reduced = np.delete(mle_params, param_index)

        try:
            result = estimator._optimize(
                panel=panel,
                utility=wrapped_utility,
                problem=problem,
                transitions=transitions,
                initial_params=jnp.array(init_reduced),
            )
            profile_ll[i] = float(result.log_likelihood)
        except Exception:
            continue

    # Profile CI: {theta_k : LL(theta_k) > LL_max - chi2(1,alpha)/2}
    cutoff = mle_ll - stats.chi2.ppf(1 - alpha, 1) / 2.0
    above_cutoff = grid[profile_ll >= cutoff] if np.any(~np.isnan(profile_ll)) else grid[[]]

    ci_lower = float(above_cutoff[0]) if len(above_cutoff) > 0 else float("nan")
    ci_upper = float(above_cutoff[-1]) if len(above_cutoff) > 0 else float("nan")

    return {
        "grid_values": grid,
        "profile_ll": profile_ll,
        "mle_value": mle_value,
        "mle_ll": mle_ll,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "param_name": param_name,
    }


class _FixedParamUtility:
    """Wraps a utility function to fix one parameter at a given value.

    When compute(reduced_params) is called with K-1 parameters, this
    wrapper splices the fixed value back into position param_index
    and delegates to the original utility's compute(full_params).
    """

    def __init__(self, original_utility, param_index: int, fixed_value: float):
        self._original = original_utility
        self._param_index = param_index
        self.fixed_value = fixed_value

    def _expand_params(self, reduced_params: jnp.ndarray) -> jnp.ndarray:
        """Insert the fixed value at param_index."""
        return jnp.insert(reduced_params, self._param_index, self.fixed_value)

    def compute(self, parameters: jnp.ndarray) -> jnp.ndarray:
        return self._original.compute(self._expand_params(parameters))

    def compute_gradient(self, parameters: jnp.ndarray) -> jnp.ndarray:
        full_grad = self._original.compute_gradient(self._expand_params(parameters))
        # Remove the column for the fixed parameter
        return jnp.delete(full_grad, self._param_index, axis=-1)

    def get_initial_parameters(self) -> jnp.ndarray:
        full_init = self._original.get_initial_parameters()
        return jnp.delete(full_init, self._param_index)

    @property
    def num_parameters(self) -> int:
        return self._original.num_parameters - 1

    @property
    def num_states(self) -> int:
        return self._original.num_states

    @property
    def num_actions(self) -> int:
        return self._original.num_actions

    @property
    def feature_matrix(self):
        fm = self._original.feature_matrix
        if fm is not None:
            return np.delete(np.asarray(fm), self._param_index, axis=-1)
        return None

    @property
    def parameter_names(self) -> list[str]:
        names = list(self._original.parameter_names)
        names.pop(self._param_index)
        return names
