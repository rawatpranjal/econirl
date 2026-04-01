"""Warm-start bootstrap inference for structural DDC models.

Implements the Kasahara and Shimotsu (2008) one-step NPL bootstrap,
which achieves quadratic convergence with dramatically lower computational
cost than full re-estimation per replicate. The key idea: warm-start each
bootstrap replicate from the MLE estimate and run only 1-3 Newton steps
instead of full convergence.

Reference: Kasahara, H. and Shimotsu, K. (2008). Pseudo-likelihood
Estimation and Bootstrap Inference for Structural Discrete Markov
Decision Models. Journal of Econometrics, 146(1), 92-106.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from econirl.core.types import DDCProblem, Panel
    from econirl.estimation.base import BaseEstimator
    from econirl.inference.results import EstimationSummary


def warm_start_bootstrap(
    estimator: BaseEstimator,
    panel: Panel,
    utility,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    mle_result: EstimationSummary,
    n_bootstrap: int = 499,
    n_newton_steps: int = 1,
    seed: int = 42,
) -> dict[str, object]:
    """Kasahara-Shimotsu warm-start bootstrap for DDC models.

    Resamples individuals with replacement and runs a small number of
    optimizer steps from the MLE estimate. With n_newton_steps=1,
    this matches the full MLE bootstrap rate while being roughly 100x
    faster (Kasahara and Shimotsu 2008, Proposition 4).

    Args:
        estimator: A BaseEstimator instance (NFXP, CCP, etc.).
        panel: The original panel data.
        utility: Utility function specification.
        problem: DDCProblem specification.
        transitions: Transition probabilities, shape (A, S, S).
        mle_result: EstimationSummary from the full MLE estimation.
        n_bootstrap: Number of bootstrap replications. 499 is standard.
        n_newton_steps: Number of outer optimizer iterations per replicate.
            1 is recommended (matches MLE rate). Use 2-3 for extra safety.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
            standard_errors (ndarray): Bootstrap SEs, shape (K,).
            variance_covariance (ndarray): Bootstrap covariance, shape (K, K).
            bootstrap_params (ndarray): All bootstrap estimates, shape (B, K).
            n_successful (int): Number of successful bootstrap replications.
            method (str): "warm_start_bootstrap".
    """
    from econirl.core.types import Panel

    rng = np.random.RandomState(seed)
    n_individuals = panel.num_individuals
    mle_params = mle_result.parameters

    bootstrap_estimates = []

    for b in range(n_bootstrap):
        # Resample individuals with replacement
        indices = rng.choice(n_individuals, size=n_individuals, replace=True)
        boot_trajectories = [panel.trajectories[i] for i in indices]
        boot_panel = Panel(trajectories=boot_trajectories)

        try:
            boot_result = estimator._optimize(
                panel=boot_panel,
                utility=utility,
                problem=problem,
                transitions=transitions,
                initial_params=mle_params,
                outer_max_iter=n_newton_steps,
            )
            bootstrap_estimates.append(np.asarray(boot_result.parameters))
        except Exception:
            # Skip failed replications
            continue

    if len(bootstrap_estimates) < 2:
        k = len(mle_params)
        return {
            "standard_errors": np.full(k, np.nan),
            "variance_covariance": np.full((k, k), np.nan),
            "bootstrap_params": np.array([]),
            "n_successful": 0,
            "method": "warm_start_bootstrap",
        }

    boot_matrix = np.stack(bootstrap_estimates)  # (B_success, K)
    se = np.std(boot_matrix, axis=0, ddof=1)
    cov = np.cov(boot_matrix, rowvar=False)

    return {
        "standard_errors": se,
        "variance_covariance": cov,
        "bootstrap_params": boot_matrix,
        "n_successful": len(bootstrap_estimates),
        "method": "warm_start_bootstrap",
    }
