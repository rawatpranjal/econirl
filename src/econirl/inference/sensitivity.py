"""Discount factor sensitivity analysis for structural estimation.

The discount factor beta cannot be separately identified from flow
utilities without exclusion restrictions (Magnac and Thesmar, 2002).
This module provides a sensitivity table that re-estimates the model
at a grid of beta values, revealing how reward parameters, log-likelihood,
and predicted policies change with the assumed discount factor.
"""

from __future__ import annotations

from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from econirl.core.types import DDCProblem, Panel
    from econirl.estimation.base import BaseEstimator


def discount_factor_sensitivity(
    estimator: BaseEstimator,
    panel: Panel,
    utility,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    beta_grid: list[float] | None = None,
) -> dict[str, list]:
    """Re-estimate the model at a grid of discount factors.

    For each beta in the grid, creates a new DDCProblem with that
    discount factor and calls the estimator. Results are collected
    into a table showing how parameters shift with beta.

    Large sensitivity to the discount factor signals identification
    concerns (Magnac and Thesmar, 2002; Abbring and Daljord, 2020).

    Args:
        estimator: A fitted BaseEstimator instance (NFXP, CCP, etc.).
        panel: Panel data used for estimation.
        utility: Utility function specification.
        problem: DDCProblem with the baseline discount factor.
        transitions: Transition probabilities, shape (A, S, S).
        beta_grid: List of discount factors to try.
            Defaults to [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999].

    Returns:
        Dict with keys:
            beta_values (list[float]): The discount factors tested.
            params (list[dict]): Parameter estimates at each beta.
            log_likelihoods (list[float]): Log-likelihood at each beta.
            standard_errors (list[dict]): SEs at each beta.
            converged (list[bool]): Whether optimization converged.
    """
    if beta_grid is None:
        beta_grid = [0.90, 0.95, 0.99, 0.995, 0.999, 0.9999]

    results = {
        "beta_values": [],
        "params": [],
        "log_likelihoods": [],
        "standard_errors": [],
        "converged": [],
    }

    prev_params = None

    for beta in beta_grid:
        # Create problem with modified discount factor
        modified_problem = replace(problem, discount_factor=beta)

        try:
            summary = estimator.estimate(
                panel=panel,
                utility=utility,
                problem=modified_problem,
                transitions=transitions,
                initial_params=prev_params,
            )

            param_dict = {
                name: float(val)
                for name, val in zip(summary.parameter_names, summary.parameters)
            }
            se_dict = {
                name: float(val)
                for name, val in zip(summary.parameter_names, summary.standard_errors)
            }

            results["beta_values"].append(beta)
            results["params"].append(param_dict)
            results["log_likelihoods"].append(
                float(summary.log_likelihood) if summary.log_likelihood is not None else None
            )
            results["standard_errors"].append(se_dict)
            results["converged"].append(summary.converged)

            prev_params = summary.parameters

        except Exception as e:
            results["beta_values"].append(beta)
            results["params"].append({"error": str(e)})
            results["log_likelihoods"].append(None)
            results["standard_errors"].append({})
            results["converged"].append(False)

    return results
