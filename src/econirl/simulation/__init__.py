"""Simulation and counterfactual analysis."""

from econirl.simulation.synthetic import simulate_panel
from econirl.simulation.counterfactual import (
    CounterfactualType,
    CounterfactualResult,
    counterfactual,
    counterfactual_policy,
    counterfactual_transitions,
    compute_stationary_distribution,
    compute_welfare_effect,
    discount_factor_change,
    elasticity_analysis,
    simulate_counterfactual,
    state_extrapolation,
    welfare_decomposition,
)

__all__ = [
    "CounterfactualType",
    "CounterfactualResult",
    "counterfactual",
    "counterfactual_policy",
    "counterfactual_transitions",
    "compute_stationary_distribution",
    "compute_welfare_effect",
    "discount_factor_change",
    "elasticity_analysis",
    "simulate_counterfactual",
    "simulate_panel",
    "state_extrapolation",
    "welfare_decomposition",
]
