"""Simulation and counterfactual analysis."""

from econirl.simulation.synthetic import simulate_panel
from econirl.simulation.counterfactual import counterfactual_policy, simulate_counterfactual

__all__ = ["simulate_panel", "counterfactual_policy", "simulate_counterfactual"]
