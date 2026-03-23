"""Counterfactual analysis for policy evaluation.

This module provides tools for analyzing how changes in the economic
environment (parameters, transitions, constraints) affect optimal
behavior and outcomes.

Counterfactual analysis is a key application of structural estimation:
once we have estimated preferences, we can predict behavior under
scenarios not observed in the data.

Common counterfactual exercises:
- Policy changes: What if costs/benefits change?
- Transition changes: What if state dynamics change?
- Constraint changes: What if some actions become unavailable?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.inference.results import EstimationSummary
from econirl.preferences.base import UtilityFunction


@dataclass
class CounterfactualResult:
    """Results from counterfactual analysis.

    Attributes:
        baseline_policy: Original choice probabilities
        counterfactual_policy: New choice probabilities
        baseline_value: Original value function
        counterfactual_value: New value function
        policy_change: Change in choice probabilities
        value_change: Change in value function
        welfare_change: Average change in expected utility
        description: Description of the counterfactual
    """

    baseline_policy: torch.Tensor
    counterfactual_policy: torch.Tensor
    baseline_value: torch.Tensor
    counterfactual_value: torch.Tensor
    policy_change: torch.Tensor
    value_change: torch.Tensor
    welfare_change: float
    description: str
    metadata: dict[str, Any]


def counterfactual_policy(
    result: EstimationSummary,
    new_parameters: torch.Tensor | dict[str, float],
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
) -> CounterfactualResult:
    """Compute optimal policy under different parameters.

    Given estimated preferences, compute how behavior would change
    if utility parameters were different.

    Example: What if replacement cost increases by 50%?

    Args:
        result: Estimation result with baseline policy
        new_parameters: New parameter values (tensor or dict)
        utility: Utility function specification
        problem: Problem specification
        transitions: Transition matrices

    Returns:
        CounterfactualResult comparing baseline and counterfactual

    Example:
        >>> # What if replacement cost doubles?
        >>> new_params = result.parameters.clone()
        >>> new_params[1] *= 2  # replacement_cost
        >>> cf = counterfactual_policy(result, new_params, utility, problem, trans)
        >>> print(f"Welfare change: {cf.welfare_change:.2f}")
    """
    # Convert dict to tensor if needed
    if isinstance(new_parameters, dict):
        new_params = torch.tensor(
            [new_parameters[name] for name in utility.parameter_names],
            dtype=torch.float32,
        )
    else:
        new_params = new_parameters

    # Baseline (from estimation)
    baseline_policy = result.policy
    baseline_value = result.value_function

    # Counterfactual
    operator = SoftBellmanOperator(problem, transitions)
    new_utility = utility.compute(new_params)
    cf_result = value_iteration(operator, new_utility)

    counterfactual_policy = cf_result.policy
    counterfactual_value = cf_result.V

    # Compute changes
    policy_change = counterfactual_policy - baseline_policy
    value_change = counterfactual_value - baseline_value

    # Welfare change: average change in value across states
    # (could weight by stationary distribution)
    welfare_change = value_change.mean().item()

    return CounterfactualResult(
        baseline_policy=baseline_policy,
        counterfactual_policy=counterfactual_policy,
        baseline_value=baseline_value,
        counterfactual_value=counterfactual_value,
        policy_change=policy_change,
        value_change=value_change,
        welfare_change=welfare_change,
        description="Parameter change counterfactual",
        metadata={
            "baseline_parameters": result.parameters.tolist(),
            "counterfactual_parameters": new_params.tolist(),
        },
    )


def counterfactual_transitions(
    result: EstimationSummary,
    new_transitions: torch.Tensor,
    utility: UtilityFunction,
    problem: DDCProblem,
    baseline_transitions: torch.Tensor,
) -> CounterfactualResult:
    """Compute optimal policy under different transition dynamics.

    Analyze how behavior changes if the state transition probabilities
    change (e.g., different depreciation rates, different job arrival rates).

    Args:
        result: Estimation result
        new_transitions: New transition matrices
        utility: Utility specification
        problem: Problem specification
        baseline_transitions: Original transition matrices

    Returns:
        CounterfactualResult
    """
    # Baseline
    baseline_operator = SoftBellmanOperator(problem, baseline_transitions)
    baseline_utility = utility.compute(result.parameters)
    baseline_result = value_iteration(baseline_operator, baseline_utility)

    # Counterfactual
    cf_operator = SoftBellmanOperator(problem, new_transitions)
    cf_result = value_iteration(cf_operator, baseline_utility)

    policy_change = cf_result.policy - baseline_result.policy
    value_change = cf_result.V - baseline_result.V
    welfare_change = value_change.mean().item()

    return CounterfactualResult(
        baseline_policy=baseline_result.policy,
        counterfactual_policy=cf_result.policy,
        baseline_value=baseline_result.V,
        counterfactual_value=cf_result.V,
        policy_change=policy_change,
        value_change=value_change,
        welfare_change=welfare_change,
        description="Transition dynamics counterfactual",
        metadata={},
    )


def simulate_counterfactual(
    result: EstimationSummary,
    counterfactual: CounterfactualResult,
    problem: DDCProblem,
    transitions: torch.Tensor,
    initial_distribution: torch.Tensor | None = None,
    n_individuals: int = 1000,
    n_periods: int = 100,
    seed: int | None = None,
) -> dict[str, Any]:
    """Simulate outcomes under baseline and counterfactual policies.

    Generates synthetic data under both policies to compare
    aggregate outcomes (choice frequencies, state distributions, etc.).

    Args:
        result: Estimation result
        counterfactual: Counterfactual result
        problem: Problem specification
        transitions: Transition matrices (use counterfactual if changed)
        initial_distribution: Initial state distribution
        n_individuals: Number of individuals to simulate
        n_periods: Periods per individual
        seed: Random seed

    Returns:
        Dictionary with simulation comparison statistics
    """
    from econirl.simulation.synthetic import simulate_panel_from_policy

    if initial_distribution is None:
        # Uniform initial distribution
        initial_distribution = torch.ones(problem.num_states) / problem.num_states

    # Simulate under baseline
    baseline_panel = simulate_panel_from_policy(
        problem=problem,
        transitions=transitions,
        policy=counterfactual.baseline_policy,
        initial_distribution=initial_distribution,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )

    # Simulate under counterfactual
    cf_seed = seed + 1 if seed is not None else None
    cf_panel = simulate_panel_from_policy(
        problem=problem,
        transitions=transitions,
        policy=counterfactual.counterfactual_policy,
        initial_distribution=initial_distribution,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=cf_seed,
    )

    # Compute comparison statistics
    baseline_actions = baseline_panel.get_all_actions()
    cf_actions = cf_panel.get_all_actions()

    baseline_states = baseline_panel.get_all_states()
    cf_states = cf_panel.get_all_states()

    # Action frequencies
    baseline_action_freq = torch.zeros(problem.num_actions)
    cf_action_freq = torch.zeros(problem.num_actions)

    for a in range(problem.num_actions):
        baseline_action_freq[a] = (baseline_actions == a).float().mean()
        cf_action_freq[a] = (cf_actions == a).float().mean()

    # State frequencies
    baseline_state_freq = torch.zeros(problem.num_states)
    cf_state_freq = torch.zeros(problem.num_states)

    for s in range(problem.num_states):
        baseline_state_freq[s] = (baseline_states == s).float().mean()
        cf_state_freq[s] = (cf_states == s).float().mean()

    return {
        "baseline_action_frequencies": baseline_action_freq,
        "counterfactual_action_frequencies": cf_action_freq,
        "action_frequency_change": cf_action_freq - baseline_action_freq,
        "baseline_state_frequencies": baseline_state_freq,
        "counterfactual_state_frequencies": cf_state_freq,
        "state_frequency_change": cf_state_freq - baseline_state_freq,
        "baseline_mean_state": baseline_states.float().mean().item(),
        "counterfactual_mean_state": cf_states.float().mean().item(),
        "n_individuals": n_individuals,
        "n_periods": n_periods,
    }


def compute_stationary_distribution(
    policy: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """Compute the stationary state distribution under a policy.

    The stationary distribution μ satisfies:
        μ(s') = Σ_s μ(s) Σ_a π(a|s) P(s'|s,a)

    Args:
        policy: Choice probabilities π(a|s), shape (num_states, num_actions)
        transitions: Transition matrices P(s'|s,a), shape (num_actions, num_states, num_states)

    Returns:
        Stationary distribution, shape (num_states,)
    """
    num_states = policy.shape[0]

    # Policy-weighted transition matrix P^π(s,s') = Σ_a π(a|s) P(s'|s,a)
    # transitions shape: (num_actions, num_states, num_states) = [a, from_s, to_s]
    # policy shape: (num_states, num_actions) = [from_s, a]
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    # Find stationary distribution as left eigenvector with eigenvalue 1
    # Solve μ P = μ, or equivalently (P' - I) μ = 0 with Σ μ = 1
    # Use power iteration for simplicity
    mu = torch.ones(num_states) / num_states

    for _ in range(1000):
        mu_new = P_pi.T @ mu
        mu_new = mu_new / mu_new.sum()

        if torch.abs(mu_new - mu).max() < 1e-10:
            break
        mu = mu_new

    return mu


def compute_welfare_effect(
    counterfactual: CounterfactualResult,
    transitions: torch.Tensor,
    use_stationary: bool = True,
) -> dict[str, float]:
    """Compute welfare effects of a counterfactual.

    Computes the change in expected discounted utility, potentially
    weighted by the stationary distribution.

    Args:
        counterfactual: Counterfactual result
        transitions: Transition matrices
        use_stationary: Whether to weight by stationary distribution

    Returns:
        Dictionary with welfare measures
    """
    if use_stationary:
        # Weight by stationary distribution under baseline policy
        mu_baseline = compute_stationary_distribution(
            counterfactual.baseline_policy, transitions
        )
        mu_cf = compute_stationary_distribution(
            counterfactual.counterfactual_policy, transitions
        )

        # Expected value under each distribution
        ev_baseline = (mu_baseline * counterfactual.baseline_value).sum().item()
        ev_cf = (mu_cf * counterfactual.counterfactual_value).sum().item()

        # Welfare change holding distribution fixed
        welfare_fixed_dist = (
            mu_baseline * counterfactual.value_change
        ).sum().item()

        return {
            "baseline_expected_value": ev_baseline,
            "counterfactual_expected_value": ev_cf,
            "total_welfare_change": ev_cf - ev_baseline,
            "welfare_change_fixed_distribution": welfare_fixed_dist,
            "distribution_effect": (ev_cf - ev_baseline) - welfare_fixed_dist,
        }
    else:
        # Simple average across states
        return {
            "mean_value_change": counterfactual.value_change.mean().item(),
            "max_value_change": counterfactual.value_change.max().item(),
            "min_value_change": counterfactual.value_change.min().item(),
        }


def elasticity_analysis(
    result: EstimationSummary,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
    parameter_name: str,
    pct_changes: list[float] = [-0.1, -0.05, 0.05, 0.1],
) -> dict[str, Any]:
    """Analyze sensitivity of policy to parameter changes.

    Computes how choice probabilities change as a parameter
    varies by different percentages.

    Args:
        result: Estimation result
        utility: Utility specification
        problem: Problem specification
        transitions: Transition matrices
        parameter_name: Name of parameter to vary
        pct_changes: List of percentage changes to analyze

    Returns:
        Dictionary with elasticity analysis results
    """
    param_idx = utility.parameter_names.index(parameter_name)
    baseline_value = result.parameters[param_idx].item()

    results = {
        "parameter": parameter_name,
        "baseline_value": baseline_value,
        "pct_changes": pct_changes,
        "policy_changes": [],
        "welfare_changes": [],
    }

    for pct in pct_changes:
        new_params = result.parameters.clone()
        new_params[param_idx] = baseline_value * (1 + pct)

        cf = counterfactual_policy(
            result=result,
            new_parameters=new_params,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )

        # Average absolute policy change
        avg_policy_change = cf.policy_change.abs().mean().item()
        results["policy_changes"].append(avg_policy_change)
        results["welfare_changes"].append(cf.welfare_change)

    # Compute approximate elasticities
    if len(pct_changes) >= 2:
        # Use central difference if we have symmetric changes
        policy_elasticity = np.gradient(
            results["policy_changes"], pct_changes
        ).mean()
        welfare_elasticity = np.gradient(
            results["welfare_changes"], pct_changes
        ).mean()

        results["policy_elasticity"] = policy_elasticity
        results["welfare_elasticity"] = welfare_elasticity

    return results
