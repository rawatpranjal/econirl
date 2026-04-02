"""Counterfactual analysis for policy evaluation.

This module provides tools for analyzing how changes in the economic
environment (parameters, transitions, constraints) affect optimal
behavior and outcomes. Every counterfactual falls into one of four
types, following the taxonomy in Rawat (2026).

Type 1 (state extrapolation) shifts realized state values while
holding the MDP fixed. No Bellman equation is re-solved.

Type 2 (environment change) modifies the transition kernel or action
sets while holding the reward fixed. The Bellman equation must be
re-solved with the structural reward under new dynamics.

Type 3 (reward parameter change) modifies the reward function itself
through parameter perturbations, taxes, subsidies, or discount factor
shifts. This requires knowledge of the reward in levels.

Type 4 (welfare decomposition) decomposes a total welfare change into
the reward channel, the transition channel, and their interaction
using Shapley-value averaging over orderings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable

import numpy as np
import jax
import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.inference.results import EstimationSummary
from econirl.preferences.base import UtilityFunction


class CounterfactualType(IntEnum):
    """Taxonomy of counterfactual exercises.

    Each type requires progressively stronger identification of the
    reward function. Type 1 needs only the advantage function. Type 2
    needs the structural reward separated from continuation values.
    Type 3 needs the reward in levels. Type 4 decomposes welfare
    changes into interpretable channels.
    """

    STATE_EXTRAPOLATION = 1
    ENVIRONMENT_CHANGE = 2
    REWARD_CHANGE = 3
    WELFARE_DECOMPOSITION = 4


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
        counterfactual_type: Which of the four counterfactual types
        description: Description of the counterfactual
    """

    baseline_policy: jnp.ndarray
    counterfactual_policy: jnp.ndarray
    baseline_value: jnp.ndarray
    counterfactual_value: jnp.ndarray
    policy_change: jnp.ndarray
    value_change: jnp.ndarray
    welfare_change: float
    counterfactual_type: CounterfactualType = CounterfactualType.REWARD_CHANGE
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def state_extrapolation(
    result: EstimationSummary,
    state_mapping: dict[int, int] | jnp.ndarray,
    problem: DDCProblem,
    transitions: jnp.ndarray,
) -> CounterfactualResult:
    """Type 1 counterfactual: evaluate policy at shifted state values.

    The MDP structure (transitions and reward) is unchanged. Only the
    realized state values shift. The counterfactual evaluates the
    existing policy at new state indices without re-solving any
    Bellman equation.

    Every estimator that produces a policy can handle Type 1,
    including behavioral cloning and reduced-form Q-estimation.

    Args:
        result: Estimation result with baseline policy and value function
        state_mapping: Maps each state index to the index it should be
            evaluated as. A dict {50: 30} means state 50 behaves as
            state 30. An ndarray of shape (S,) maps every state.
        problem: Problem specification
        transitions: Transition matrices (unused, for API consistency)

    Returns:
        CounterfactualResult with Type 1 tag

    Example:
        >>> # What if all states shift down by 10 mileage bins?
        >>> mapping = {s: max(0, s - 10) for s in range(problem.num_states)}
        >>> cf = state_extrapolation(result, mapping, problem, transitions)
    """
    baseline_policy = result.policy
    baseline_value = result.value_function

    # Build full mapping array
    if isinstance(state_mapping, dict):
        mapping_arr = jnp.arange(problem.num_states)
        for src, dst in state_mapping.items():
            mapping_arr = mapping_arr.at[src].set(dst)
    else:
        mapping_arr = jnp.asarray(state_mapping, dtype=jnp.int32)

    # Evaluate policy and value at mapped states
    cf_policy = baseline_policy[mapping_arr]
    cf_value = baseline_value[mapping_arr]

    policy_change = cf_policy - baseline_policy
    value_change = cf_value - baseline_value
    welfare_change = float(value_change.mean())

    return CounterfactualResult(
        baseline_policy=baseline_policy,
        counterfactual_policy=cf_policy,
        baseline_value=baseline_value,
        counterfactual_value=cf_value,
        policy_change=policy_change,
        value_change=value_change,
        welfare_change=welfare_change,
        counterfactual_type=CounterfactualType.STATE_EXTRAPOLATION,
        description="Type 1: state-value extrapolation",
        metadata={"state_mapping": mapping_arr.tolist()},
    )


def counterfactual_policy(
    result: EstimationSummary,
    new_parameters: jnp.ndarray | dict[str, float],
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: jnp.ndarray,
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
        >>> new_params = result.parameters.copy()
        >>> new_params[1] *= 2  # replacement_cost
        >>> cf = counterfactual_policy(result, new_params, utility, problem, trans)
        >>> print(f"Welfare change: {cf.welfare_change:.2f}")
    """
    # Convert dict to tensor if needed
    if isinstance(new_parameters, dict):
        new_params = jnp.array(
            [new_parameters[name] for name in utility.parameter_names],
            dtype=jnp.float32,
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
    welfare_change = value_change.mean()

    return CounterfactualResult(
        baseline_policy=baseline_policy,
        counterfactual_policy=counterfactual_policy,
        baseline_value=baseline_value,
        counterfactual_value=counterfactual_value,
        policy_change=policy_change,
        value_change=value_change,
        welfare_change=welfare_change,
        counterfactual_type=CounterfactualType.REWARD_CHANGE,
        description="Type 3: reward parameter change",
        metadata={
            "baseline_parameters": result.parameters.tolist(),
            "counterfactual_parameters": new_params.tolist(),
        },
    )


def counterfactual_transitions(
    result: EstimationSummary,
    new_transitions: jnp.ndarray,
    utility: UtilityFunction,
    problem: DDCProblem,
    baseline_transitions: jnp.ndarray,
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
    welfare_change = value_change.mean()

    return CounterfactualResult(
        baseline_policy=baseline_result.policy,
        counterfactual_policy=cf_result.policy,
        baseline_value=baseline_result.V,
        counterfactual_value=cf_result.V,
        policy_change=policy_change,
        value_change=value_change,
        welfare_change=welfare_change,
        counterfactual_type=CounterfactualType.ENVIRONMENT_CHANGE,
        description="Type 2: environment change",
        metadata={},
    )


def simulate_counterfactual(
    result: EstimationSummary,
    counterfactual: CounterfactualResult,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    initial_distribution: jnp.ndarray | None = None,
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
        initial_distribution = jnp.ones(problem.num_states) / problem.num_states

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

    # Action frequencies (functional — no in-place mutation)
    baseline_action_freq = jnp.array([
        float((baseline_actions == a).astype(jnp.float32).mean())
        for a in range(problem.num_actions)
    ])
    cf_action_freq = jnp.array([
        float((cf_actions == a).astype(jnp.float32).mean())
        for a in range(problem.num_actions)
    ])

    # State frequencies
    baseline_state_freq = jnp.array([
        float((baseline_states == s).astype(jnp.float32).mean())
        for s in range(problem.num_states)
    ])
    cf_state_freq = jnp.array([
        float((cf_states == s).astype(jnp.float32).mean())
        for s in range(problem.num_states)
    ])

    return {
        "baseline_action_frequencies": baseline_action_freq,
        "counterfactual_action_frequencies": cf_action_freq,
        "action_frequency_change": cf_action_freq - baseline_action_freq,
        "baseline_state_frequencies": baseline_state_freq,
        "counterfactual_state_frequencies": cf_state_freq,
        "state_frequency_change": cf_state_freq - baseline_state_freq,
        "baseline_mean_state": baseline_states.astype(jnp.float32).mean(),
        "counterfactual_mean_state": cf_states.astype(jnp.float32).mean(),
        "n_individuals": n_individuals,
        "n_periods": n_periods,
    }


def compute_stationary_distribution(
    policy: jnp.ndarray,
    transitions: jnp.ndarray,
) -> jnp.ndarray:
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
    P_pi = jnp.einsum("sa,ast->st", policy, transitions)

    # Find stationary distribution as left eigenvector with eigenvalue 1
    # Solve μ P = μ, or equivalently (P' - I) μ = 0 with Σ μ = 1
    # Use power iteration for simplicity
    mu = jnp.ones(num_states) / num_states

    for _ in range(1000):
        mu_new = P_pi.T @ mu
        mu_new = mu_new / mu_new.sum()

        if jnp.abs(mu_new - mu).max() < 1e-10:
            break
        mu = mu_new

    return mu


def compute_welfare_effect(
    counterfactual: CounterfactualResult,
    transitions: jnp.ndarray,
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
        ev_baseline = (mu_baseline * counterfactual.baseline_value).sum()
        ev_cf = (mu_cf * counterfactual.counterfactual_value).sum()

        # Welfare change holding distribution fixed
        welfare_fixed_dist = (
            mu_baseline * counterfactual.value_change
        ).sum()

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
            "mean_value_change": counterfactual.value_change.mean(),
            "max_value_change": counterfactual.value_change.max(),
            "min_value_change": counterfactual.value_change.min(),
        }


def discount_factor_change(
    result: EstimationSummary,
    new_discount: float,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: jnp.ndarray,
) -> CounterfactualResult:
    """Type 3 counterfactual: change the discount factor.

    Re-solves the Bellman equation with the same reward function but
    a different discount factor beta. A lower beta makes the agent
    more myopic and less sensitive to future consequences.

    Args:
        result: Estimation result with baseline policy
        new_discount: New discount factor beta-tilde
        utility: Utility function specification
        problem: Problem specification (contains original beta)
        transitions: Transition matrices

    Returns:
        CounterfactualResult with Type 3 tag
    """
    # Baseline
    baseline_policy = result.policy
    baseline_value = result.value_function

    # Counterfactual: new DDCProblem with different discount factor
    cf_problem = DDCProblem(
        num_states=problem.num_states,
        num_actions=problem.num_actions,
        discount_factor=new_discount,
        scale_parameter=problem.scale_parameter,
    )
    cf_operator = SoftBellmanOperator(cf_problem, transitions)
    reward = utility.compute(result.parameters)
    cf_result = value_iteration(cf_operator, reward)

    policy_change = cf_result.policy - baseline_policy
    value_change = cf_result.V - baseline_value
    welfare_change = float(value_change.mean())

    return CounterfactualResult(
        baseline_policy=baseline_policy,
        counterfactual_policy=cf_result.policy,
        baseline_value=baseline_value,
        counterfactual_value=cf_result.V,
        policy_change=policy_change,
        value_change=value_change,
        welfare_change=welfare_change,
        counterfactual_type=CounterfactualType.REWARD_CHANGE,
        description="Type 3: discount factor change",
        metadata={
            "baseline_discount": problem.discount_factor,
            "counterfactual_discount": new_discount,
        },
    )


def welfare_decomposition(
    result: EstimationSummary,
    utility: UtilityFunction,
    problem: DDCProblem,
    baseline_transitions: jnp.ndarray,
    new_parameters: jnp.ndarray | None = None,
    new_transitions: jnp.ndarray | None = None,
) -> dict[str, float]:
    """Type 4 counterfactual: decompose welfare change into channels.

    When a counterfactual involves both a reward change and a
    transition change, the total welfare effect can be decomposed
    into the reward channel (direct effect of preference change),
    the transition channel (indirect effect of environment change),
    and their interaction. The decomposition uses Shapley-value
    averaging over the two orderings, which requires four Bellman
    solves.

    At least one of new_parameters or new_transitions must be provided.

    Args:
        result: Estimation result
        utility: Utility function specification
        problem: Problem specification
        baseline_transitions: Original transition matrices
        new_parameters: Counterfactual parameter values (or None)
        new_transitions: Counterfactual transition matrices (or None)

    Returns:
        Dictionary with total_welfare_change, reward_channel,
        transition_channel, and interaction_effect. The three
        components sum to the total.
    """
    if new_parameters is None and new_transitions is None:
        raise ValueError(
            "At least one of new_parameters or new_transitions must be provided"
        )

    old_reward = utility.compute(result.parameters)
    new_reward = (
        utility.compute(new_parameters) if new_parameters is not None else old_reward
    )
    cf_transitions = (
        new_transitions if new_transitions is not None else baseline_transitions
    )

    operator_old_p = SoftBellmanOperator(problem, baseline_transitions)
    operator_new_p = SoftBellmanOperator(problem, cf_transitions)

    # Four corners: (old_r, old_P), (new_r, old_P), (old_r, new_P), (new_r, new_P)
    res_oo = value_iteration(operator_old_p, old_reward)
    res_ro = value_iteration(operator_old_p, new_reward)
    res_ot = value_iteration(operator_new_p, old_reward)
    res_rt = value_iteration(operator_new_p, new_reward)

    # Welfare = stationary-distribution-weighted expected value
    def _welfare(vi_result, transitions_used):
        mu = compute_stationary_distribution(vi_result.policy, transitions_used)
        return float((mu * vi_result.V).sum())

    w_oo = _welfare(res_oo, baseline_transitions)
    w_ro = _welfare(res_ro, baseline_transitions)
    w_ot = _welfare(res_ot, cf_transitions)
    w_rt = _welfare(res_rt, cf_transitions)

    total = w_rt - w_oo

    # Shapley values: average marginal contributions over both orderings
    # Ordering 1: reward first, then transition
    reward_first = w_ro - w_oo
    transition_second = w_rt - w_ro
    # Ordering 2: transition first, then reward
    transition_first = w_ot - w_oo
    reward_second = w_rt - w_ot

    reward_channel = (reward_first + reward_second) / 2
    transition_channel = (transition_first + transition_second) / 2
    interaction = total - reward_channel - transition_channel

    return {
        "total_welfare_change": total,
        "reward_channel": reward_channel,
        "transition_channel": transition_channel,
        "interaction_effect": interaction,
        "welfare_baseline": w_oo,
        "welfare_counterfactual": w_rt,
    }


def counterfactual(
    result: EstimationSummary,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: jnp.ndarray,
    new_parameters: jnp.ndarray | dict[str, float] | None = None,
    new_transitions: jnp.ndarray | None = None,
    state_mapping: dict[int, int] | jnp.ndarray | None = None,
    new_discount: float | None = None,
) -> CounterfactualResult:
    """Unified counterfactual dispatcher.

    Automatically selects the counterfactual type based on which
    arguments are provided. Provide exactly one type of change, or
    combine new_parameters with new_transitions for a joint
    Type 2 and Type 3 counterfactual.

    Args:
        result: Estimation result
        utility: Utility function specification
        problem: Problem specification
        transitions: Baseline transition matrices
        new_parameters: Changed parameters (Type 3)
        new_transitions: Changed transitions (Type 2)
        state_mapping: State index remapping (Type 1)
        new_discount: Changed discount factor (Type 3)

    Returns:
        CounterfactualResult with the appropriate type tag

    Raises:
        ValueError: If argument combination is invalid
    """
    has_mapping = state_mapping is not None
    has_params = new_parameters is not None
    has_transitions = new_transitions is not None
    has_discount = new_discount is not None

    # Type 1: state mapping only
    if has_mapping:
        if has_params or has_transitions or has_discount:
            raise ValueError(
                "state_mapping (Type 1) cannot be combined with other changes"
            )
        return state_extrapolation(result, state_mapping, problem, transitions)

    # Type 3: discount factor change
    if has_discount:
        if has_params or has_transitions:
            raise ValueError(
                "new_discount cannot be combined with new_parameters or "
                "new_transitions in the unified dispatcher. Use the "
                "individual functions directly for combined changes."
            )
        return discount_factor_change(result, new_discount, utility, problem, transitions)

    # Type 2: transition change only
    if has_transitions and not has_params:
        return counterfactual_transitions(
            result, new_transitions, utility, problem, transitions
        )

    # Type 3: parameter change only
    if has_params and not has_transitions:
        return counterfactual_policy(
            result, new_parameters, utility, problem, transitions
        )

    # Combined Type 2+3: both parameter and transition change
    if has_params and has_transitions:
        if isinstance(new_parameters, dict):
            new_params = jnp.array(
                [new_parameters[name] for name in utility.parameter_names],
                dtype=jnp.float32,
            )
        else:
            new_params = new_parameters

        baseline_operator = SoftBellmanOperator(problem, transitions)
        baseline_utility = utility.compute(result.parameters)
        baseline_result = value_iteration(baseline_operator, baseline_utility)

        cf_operator = SoftBellmanOperator(problem, new_transitions)
        cf_utility = utility.compute(new_params)
        cf_result = value_iteration(cf_operator, cf_utility)

        policy_change = cf_result.policy - baseline_result.policy
        value_change = cf_result.V - baseline_result.V
        welfare_change = float(value_change.mean())

        return CounterfactualResult(
            baseline_policy=baseline_result.policy,
            counterfactual_policy=cf_result.policy,
            baseline_value=baseline_result.V,
            counterfactual_value=cf_result.V,
            policy_change=policy_change,
            value_change=value_change,
            welfare_change=welfare_change,
            counterfactual_type=CounterfactualType.ENVIRONMENT_CHANGE,
            description="Type 2+3: joint parameter and environment change",
            metadata={
                "baseline_parameters": result.parameters.tolist(),
                "counterfactual_parameters": new_params.tolist(),
            },
        )

    raise ValueError(
        "No counterfactual change specified. Provide at least one of: "
        "state_mapping, new_parameters, new_transitions, or new_discount."
    )


def elasticity_analysis(
    result: EstimationSummary,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: jnp.ndarray,
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
    baseline_value = float(result.parameters[param_idx])

    results = {
        "parameter": parameter_name,
        "baseline_value": baseline_value,
        "pct_changes": pct_changes,
        "policy_changes": [],
        "welfare_changes": [],
    }

    for pct in pct_changes:
        new_params = result.parameters.at[param_idx].set(baseline_value * (1 + pct))

        cf = counterfactual_policy(
            result=result,
            new_parameters=new_params,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )

        # Average absolute policy change
        avg_policy_change = float(jnp.abs(cf.policy_change).mean())
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
