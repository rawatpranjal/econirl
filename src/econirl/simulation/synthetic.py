"""Synthetic data generation for Monte Carlo studies.

This module provides functions for simulating panel data from dynamic
discrete choice models with known parameters. This is essential for:
- Testing estimator performance (parameter recovery)
- Monte Carlo experiments (coverage of confidence intervals)
- Power analysis for hypothesis tests
- Understanding model behavior

The simulation follows the data generating process:
1. Draw initial state from distribution
2. At each period:
   a. Draw preference shocks ε ~ Type I Extreme Value
   b. Choose action maximizing U(s,a;θ) + ε(a)
   c. Transition to next state according to P(s'|s,a)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.base import DDCEnvironment


def simulate_panel(
    env: DDCEnvironment,
    n_individuals: int = 100,
    n_periods: int = 100,
    seed: int | None = None,
    use_optimal_policy: bool = True,
    policy: jnp.ndarray | None = None,
) -> Panel:
    """Simulate panel data from a DDC environment.

    Generates synthetic data by simulating the decision process for
    multiple individuals over multiple time periods.

    Args:
        env: DDCEnvironment with true parameters
        n_individuals: Number of individuals to simulate
        n_periods: Number of time periods per individual
        seed: Random seed for reproducibility
        use_optimal_policy: If True, compute optimal policy from true params
        policy: Pre-computed policy to use (overrides use_optimal_policy)

    Returns:
        Panel object with simulated trajectories

    Example:
        >>> env = RustBusEnvironment(operating_cost=0.001, replacement_cost=3.0)
        >>> panel = simulate_panel(env, n_individuals=500, n_periods=100, seed=42)
        >>> print(f"Generated {panel.num_observations} observations")
    """
    rng = np.random.default_rng(seed)

    # Get problem specification
    problem = env.problem_spec
    transitions = env.transition_matrices

    # Compute optimal policy if needed
    if policy is None and use_optimal_policy:
        policy = _compute_optimal_policy(env)
    elif policy is None:
        raise ValueError("Must provide policy or set use_optimal_policy=True")

    # Simulate trajectories
    trajectories = []

    for i in range(n_individuals):
        traj = _simulate_trajectory(
            env=env,
            policy=policy,
            n_periods=n_periods,
            individual_id=i,
            rng=rng,
        )
        trajectories.append(traj)

    return Panel(
        trajectories=trajectories,
        metadata={
            "n_individuals": n_individuals,
            "n_periods": n_periods,
            "seed": seed,
            "true_parameters": env.true_parameters,
        },
    )


def _compute_optimal_policy(env: DDCEnvironment) -> jnp.ndarray:
    """Compute the optimal choice probabilities for an environment.

    Args:
        env: Environment with true parameters

    Returns:
        Policy tensor of shape (num_states, num_actions)
    """
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = env.compute_utility_matrix()

    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, utility)

    return result.policy


def _simulate_trajectory(
    env: DDCEnvironment,
    policy: jnp.ndarray,
    n_periods: int,
    individual_id: int | str,
    rng: np.random.Generator,
) -> Trajectory:
    """Simulate a single individual's trajectory.

    Uses the Gumbel trick: sample action by adding Gumbel noise to
    Q-values and taking argmax, which is equivalent to sampling from
    the softmax policy.

    Args:
        env: Environment
        policy: Choice probabilities π(a|s)
        n_periods: Number of periods to simulate
        individual_id: Identifier for this individual
        rng: Random number generator

    Returns:
        Trajectory object
    """
    num_states = env.num_states
    num_actions = env.num_actions

    states = []
    actions = []
    next_states = []

    # Sample initial state
    state, _ = env.reset(seed=int(rng.integers(0, 2**31)))

    for t in range(n_periods):
        states.append(state)

        # Sample action from policy (normalize for float32 rounding)
        action_probs = np.asarray(policy[state], dtype=np.float64)
        action_probs = action_probs / action_probs.sum()
        action = rng.choice(num_actions, p=action_probs)
        actions.append(action)

        # Transition to next state
        next_state, _, _, _, _ = env.step(action)
        next_states.append(next_state)

        state = next_state

    return Trajectory(
        states=jnp.array(states, dtype=jnp.int32),
        actions=jnp.array(actions, dtype=jnp.int32),
        next_states=jnp.array(next_states, dtype=jnp.int32),
        individual_id=individual_id,
    )


def simulate_panel_from_policy(
    problem: DDCProblem,
    transitions: jnp.ndarray,
    policy: jnp.ndarray,
    initial_distribution: jnp.ndarray,
    n_individuals: int = 100,
    n_periods: int = 100,
    seed: int | None = None,
) -> Panel:
    """Simulate panel data from a given policy (without environment).

    This is useful when you have estimated a policy and want to
    simulate from it without reconstructing the environment.

    Args:
        problem: DDCProblem specification
        transitions: Transition matrices P(s'|s,a)
        policy: Choice probabilities π(a|s)
        initial_distribution: Initial state distribution
        n_individuals: Number of individuals
        n_periods: Number of periods
        seed: Random seed

    Returns:
        Simulated Panel
    """
    rng = np.random.default_rng(seed)

    num_states = problem.num_states
    num_actions = problem.num_actions

    trajectories = []

    for i in range(n_individuals):
        states = []
        actions = []
        next_states = []

        # Sample initial state
        state = rng.choice(num_states, p=initial_distribution)

        for t in range(n_periods):
            states.append(state)

            # Sample action (normalize for float32 rounding)
            action_probs = np.asarray(policy[state], dtype=np.float64)
            action_probs = action_probs / action_probs.sum()
            action = rng.choice(num_actions, p=action_probs)
            actions.append(action)

            # Sample next state (normalize for float32 rounding)
            trans_probs = np.asarray(transitions[action, state], dtype=np.float64)
            trans_probs = trans_probs / trans_probs.sum()
            next_state = rng.choice(num_states, p=trans_probs)
            next_states.append(next_state)

            state = next_state

        traj = Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=i,
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo simulation study.

    Attributes:
        true_parameters: True parameter values
        estimates: Array of estimates from each replication
        standard_errors: Array of SEs from each replication
        mean_estimate: Average estimate across replications
        std_estimate: Standard deviation of estimates
        bias: Mean estimate - true value
        rmse: Root mean squared error
        coverage_95: Fraction of 95% CIs containing true value
    """

    true_parameters: dict[str, float]
    parameter_names: list[str]
    estimates: np.ndarray  # (n_replications, n_params)
    standard_errors: np.ndarray  # (n_replications, n_params)
    mean_estimate: np.ndarray
    std_estimate: np.ndarray
    bias: np.ndarray
    rmse: np.ndarray
    coverage_95: np.ndarray


def run_monte_carlo(
    env: DDCEnvironment,
    estimator,
    utility,
    n_replications: int = 100,
    n_individuals: int = 100,
    n_periods: int = 100,
    seed: int | None = None,
    verbose: bool = True,
) -> MonteCarloResult:
    """Run a Monte Carlo simulation study.

    Repeatedly:
    1. Simulate data from true parameters
    2. Estimate parameters
    3. Collect estimates and standard errors

    Then compute bias, RMSE, and coverage.

    Args:
        env: Environment with true parameters
        estimator: Estimator to evaluate
        utility: Utility specification
        n_replications: Number of Monte Carlo replications
        n_individuals: Individuals per simulated panel
        n_periods: Periods per individual
        seed: Random seed
        verbose: Whether to print progress

    Returns:
        MonteCarloResult with simulation study results
    """
    from scipy import stats

    rng = np.random.default_rng(seed)

    true_params = env.true_parameters
    param_names = env.parameter_names
    n_params = len(param_names)

    estimates = np.zeros((n_replications, n_params))
    standard_errors = np.zeros((n_replications, n_params))

    problem = env.problem_spec
    transitions = env.transition_matrices

    for rep in range(n_replications):
        if verbose and (rep + 1) % 10 == 0:
            print(f"Replication {rep + 1}/{n_replications}")

        # Simulate data
        rep_seed = rng.integers(0, 2**31)
        panel = simulate_panel(
            env, n_individuals=n_individuals, n_periods=n_periods, seed=rep_seed
        )

        # Estimate
        result = estimator.estimate(panel, utility, problem, transitions)

        estimates[rep] = result.parameters
        standard_errors[rep] = result.standard_errors

    # Compute summary statistics
    true_vec = np.array([true_params[name] for name in param_names])

    mean_estimate = estimates.mean(axis=0)
    std_estimate = estimates.std(axis=0)
    bias = mean_estimate - true_vec
    rmse = np.sqrt(((estimates - true_vec) ** 2).mean(axis=0))

    # Coverage: fraction of CIs containing true value
    z = stats.norm.ppf(0.975)
    lower = estimates - z * standard_errors
    upper = estimates + z * standard_errors
    coverage_95 = ((lower <= true_vec) & (true_vec <= upper)).mean(axis=0)

    return MonteCarloResult(
        true_parameters=true_params,
        parameter_names=param_names,
        estimates=estimates,
        standard_errors=standard_errors,
        mean_estimate=mean_estimate,
        std_estimate=std_estimate,
        bias=bias,
        rmse=rmse,
        coverage_95=coverage_95,
    )
