"""Smoke tests for Behavioral Cloning (frequency-based imitation baseline)."""
import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


@pytest.fixture
def small_bc_setup():
    """Small environment for quick BC tests."""
    env = RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        seed=42,
    )
    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=123)
    utility = LinearUtility.from_environment(env)
    return env, panel, utility


def test_bc_init():
    """BehavioralCloningEstimator can be instantiated without error."""
    estimator = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    assert estimator is not None


def test_bc_estimate_runs(small_bc_setup):
    """BC runs on a small problem and returns a valid policy."""
    env, panel, utility = small_bc_setup
    estimator = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    assert result.policy is not None
    assert result.policy.shape == (20, 2)


def test_bc_policy_valid(small_bc_setup):
    """BC policy should be a valid probability distribution."""
    env, panel, utility = small_bc_setup
    estimator = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    policy = result.policy
    assert float(policy.min()) >= 0.0, "Policy has negative probabilities"
    row_sums = policy.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)


def test_bc_matches_frequencies(small_bc_setup):
    """BC policy should approximate empirical action frequencies."""
    env, panel, utility = small_bc_setup
    n_states = env.problem_spec.num_states
    n_actions = env.problem_spec.num_actions

    # Compute empirical frequencies from panel
    counts = np.zeros((n_states, n_actions), dtype=np.float64)
    for traj in panel.trajectories:
        for s, a in zip(traj.states, traj.actions):
            counts[int(s), int(a)] += 1

    # Normalize to probabilities (with add-1 smoothing to match estimator)
    counts_smoothed = counts + 1.0
    empirical_policy = counts_smoothed / counts_smoothed.sum(axis=1, keepdims=True)

    # Estimate with BC (same smoothing=1.0)
    estimator = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    # BC should match empirical frequencies closely
    visited_states = counts.sum(axis=1) > 0
    np.testing.assert_allclose(
        result.policy[visited_states],
        empirical_policy[visited_states],
        atol=0.05,
    )
