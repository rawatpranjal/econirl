"""Smoke tests for f-IRL (state-marginal matching IRL)."""

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.f_irl import FIRLEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


@pytest.fixture
def small_firl_setup():
    """Small environment for quick f-IRL tests."""
    env = RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        seed=42,
    )
    panel = simulate_panel(env, n_individuals=50, n_periods=50, seed=123)
    utility = LinearUtility.from_environment(env)
    return env, panel, utility


def test_firl_init():
    """FIRLEstimator can be instantiated without error."""
    estimator = FIRLEstimator(f_divergence="kl", verbose=False)
    assert estimator is not None


def test_firl_estimate(small_firl_setup):
    """f-IRL runs on a small problem and returns a result with a policy."""
    env, panel, utility = small_firl_setup
    estimator = FIRLEstimator(
        f_divergence="kl",
        lr=0.1,
        max_iter=50,
        verbose=False,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    assert result.policy is not None
    assert result.policy.shape == (20, 2)


def test_firl_policy_valid(small_firl_setup):
    """f-IRL policy should be a valid probability distribution."""
    env, panel, utility = small_firl_setup
    estimator = FIRLEstimator(
        f_divergence="kl",
        lr=0.1,
        max_iter=50,
        verbose=False,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    policy = result.policy
    # Non-negative
    assert float(policy.min()) >= 0.0, "Policy has negative probabilities"
    # Rows sum to 1
    row_sums = policy.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-4)
