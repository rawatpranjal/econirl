"""Unit tests for the shape-shifting synthetic DGP.

These tests verify ground-truth consistency, seed reproducibility,
and the diagnostics the JSS deep-run Tier 4 cells rely on. They are
designed to be cheap so they can run on a CPU pod via cloud_test.py.

The tests do not exercise estimators; that is the job of the matrix
cells. Here we only check the environment itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from econirl.environments.shapeshifter import (
    ShapeshifterConfig,
    ShapeshifterEnvironment,
)
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration


def _spine_config(**overrides) -> ShapeshifterConfig:
    """The Tier-4 spine cell config; every test overrides one axis."""
    base = dict(
        num_states=16,
        num_actions=3,
        num_features=4,
        discount_factor=0.9,
        reward_type="linear",
        feature_type="linear",
        action_dependent=True,
        stochastic_transitions=True,
        stochastic_rewards=False,
        state_dim=1,
        seed=0,
    )
    base.update(overrides)
    return ShapeshifterConfig(**base)


# ---------------------------------------------------------------------------
# Shape and dtype contracts
# ---------------------------------------------------------------------------


def test_spine_shapes() -> None:
    env = ShapeshifterEnvironment(_spine_config())
    assert env.feature_matrix.shape == (16, 3, 4)
    assert env.transition_matrices.shape == (3, 16, 16)
    assert env.true_reward_matrix.shape == (16, 3)
    # Transition rows must sum to 1.
    row_sums = np.asarray(env.transition_matrices).sum(axis=2)
    np.testing.assert_allclose(row_sums, np.ones_like(row_sums), atol=1e-5)


def test_neural_reward_shape() -> None:
    env = ShapeshifterEnvironment(_spine_config(reward_type="neural"))
    assert env.true_reward_matrix.shape == (16, 3)
    # Neural reward has no finite-dim theta, so parameter_names is empty.
    assert env.parameter_names == []


def test_state_only_neural_reward_collapses_action_dim() -> None:
    env = ShapeshifterEnvironment(
        _spine_config(reward_type="neural", action_dependent=False)
    )
    reward = np.asarray(env.true_reward_matrix)
    for a in range(1, reward.shape[1]):
        np.testing.assert_allclose(reward[:, 0], reward[:, a])


def test_neural_features_shape() -> None:
    env = ShapeshifterEnvironment(_spine_config(feature_type="neural"))
    assert env.feature_matrix.shape == (16, 3, 4)


def test_product_state_total() -> None:
    env = ShapeshifterEnvironment(_spine_config(num_states=8, state_dim=2))
    assert env.num_states == 64
    assert env.feature_matrix.shape == (64, 3, 4)
    coords = env.encode_states(np.arange(64))
    assert coords.shape == (64, 2)


# ---------------------------------------------------------------------------
# Determinism / reproducibility
# ---------------------------------------------------------------------------


def test_seed_reproducibility() -> None:
    a = ShapeshifterEnvironment(_spine_config(seed=7))
    b = ShapeshifterEnvironment(_spine_config(seed=7))
    np.testing.assert_array_equal(
        np.asarray(a.feature_matrix), np.asarray(b.feature_matrix)
    )
    np.testing.assert_array_equal(
        np.asarray(a.transition_matrices), np.asarray(b.transition_matrices)
    )
    np.testing.assert_array_equal(
        np.asarray(a.true_reward_matrix), np.asarray(b.true_reward_matrix)
    )


def test_seed_changes_state() -> None:
    a = ShapeshifterEnvironment(_spine_config(seed=1))
    b = ShapeshifterEnvironment(_spine_config(seed=2))
    # At least one of these arrays must differ across seeds.
    same_T = np.array_equal(
        np.asarray(a.transition_matrices), np.asarray(b.transition_matrices)
    )
    same_R = np.array_equal(
        np.asarray(a.true_reward_matrix), np.asarray(b.true_reward_matrix)
    )
    assert not (same_T and same_R)


# ---------------------------------------------------------------------------
# Axis flips: deterministic transitions, state-only features
# ---------------------------------------------------------------------------


def test_deterministic_transitions_are_one_hot() -> None:
    env = ShapeshifterEnvironment(_spine_config(stochastic_transitions=False))
    T = np.asarray(env.transition_matrices)
    # Every row should be a one-hot.
    nonzero = (T > 0).sum(axis=2)
    np.testing.assert_array_equal(nonzero, np.ones_like(nonzero))


def test_state_only_features_collapse_action_dim() -> None:
    env = ShapeshifterEnvironment(_spine_config(action_dependent=False))
    phi = np.asarray(env.feature_matrix)
    # All actions must yield identical features.
    for a in range(1, phi.shape[1]):
        np.testing.assert_array_equal(phi[:, 0, :], phi[:, a, :])


def test_action_dependent_features_differ_across_actions() -> None:
    env = ShapeshifterEnvironment(_spine_config(action_dependent=True))
    phi = np.asarray(env.feature_matrix)
    assert not np.array_equal(phi[:, 0, :], phi[:, 1, :])
    np.testing.assert_array_equal(phi[:, 0, :], np.zeros_like(phi[:, 0, :]))


def test_action_dependent_neural_reward_anchors_action_zero() -> None:
    env = ShapeshifterEnvironment(
        _spine_config(reward_type="neural", action_dependent=True)
    )
    reward = np.asarray(env.true_reward_matrix)
    np.testing.assert_array_equal(reward[:, 0], np.zeros_like(reward[:, 0]))
    assert not np.array_equal(reward[:, 0], reward[:, 1])


# ---------------------------------------------------------------------------
# Identification diagnostics (per CLAUDE.md "Pre-Estimation Diagnostics")
# ---------------------------------------------------------------------------


def test_feature_matrix_full_rank_when_action_dependent() -> None:
    env = ShapeshifterEnvironment(_spine_config(action_dependent=True))
    phi = np.asarray(env.feature_matrix).reshape(-1, env.feature_matrix.shape[-1])
    assert np.linalg.matrix_rank(phi) == phi.shape[1]


def test_feature_matrix_rank_deficient_when_state_only() -> None:
    """State-only features are rank deficient when reshaped to (S*A, K)."""
    env = ShapeshifterEnvironment(_spine_config(action_dependent=False))
    phi = np.asarray(env.feature_matrix).reshape(-1, env.feature_matrix.shape[-1])
    # State-only features have at most S unique rows tiled across A copies,
    # so the reshaped matrix has rank <= K but the "effective" rank for
    # action-discrimination is 0. We check that no column varies across
    # actions for a fixed state, which is the identifiability problem.
    per_state = np.asarray(env.feature_matrix)
    action_variation = (per_state[:, 1:, :] - per_state[:, :1, :]).max(axis=(0, 1))
    np.testing.assert_array_equal(action_variation, np.zeros_like(action_variation))


# ---------------------------------------------------------------------------
# Linear reward = phi @ theta consistency
# ---------------------------------------------------------------------------


def test_linear_reward_matches_features_times_theta() -> None:
    env = ShapeshifterEnvironment(_spine_config(reward_type="linear"))
    theta = env.get_true_parameter_vector()
    R_from_features = env.compute_utility_matrix(theta)
    np.testing.assert_allclose(
        np.asarray(R_from_features),
        np.asarray(env.true_reward_matrix),
        atol=1e-5,
    )


# ---------------------------------------------------------------------------
# Solver consistency on the spine (linear infinite-horizon ground truth)
# ---------------------------------------------------------------------------


def test_hybrid_iteration_converges_on_spine() -> None:
    env = ShapeshifterEnvironment(_spine_config())
    op = SoftBellmanOperator(
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    utility = env.compute_utility_matrix()
    result = hybrid_iteration(op, utility, tol=1e-8, max_iter=2000)
    assert bool(result.converged)
    assert result.policy.shape == (env.num_states, env.num_actions)
    # Each policy row sums to 1.
    np.testing.assert_allclose(
        np.asarray(result.policy.sum(axis=1)),
        np.ones(env.num_states),
        atol=1e-5,
    )


def test_solver_consistent_under_seed() -> None:
    """Same config, same solver, same value function."""
    env_a = ShapeshifterEnvironment(_spine_config(seed=11))
    env_b = ShapeshifterEnvironment(_spine_config(seed=11))
    op_a = SoftBellmanOperator(
        problem=env_a.problem_spec, transitions=env_a.transition_matrices
    )
    op_b = SoftBellmanOperator(
        problem=env_b.problem_spec, transitions=env_b.transition_matrices
    )
    res_a = hybrid_iteration(op_a, env_a.compute_utility_matrix(), tol=1e-8, max_iter=2000)
    res_b = hybrid_iteration(op_b, env_b.compute_utility_matrix(), tol=1e-8, max_iter=2000)
    np.testing.assert_allclose(np.asarray(res_a.V), np.asarray(res_b.V), atol=1e-6)


# ---------------------------------------------------------------------------
# Finite horizon
# ---------------------------------------------------------------------------


def test_finite_horizon_problem_spec_carries_num_periods() -> None:
    env = ShapeshifterEnvironment(_spine_config(num_periods=20))
    assert env.problem_spec.num_periods == 20


def test_finite_horizon_step_terminates() -> None:
    env = ShapeshifterEnvironment(_spine_config(num_periods=5))
    env.reset(seed=0)
    terminated = False
    for _ in range(5):
        _, _, terminated, _, _ = env.step(0)
    assert terminated is True


# ---------------------------------------------------------------------------
# Stochastic rewards add noise on top of the deterministic flow utility
# ---------------------------------------------------------------------------


def test_stochastic_rewards_inject_noise() -> None:
    env = ShapeshifterEnvironment(
        _spine_config(stochastic_rewards=True, stochastic_reward_scale=1.0)
    )
    env.reset(seed=42)
    state = env.current_state
    base = float(env.true_reward_matrix[state, 0])
    samples = []
    for _ in range(100):
        samples.append(env._compute_flow_utility(state, 0))
    sd = float(np.std(samples))
    mean = float(np.mean(samples))
    assert sd > 0.1
    # Mean should be close to base reward (noise is zero-mean Gaussian).
    assert abs(mean - base) < 0.5
