"""Tests for the ObjectworldEnvironment."""

import numpy as np
import pytest
import torch

from econirl.environments.objectworld import ObjectworldEnvironment, _build_grid_transitions
from econirl.core.types import DDCProblem, Panel


class TestConstruction:
    """Basic construction and property tests."""

    def test_default_construction(self):
        """Default ObjectworldEnvironment should construct without error."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        assert env.num_states == 64
        assert env.num_actions == 5

    def test_num_states_matches_grid(self):
        """num_states should equal grid_size squared."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        assert env.num_states == 64

    def test_num_actions_is_five(self):
        """There should be exactly 5 actions."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        assert env.num_actions == 5


class TestTransitions:
    """Tests for the deterministic transition matrices."""

    def test_shape(self):
        """Transition matrices should have shape (5, 64, 64) for grid_size=8."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        T = env.transition_matrices
        assert T.shape == (5, 64, 64)

    def test_rows_sum_to_one(self):
        """All rows should sum to 1."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        T = env.transition_matrices
        for a in range(5):
            row_sums = T[a].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(64))

    def test_deterministic(self):
        """Each row should have a single 1.0 entry (deterministic transitions)."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        T = env.transition_matrices
        for a in range(5):
            assert T[a].max(dim=1).values.min().item() == pytest.approx(1.0)


class TestContinuousFeatures:
    """Tests for continuous feature type."""

    def test_shape(self):
        """Continuous features should have shape (64, 5, 2) for C=2."""
        env = ObjectworldEnvironment(grid_size=8, n_colors=2, seed=0,
                                     feature_type="continuous")
        F = env.feature_matrix
        assert F.shape == (64, 5, 2)

    def test_values_in_unit_interval(self):
        """Continuous features should be in [0, 1]."""
        env = ObjectworldEnvironment(grid_size=8, n_colors=2, seed=0,
                                     feature_type="continuous")
        F = env.feature_matrix
        assert F.min().item() >= 0.0
        assert F.max().item() <= 1.0


class TestDiscreteFeatures:
    """Tests for discrete (binary indicator) feature type."""

    def test_shape(self):
        """Discrete features should have shape (64, 5, 16) for C=2, M=8."""
        env = ObjectworldEnvironment(grid_size=8, n_colors=2, seed=0,
                                     feature_type="discrete",
                                     max_distance=8)
        F = env.feature_matrix
        assert F.shape == (64, 5, 16)

    def test_binary_values(self):
        """Discrete features should contain only 0 and 1."""
        env = ObjectworldEnvironment(grid_size=8, n_colors=2, seed=0,
                                     feature_type="discrete",
                                     max_distance=8)
        F = env.feature_matrix
        unique_vals = torch.unique(F)
        assert all(v in [0.0, 1.0] for v in unique_vals.tolist())


class TestReward:
    """Tests for the reward function."""

    def test_shape(self):
        """true_reward should have shape (num_states,)."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        R = env.true_reward
        assert R.shape == (64,)

    def test_values_in_set(self):
        """Reward values should be in {-1, 0, 1}."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        R = env.true_reward
        for val in R:
            assert val.item() in {-1.0, 0.0, 1.0}


class TestSeedReproducibility:
    """Tests for random seed behavior."""

    def test_same_seed_same_result(self):
        """Same seed should produce identical reward and features."""
        env1 = ObjectworldEnvironment(grid_size=8, seed=42)
        env2 = ObjectworldEnvironment(grid_size=8, seed=42)
        assert torch.equal(env1.true_reward, env2.true_reward)
        assert torch.equal(env1.feature_matrix, env2.feature_matrix)

    def test_different_seed_different_result(self):
        """Different seeds should produce different rewards."""
        env1 = ObjectworldEnvironment(grid_size=8, seed=0)
        env2 = ObjectworldEnvironment(grid_size=8, seed=999)
        assert not torch.equal(env1.true_reward, env2.true_reward)


class TestProblemSpec:
    """Tests for the DDCProblem specification."""

    def test_problem_spec(self):
        """problem_spec should have correct fields."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        spec = env.problem_spec
        assert isinstance(spec, DDCProblem)
        assert spec.num_states == 64
        assert spec.num_actions == 5
        assert spec.discount_factor == 0.9


class TestDemonstrations:
    """Tests for the simulate_demonstrations method."""

    def test_produces_valid_panel(self):
        """simulate_demonstrations should return a Panel with valid data."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        panel = env.simulate_demonstrations(n_demos=4, max_steps=50, seed=0)
        assert isinstance(panel, Panel)
        assert panel.num_individuals == 4

    def test_states_and_actions_in_range(self):
        """Demonstrated states and actions should be within valid ranges."""
        env = ObjectworldEnvironment(grid_size=8, seed=0)
        panel = env.simulate_demonstrations(n_demos=4, max_steps=50, seed=0)
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        assert (all_states >= 0).all()
        assert (all_states < 64).all()
        assert (all_actions >= 0).all()
        assert (all_actions < 5).all()


class TestBuildGridTransitions:
    """Tests for the module-level _build_grid_transitions helper."""

    def test_shape(self):
        """Helper should return shape (5, S, S) for a grid."""
        T = _build_grid_transitions(4)
        assert T.shape == (5, 16, 16)

    def test_rows_sum_to_one(self):
        """All rows should sum to 1."""
        T = _build_grid_transitions(4)
        for a in range(5):
            row_sums = T[a].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(16))

    def test_no_absorbing_state(self):
        """Unlike GridworldEnvironment, no state should be absorbing for all actions.

        For a non-corner, non-edge state, at least some actions should move
        the agent to a different state.
        """
        T = _build_grid_transitions(4)
        # State 5 (row=1, col=1) in a 4x4 grid is interior.
        # Left should go to state 4, Right to 6, Up to 1, Down to 9.
        assert T[0, 5, 4].item() == pytest.approx(1.0)  # Left
        assert T[1, 5, 6].item() == pytest.approx(1.0)  # Right
        assert T[2, 5, 1].item() == pytest.approx(1.0)  # Up
        assert T[3, 5, 9].item() == pytest.approx(1.0)  # Down
        assert T[4, 5, 5].item() == pytest.approx(1.0)  # Stay
