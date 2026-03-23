"""Tests for MultiComponentBusEnvironment."""

import numpy as np
import pytest
import torch

from econirl.environments.multi_component_bus import MultiComponentBusEnvironment


class TestStateEncoding:
    """State encoding / decoding roundtrip tests."""

    @pytest.mark.parametrize("K,M", [(1, 20), (2, 10), (3, 5)])
    def test_roundtrip_all_states(self, K, M):
        """Encoding then decoding should recover the original state."""
        env = MultiComponentBusEnvironment(K=K, M=M, discount_factor=0.99)
        for s in range(env.num_states):
            components = env.state_to_components(s)
            assert len(components) == K
            assert all(0 <= c < M for c in components), (
                f"Components {components} out of range for M={M}"
            )
            assert env.components_to_state(components) == s

    def test_roundtrip_components(self):
        """Decoding then encoding should recover the original components."""
        env = MultiComponentBusEnvironment(K=3, M=6, discount_factor=0.99)
        for m0 in range(env.M):
            for m1 in range(env.M):
                for m2 in range(env.M):
                    components = [m0, m1, m2]
                    state = env.components_to_state(components)
                    recovered = env.state_to_components(state)
                    assert recovered == components

    def test_state_zero_is_all_zero(self):
        """State 0 should decode to all-zero components."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        assert env.state_to_components(0) == [0, 0]

    def test_last_state(self):
        """The last state should decode to all (M-1) components."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        last = env.num_states - 1
        assert env.state_to_components(last) == [9, 9]


class TestTransitionMatrices:
    """Tests for transition matrix construction and properties."""

    @pytest.mark.parametrize("K,M", [(1, 20), (2, 10), (3, 5)])
    def test_rows_sum_to_one(self, K, M):
        """Every row of the transition matrix must sum to 1."""
        env = MultiComponentBusEnvironment(K=K, M=M, discount_factor=0.99)
        T = env.transition_matrices  # (2, N, N)
        for a in range(2):
            row_sums = T[a].sum(dim=1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
                f"Action {a}: row sums deviate from 1, "
                f"max deviation = {(row_sums - 1).abs().max().item()}"
            )

    @pytest.mark.parametrize("K,M", [(1, 20), (2, 10), (3, 5)])
    def test_non_negative(self, K, M):
        """Transition probabilities must be non-negative."""
        env = MultiComponentBusEnvironment(K=K, M=M, discount_factor=0.99)
        T = env.transition_matrices
        assert (T >= 0).all(), "Found negative transition probabilities"

    def test_replace_rows_identical(self):
        """All rows of the replace transition should be the same."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        T_replace = env.transition_matrices[env.REPLACE]  # (N, N)
        first_row = T_replace[0]
        for s in range(1, env.num_states):
            assert torch.allclose(T_replace[s], first_row, atol=1e-6), (
                f"Replace row {s} differs from row 0"
            )

    def test_replace_equals_keep_from_zero(self):
        """Replace row should equal keep transition from state 0."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        T = env.transition_matrices
        assert torch.allclose(T[env.REPLACE, 0], T[env.KEEP, 0], atol=1e-6)

    def test_shape(self):
        """Transition matrices should have shape (2, N, N)."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        N = env.num_states
        assert env.transition_matrices.shape == (2, N, N)


class TestStateSpaceSize:
    """Tests for correct state-space dimensionality."""

    def test_k1_state_space(self):
        """K=1 should give M states, matching the single-component case."""
        M = 15
        env = MultiComponentBusEnvironment(K=1, M=M, discount_factor=0.99)
        assert env.num_states == M

    def test_k2_state_space(self):
        """K=2 should give M^2 states."""
        M = 10
        env = MultiComponentBusEnvironment(K=2, M=M, discount_factor=0.99)
        assert env.num_states == M ** 2

    def test_k3_state_space(self):
        """K=3 should give M^3 states."""
        M = 5
        env = MultiComponentBusEnvironment(K=3, M=M, discount_factor=0.99)
        assert env.num_states == M ** 3


class TestFeatureMatrix:
    """Tests for the feature matrix."""

    @pytest.mark.parametrize("K,M", [(1, 20), (2, 10), (3, 5)])
    def test_shape(self, K, M):
        """Feature matrix should have shape (N, 2, 3)."""
        env = MultiComponentBusEnvironment(K=K, M=M, discount_factor=0.99)
        assert env.feature_matrix.shape == (env.num_states, 2, 3)

    def test_keep_features_at_state_zero(self):
        """At state 0, x(s) = 0, so keep features are [0, 0, 0]."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        keep_features = env.feature_matrix[0, env.KEEP]
        assert torch.allclose(keep_features, torch.zeros(3))

    def test_replace_features_constant(self):
        """Replace features should be [-1, 0, 0] for every state."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        for s in range(env.num_states):
            replace_f = env.feature_matrix[s, env.REPLACE]
            expected = torch.tensor([-1.0, 0.0, 0.0])
            assert torch.allclose(replace_f, expected), (
                f"State {s}: replace features {replace_f} != {expected}"
            )

    def test_keep_features_match_state_feature(self):
        """Keep features should be [0, -x(s), -x(s)^2]."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        for s in range(env.num_states):
            x = env._state_feature(s)
            expected = torch.tensor([0.0, -x, -(x ** 2)])
            actual = env.feature_matrix[s, env.KEEP]
            assert torch.allclose(actual, expected, atol=1e-6), (
                f"State {s}: keep features {actual} != {expected}"
            )

    def test_utility_matches_flow_utility(self):
        """Utility via feature matrix should match _compute_flow_utility."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        params = env.get_true_parameter_vector()
        utility_matrix = env.compute_utility_matrix()
        for s in range(env.num_states):
            for a in range(2):
                expected = env._compute_flow_utility(s, a)
                actual = utility_matrix[s, a].item()
                assert abs(actual - expected) < 1e-5, (
                    f"State {s}, action {a}: {actual} != {expected}"
                )


class TestSimulation:
    """Tests for simulation with the multi-component environment."""

    def test_simulate_panel(self):
        """simulate_panel should work and return the right structure."""
        from econirl.simulation.synthetic import simulate_panel

        env = MultiComponentBusEnvironment(
            K=2, M=10, discount_factor=0.99, seed=42
        )
        panel = simulate_panel(env, n_individuals=5, n_periods=20, seed=42)

        assert panel.num_individuals == 5
        assert panel.num_observations == 5 * 20

        # All states should be in valid range
        all_states = panel.get_all_states()
        assert (all_states >= 0).all()
        assert (all_states < env.num_states).all()

        # All actions should be 0 or 1
        all_actions = panel.get_all_actions()
        assert set(all_actions.numpy().tolist()).issubset({0, 1})

    def test_step_and_reset(self):
        """Basic step / reset should work."""
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99, seed=0)
        obs, info = env.reset(seed=0)
        assert 0 <= obs < env.num_states

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            assert 0 <= obs < env.num_states
            assert not terminated
            assert not truncated


class TestParameterValidation:
    """Tests for constructor parameter validation."""

    def test_k_too_large(self):
        """K >= 4 should raise ValueError."""
        with pytest.raises(ValueError, match="K must be <= 3"):
            MultiComponentBusEnvironment(K=4, M=5, discount_factor=0.99)

    def test_k_too_small(self):
        """K < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="K must be >= 1"):
            MultiComponentBusEnvironment(K=0, M=5, discount_factor=0.99)

    def test_m_too_small(self):
        """M < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="M must be >= 2"):
            MultiComponentBusEnvironment(K=2, M=1, discount_factor=0.99)

    def test_bad_transition_probs(self):
        """Transition probs not summing to 1 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1"):
            MultiComponentBusEnvironment(
                K=2, M=5,
                mileage_transition_probs=(0.5, 0.3, 0.1),
                discount_factor=0.99,
            )


class TestProperties:
    """Tests for basic properties and accessors."""

    def test_parameter_names(self):
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        assert env.parameter_names == [
            "replacement_cost", "operating_cost", "quadratic_cost"
        ]

    def test_true_parameters(self):
        env = MultiComponentBusEnvironment(
            K=2, M=10,
            operating_cost=0.002,
            quadratic_cost=0.001,
            replacement_cost=5.0,
            discount_factor=0.99,
        )
        assert env.true_parameters == {
            "replacement_cost": 5.0,
            "operating_cost": 0.002,
            "quadratic_cost": 0.001,
        }

    def test_problem_spec(self):
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        spec = env.problem_spec
        assert spec.num_states == 100
        assert spec.num_actions == 2
        assert spec.discount_factor == 0.99

    def test_k_and_m_accessors(self):
        env = MultiComponentBusEnvironment(K=3, M=7, discount_factor=0.99)
        assert env.K == 3
        assert env.M == 7

    def test_describe(self):
        env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
        desc = env.describe()
        assert "Multi-Component" in desc
        assert "100" in desc


class TestK1MatchesRust:
    """When K=1, transitions should match the single-component Rust bus."""

    def test_keep_transition_matches(self):
        """K=1 keep transition should match RustBusEnvironment's keep."""
        from econirl.environments.rust_bus import RustBusEnvironment

        M = 20
        probs = (0.3919, 0.5953, 0.0128)
        env_multi = MultiComponentBusEnvironment(
            K=1, M=M, mileage_transition_probs=probs, discount_factor=0.99
        )
        env_rust = RustBusEnvironment(
            num_mileage_bins=M, mileage_transition_probs=probs, discount_factor=0.99
        )

        T_multi = env_multi.transition_matrices  # (2, M, M)
        T_rust = env_rust.transition_matrices  # (2, M, M)

        assert torch.allclose(T_multi[0], T_rust[0], atol=1e-5), (
            "K=1 keep transition does not match Rust bus"
        )

    def test_replace_transition_matches(self):
        """K=1 replace transition should match RustBusEnvironment's replace."""
        from econirl.environments.rust_bus import RustBusEnvironment

        M = 20
        probs = (0.3919, 0.5953, 0.0128)
        env_multi = MultiComponentBusEnvironment(
            K=1, M=M, mileage_transition_probs=probs, discount_factor=0.99
        )
        env_rust = RustBusEnvironment(
            num_mileage_bins=M, mileage_transition_probs=probs, discount_factor=0.99
        )

        T_multi = env_multi.transition_matrices
        T_rust = env_rust.transition_matrices

        assert torch.allclose(T_multi[1], T_rust[1], atol=1e-5), (
            "K=1 replace transition does not match Rust bus"
        )
