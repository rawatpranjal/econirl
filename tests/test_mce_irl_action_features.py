"""Test MCE IRL with action-dependent features."""
import pytest
import jax.numpy as jnp
import numpy as np
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.simulation import simulate_panel
from econirl.core.types import DDCProblem, Panel, Trajectory


class TestMCEIRLActionFeatures:
    """Tests for MCE IRL with action-dependent features."""

    @pytest.fixture
    def rust_env(self):
        return RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.99,  # Lower for faster convergence
        )

    def test_mce_irl_handles_3d_features(self, rust_env):
        """MCE IRL should correctly handle 3D feature matrices."""
        panel = simulate_panel(rust_env, n_individuals=50, n_periods=20, seed=123)

        reward = ActionDependentReward.from_rust_environment(rust_env)

        # Verify feature matrix is 3D
        assert reward.feature_matrix.ndim == 3, "Feature matrix should be 3D"
        assert reward.feature_matrix.shape == (90, 2, 2), (
            f"Expected shape (90, 2, 2), got {reward.feature_matrix.shape}"
        )

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=100,
            learning_rate=0.1,
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)

        # Should not raise an error
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # Parameters should be finite
        assert jnp.isfinite(result.parameters).all(), "Parameters should be finite"

    def test_mce_irl_feature_expectations(self, rust_env):
        """Test that feature expectations are computed correctly for 3D features."""
        panel = simulate_panel(rust_env, n_individuals=100, n_periods=30, seed=456)

        reward = ActionDependentReward.from_rust_environment(rust_env)

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=50,
            learning_rate=0.1,
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
        )

        # Check that feature expectations are stored in metadata
        assert "empirical_features" in result.metadata, "Should store empirical features"
        assert "final_expected_features" in result.metadata, "Should store expected features"

        # Empirical features should have correct dimension
        empirical = result.metadata["empirical_features"]
        assert len(empirical) == 2, f"Expected 2 features, got {len(empirical)}"

    def test_mce_irl_with_warm_start(self, rust_env):
        """MCE IRL with warm start from true params should stay close."""
        panel = simulate_panel(rust_env, n_individuals=200, n_periods=50, seed=42)

        reward = ActionDependentReward.from_rust_environment(rust_env)
        true_params = rust_env.get_true_parameter_vector()

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=100,
            learning_rate=0.001,  # Small learning rate for stability
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
            initial_params=true_params,
        )

        # With finite Rust panels, discounted MCE moments have sampling and
        # truncation error; the warm start should still keep the moment gap
        # controlled relative to the empirical feature norm.
        feature_diff = result.metadata.get("feature_difference", float("inf"))
        empirical_norm = float(jnp.linalg.norm(jnp.asarray(result.metadata["empirical_features"])))
        relative_feature_diff = feature_diff / max(empirical_norm, 1e-12)
        assert relative_feature_diff < 0.30, (
            f"Feature difference too large: {feature_diff} "
            f"(relative {relative_feature_diff:.3f})"
        )

        # Parameters should stay close to true values (within factor of 5)
        # Note: MCE IRL identifies parameters up to scale, so exact match not expected
        for i, (est, true) in enumerate(zip(result.parameters, true_params)):
            ratio = float(est) / float(true) if float(true) != 0 else float("inf")
            assert 0.1 < ratio < 10, (
                f"Parameter {i} ratio out of range: estimated={float(est):.6f}, "
                f"true={float(true):.6f}, ratio={ratio:.2f}"
            )

    def test_mce_irl_recovers_ratio(self, rust_env):
        """MCE IRL should recover the correct ratio of parameters."""
        # Use more data for better ratio recovery
        panel = simulate_panel(rust_env, n_individuals=300, n_periods=100, seed=42)

        reward = ActionDependentReward.from_rust_environment(rust_env)
        true_params = rust_env.get_true_parameter_vector()

        config = MCEIRLConfig(
            verbose=False,
            inner_max_iter=10000,
            outer_max_iter=200,
            learning_rate=0.001,
            compute_se=False,
        )

        estimator = MCEIRLEstimator(config=config)
        result = estimator.estimate(
            panel=panel,
            utility=reward,
            problem=rust_env.problem_spec,
            transitions=rust_env.transition_matrices,
            initial_params=true_params,  # Warm start
        )

        # Check that the ratio theta_c/RC is recovered correctly
        # This is what MCE IRL can identify (up to scale)
        estimated_ratio = result.parameters[0] / result.parameters[1]
        true_ratio = true_params[0] / true_params[1]

        # Allow 50% relative error on the ratio (IRL is hard!)
        relative_error = abs(estimated_ratio - true_ratio) / abs(true_ratio)
        assert relative_error < 0.5, (
            f"Ratio mismatch: estimated={float(estimated_ratio):.6f}, "
            f"true={float(true_ratio):.6f}, relative_error={float(relative_error):.2%}"
        )


def test_empirical_features_use_discounted_state_action_occupancy():
    """Empirical moments for (S,A,K) features must use D_demo(s,a)."""

    features = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
            [[3.0, 0.0], [0.0, 3.0]],
        ],
        dtype=jnp.float32,
    )
    reward = ActionDependentReward(features, ["left", "right"])
    panel = Panel(
        trajectories=[
            Trajectory(
                states=jnp.array([0, 1, 1], dtype=jnp.int32),
                actions=jnp.array([0, 1, 0], dtype=jnp.int32),
                next_states=jnp.array([1, 1, 2], dtype=jnp.int32),
            )
        ]
    )
    estimator = MCEIRLEstimator(MCEIRLConfig(compute_se=False))

    empirical = estimator._compute_empirical_features(
        panel,
        reward,
        n_states=3,
        n_actions=2,
        discount=0.5,
    )
    weights = jnp.array([1.0, 0.5, 0.25])
    expected = (
        weights[0] * features[0, 0]
        + weights[1] * features[1, 1]
        + weights[2] * features[1, 0]
    ) / weights.sum()

    assert jnp.allclose(empirical, expected, atol=1e-7)


def test_expected_features_match_analytic_discounted_occupancy():
    """Expected MCE moments should equal analytic discounted occupancy moments."""

    transitions = jnp.zeros((2, 2, 2), dtype=jnp.float32)
    transitions = transitions.at[0, :, 0].set(1.0)
    transitions = transitions.at[1, :, 1].set(1.0)
    policy = jnp.array([[0.75, 0.25], [0.20, 0.80]], dtype=jnp.float32)
    initial_dist = jnp.array([1.0, 0.0], dtype=jnp.float32)
    problem = DDCProblem(num_states=2, num_actions=2, discount_factor=0.5)
    features = jnp.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ],
        dtype=jnp.float32,
    )
    reward = ActionDependentReward(features, ["a0", "a1"])
    panel = Panel(
        trajectories=[
            Trajectory(
                states=jnp.array([0], dtype=jnp.int32),
                actions=jnp.array([0], dtype=jnp.int32),
                next_states=jnp.array([0], dtype=jnp.int32),
            )
        ]
    )
    estimator = MCEIRLEstimator(MCEIRLConfig(compute_se=False))

    expected = estimator._compute_expected_features(
        panel,
        policy,
        reward,
        transitions=transitions,
        initial_dist=initial_dist,
        discount=problem.discount_factor,
    )
    d = estimator._compute_state_visitation(
        policy,
        transitions,
        problem,
        initial_dist,
    )
    analytic = jnp.einsum("s,sa,sak->k", d, policy, features)

    assert jnp.allclose(expected, analytic, atol=1e-7)
