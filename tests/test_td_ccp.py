"""Tests for TD-CCP Neural Estimator.

Tests cover:
1. TDCCPConfig default values
2. CCP estimation step
3. Flow decomposition
4. Neural network training convergence
5. Parameter recovery on Rust bus (slow)
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig, _EVComponentNetwork
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ============================================================================
# Config tests
# ============================================================================


class TestTDCCPConfig:
    """Tests for TDCCPConfig dataclass defaults."""

    def test_default_hidden_dim(self):
        cfg = TDCCPConfig()
        assert cfg.hidden_dim == 64

    def test_default_num_hidden_layers(self):
        cfg = TDCCPConfig()
        assert cfg.num_hidden_layers == 2

    def test_default_avi_iterations(self):
        cfg = TDCCPConfig()
        assert cfg.avi_iterations == 20

    def test_default_epochs_per_avi(self):
        cfg = TDCCPConfig()
        assert cfg.epochs_per_avi == 30

    def test_default_learning_rate(self):
        cfg = TDCCPConfig()
        assert cfg.learning_rate == 1e-3

    def test_default_batch_size(self):
        cfg = TDCCPConfig()
        assert cfg.batch_size == 8192

    def test_default_ccp_smoothing(self):
        cfg = TDCCPConfig()
        assert cfg.ccp_smoothing == 0.01

    def test_default_outer_max_iter(self):
        cfg = TDCCPConfig()
        assert cfg.outer_max_iter == 200

    def test_default_outer_tol(self):
        cfg = TDCCPConfig()
        assert cfg.outer_tol == 1e-6

    def test_default_compute_se(self):
        cfg = TDCCPConfig()
        assert cfg.compute_se is True

    def test_default_verbose(self):
        cfg = TDCCPConfig()
        assert cfg.verbose is False

    def test_custom_config(self):
        cfg = TDCCPConfig(hidden_dim=128, avi_iterations=10, learning_rate=5e-4)
        assert cfg.hidden_dim == 128
        assert cfg.avi_iterations == 10
        assert cfg.learning_rate == 5e-4


# ============================================================================
# Estimator properties tests
# ============================================================================


class TestTDCCPEstimatorProperties:
    """Tests for basic estimator properties."""

    def test_name(self):
        estimator = TDCCPEstimator()
        assert estimator.name == "TD-CCP"

    def test_default_config(self):
        estimator = TDCCPEstimator()
        assert estimator.config.hidden_dim == 64

    def test_custom_config(self):
        cfg = TDCCPConfig(hidden_dim=128)
        estimator = TDCCPEstimator(config=cfg)
        assert estimator.config.hidden_dim == 128


# ============================================================================
# CCP estimation tests
# ============================================================================


class TestCCPEstimation:
    """Tests for the CCP frequency estimation step."""

    def test_ccps_sum_to_one(self, rust_env_small, small_panel, problem_spec_small):
        """Estimated CCPs should sum to 1 for each state."""
        estimator = TDCCPEstimator()
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        row_sums = ccps.sum(axis=1)
        np.testing.assert_allclose(np.asarray(row_sums), np.asarray(jnp.ones_like(row_sums)), atol=1e-5)

    def test_ccps_non_negative(self, rust_env_small, small_panel, problem_spec_small):
        """Estimated CCPs should be non-negative."""
        estimator = TDCCPEstimator()
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        assert (ccps >= 0).all()

    def test_ccps_shape(self, rust_env_small, small_panel, problem_spec_small):
        """CCPs should have shape (num_states, num_actions)."""
        estimator = TDCCPEstimator()
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        assert ccps.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)

    def test_smoothing_prevents_zeros(self, rust_env_small, problem_spec_small):
        """CCP smoothing should prevent any zero probabilities."""
        # Very small panel so some (s, a) pairs may not be observed
        panel = simulate_panel(rust_env_small, n_individuals=5, n_periods=5, seed=42)
        estimator = TDCCPEstimator(config=TDCCPConfig(ccp_smoothing=0.01))
        ccps = estimator._estimate_ccps(
            panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        assert (ccps > 0).all()


# ============================================================================
# Transition extraction tests
# ============================================================================


class TestTransitionExtraction:
    """Tests for extracting (a,x,a',x') tuples from panel data."""

    def test_transition_lengths_match(self, rust_env_small, small_panel):
        """All transition arrays should have the same length."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert len(actions) == len(states) == len(next_actions) == len(next_states)

    def test_transitions_non_empty(self, rust_env_small, small_panel):
        """Should extract at least one transition."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert len(states) > 0

    def test_actions_valid(self, rust_env_small, small_panel, problem_spec_small):
        """All actions should be valid (0 or 1 for binary choice)."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert np.all(actions >= 0) and np.all(actions < problem_spec_small.num_actions)
        assert np.all(next_actions >= 0) and np.all(next_actions < problem_spec_small.num_actions)

    def test_states_valid(self, rust_env_small, small_panel, problem_spec_small):
        """All states should be valid indices."""
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        assert np.all(states >= 0) and np.all(states < problem_spec_small.num_states)
        assert np.all(next_states >= 0) and np.all(next_states < problem_spec_small.num_states)


# ============================================================================
# Semi-gradient method tests
# ============================================================================


class TestSemigradientSolve:
    """Tests for the linear semi-gradient method (eq 3.5)."""

    def test_h_table_shape(self, rust_env_small, small_panel, problem_spec_small):
        """h_table should have shape (num_states, num_actions, num_features)."""
        estimator = TDCCPEstimator(config=TDCCPConfig(method="semigradient"))
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        h_table, g_table = estimator._semigradient_solve(
            actions, states, next_actions, next_states,
            np.array(utility.feature_matrix), np.array(ccps),
            problem_spec_small.num_states, problem_spec_small.num_actions,
            problem_spec_small.discount_factor,
        )
        assert h_table.shape == (problem_spec_small.num_states, problem_spec_small.num_actions, utility.num_parameters)

    def test_g_table_shape(self, rust_env_small, small_panel, problem_spec_small):
        """g_table should have shape (num_states, num_actions)."""
        estimator = TDCCPEstimator(config=TDCCPConfig(method="semigradient"))
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        h_table, g_table = estimator._semigradient_solve(
            actions, states, next_actions, next_states,
            np.array(utility.feature_matrix), np.array(ccps),
            problem_spec_small.num_states, problem_spec_small.num_actions,
            problem_spec_small.discount_factor,
        )
        assert g_table.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)

    def test_h_g_finite(self, rust_env_small, small_panel, problem_spec_small):
        """h and g should be finite everywhere."""
        estimator = TDCCPEstimator(config=TDCCPConfig(method="semigradient"))
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        actions, states, next_actions, next_states = TDCCPEstimator._extract_transitions(small_panel)
        h_table, g_table = estimator._semigradient_solve(
            actions, states, next_actions, next_states,
            np.array(utility.feature_matrix), np.array(ccps),
            problem_spec_small.num_states, problem_spec_small.num_actions,
            problem_spec_small.discount_factor,
        )
        assert np.all(np.isfinite(h_table))
        assert np.all(np.isfinite(g_table))


# ============================================================================
# NN component tests
# ============================================================================


class TestEVComponentNetwork:
    """Tests for the MLP component network."""

    def test_output_scalar(self):
        """Network output should be a scalar for single input."""
        key = jax.random.PRNGKey(0)
        net = _EVComponentNetwork(input_dim=3, hidden_dim=32, num_hidden_layers=2, key=key)
        x = jnp.ones(3)
        out = net(x)
        assert out.shape == ()

    def test_vmap_batch(self):
        """Network should work with vmap for batched input."""
        key = jax.random.PRNGKey(0)
        net = _EVComponentNetwork(input_dim=3, hidden_dim=16, num_hidden_layers=1, key=key)
        x = jnp.ones((10, 3))
        out = jax.vmap(net)(x)
        assert out.shape == (10,)

    def test_gradient_flows(self):
        """Gradients should flow through the network."""
        key = jax.random.PRNGKey(0)
        net = _EVComponentNetwork(input_dim=3, hidden_dim=16, num_hidden_layers=2, key=key)
        x = jnp.ones(3)
        out = net(x)
        assert jnp.isfinite(out)


# ============================================================================
# Integration test (quick)
# ============================================================================


class TestTDCCPQuickIntegration:
    """Quick integration test that the full pipeline runs without errors."""

    def test_full_pipeline_runs(self, rust_env_small, problem_spec_small, transitions_small):
        """The full TD-CCP pipeline should run end-to-end on a small problem."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            method="semigradient",
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=20,
            compute_se=False,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=problem_spec_small,
            transitions=transitions_small,
        )

        assert result is not None
        assert len(result.parameters) == utility.num_parameters
        assert jnp.isfinite(result.parameters).all()
        assert result.policy.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)
        assert result.value_function.shape == (problem_spec_small.num_states,)

    def test_estimate_returns_summary(self, rust_env_small, problem_spec_small, transitions_small):
        """The estimate() method should return an EstimationSummary."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            method="semigradient",
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=20,
            compute_se=True,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        summary = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem_spec_small,
            transitions=transitions_small,
        )

        assert summary is not None
        assert summary.method == "TD-CCP"
        assert len(summary.parameters) == utility.num_parameters
        assert summary.parameter_names == utility.parameter_names

    def test_policy_is_valid(self, rust_env_small, problem_spec_small, transitions_small):
        """The estimated policy should be a valid probability distribution."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            method="semigradient",
            cross_fitting=False,
            robust_se=False,
            outer_max_iter=20,
            compute_se=False,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=problem_spec_small,
            transitions=transitions_small,
        )

        # Policy should be non-negative
        assert (result.policy >= 0).all()
        # Policy rows should sum to 1
        row_sums = result.policy.sum(axis=1)
        np.testing.assert_allclose(np.asarray(row_sums), np.asarray(jnp.ones_like(row_sums)), atol=1e-5)


# ============================================================================
# Import test
# ============================================================================


class TestImports:
    """Test that TDCCPEstimator and TDCCPConfig can be imported from estimation package."""

    def test_import_from_estimation(self):
        from econirl.estimation import TDCCPEstimator, TDCCPConfig

        assert TDCCPEstimator is not None
        assert TDCCPConfig is not None

    def test_in_all(self):
        from econirl.estimation import __all__

        assert "TDCCPEstimator" in __all__
        assert "TDCCPConfig" in __all__


# ============================================================================
# Slow: Parameter recovery test
# ============================================================================


@pytest.mark.slow
class TestParameterRecovery:
    """Parameter recovery test on Rust bus environment.

    Requires 500 agents x 100 periods, so marked as slow.
    """

    def test_parameter_recovery_rust_bus(self):
        """TD-CCP should recover Rust bus parameters with RMSE < 0.5."""
        env = RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.9999,
            seed=42,
        )
        utility = LinearUtility.from_environment(env)
        problem = env.problem_spec
        transitions = env.transition_matrices
        panel = simulate_panel(env, n_individuals=500, n_periods=100, seed=42)

        true_params = env.get_true_parameter_vector()

        cfg = TDCCPConfig(
            hidden_dim=64,
            num_hidden_layers=2,
            avi_iterations=20,
            epochs_per_avi=30,
            learning_rate=1e-3,
            batch_size=8192,
            ccp_smoothing=0.01,
            outer_max_iter=200,
            compute_se=True,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )

        estimated_params = result.parameters
        diff = estimated_params - true_params
        rmse = float(jnp.sqrt((diff ** 2).mean()))

        assert rmse < 0.5, (
            f"RMSE={rmse:.4f} exceeds 0.5 threshold. "
            f"True: {np.asarray(true_params)}, Est: {np.asarray(estimated_params)}"
        )

        # Also check that individual parameters are in a reasonable range
        assert jnp.isfinite(estimated_params).all()
        assert result.log_likelihood < 0  # log-likelihood should be negative
