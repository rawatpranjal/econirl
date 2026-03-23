"""Tests for TD-CCP Neural Estimator.

Tests cover:
1. TDCCPConfig default values
2. CCP estimation step
3. Flow decomposition
4. Neural network training convergence
5. Parameter recovery on Rust bus (slow)
"""

import pytest
import torch
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
        assert estimator.name == "TD-CCP Neural"

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
        row_sums = ccps.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

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
# Flow decomposition tests
# ============================================================================


class TestFlowDecomposition:
    """Tests for the flow decomposition step."""

    def test_flow_features_shape(self, rust_env_small, small_panel, problem_spec_small):
        """Flow features should have shape (num_states, num_features)."""
        estimator = TDCCPEstimator()
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        flow_features, flow_entropy = TDCCPEstimator._compute_flow_components(
            ccps, utility.feature_matrix
        )
        assert flow_features.shape == (problem_spec_small.num_states, utility.num_parameters)

    def test_flow_entropy_shape(self, rust_env_small, small_panel, problem_spec_small):
        """Flow entropy should have shape (num_states,)."""
        estimator = TDCCPEstimator()
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        flow_features, flow_entropy = TDCCPEstimator._compute_flow_components(
            ccps, utility.feature_matrix
        )
        assert flow_entropy.shape == (problem_spec_small.num_states,)

    def test_entropy_non_negative(self, rust_env_small, small_panel, problem_spec_small):
        """Entropy should be non-negative."""
        estimator = TDCCPEstimator()
        utility = LinearUtility.from_environment(rust_env_small)
        ccps = estimator._estimate_ccps(
            small_panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        _, flow_entropy = TDCCPEstimator._compute_flow_components(
            ccps, utility.feature_matrix
        )
        assert (flow_entropy >= -1e-8).all()

    def test_uniform_ccps_max_entropy(self):
        """Uniform CCPs should give maximum entropy = log(num_actions)."""
        num_states, num_actions, num_features = 5, 2, 2
        ccps = torch.ones(num_states, num_actions) / num_actions
        features = torch.randn(num_states, num_actions, num_features)
        _, flow_entropy = TDCCPEstimator._compute_flow_components(ccps, features)
        expected = torch.full((num_states,), np.log(num_actions))
        assert torch.allclose(flow_entropy, expected, atol=1e-5)


# ============================================================================
# NN component tests
# ============================================================================


class TestEVComponentNetwork:
    """Tests for the MLP component network."""

    def test_output_shape(self):
        """Network output should be (batch, 1)."""
        net = _EVComponentNetwork(input_dim=1, hidden_dim=32, num_hidden_layers=2)
        x = torch.randn(10, 1)
        out = net(x)
        assert out.shape == (10, 1)

    def test_single_input(self):
        """Network should handle single-sample input."""
        net = _EVComponentNetwork(input_dim=1, hidden_dim=16, num_hidden_layers=1)
        x = torch.randn(1, 1)
        out = net(x)
        assert out.shape == (1, 1)

    def test_gradient_flows(self):
        """Gradients should flow through the network."""
        net = _EVComponentNetwork(input_dim=1, hidden_dim=16, num_hidden_layers=2)
        x = torch.randn(5, 1, requires_grad=True)
        out = net(x).sum()
        out.backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


# ============================================================================
# NN training convergence tests
# ============================================================================


class TestNNTrainingConvergence:
    """Tests that NN training loss decreases over time."""

    def test_loss_decreases(self, rust_env_small, problem_spec_small):
        """Training loss should decrease from first to last epoch."""
        panel = simulate_panel(rust_env_small, n_individuals=50, n_periods=50, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            avi_iterations=5,
            epochs_per_avi=10,
            hidden_dim=32,
            num_hidden_layers=1,
            learning_rate=1e-3,
            batch_size=2048,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        ccps = estimator._estimate_ccps(
            panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        flow_features, flow_entropy = TDCCPEstimator._compute_flow_components(
            ccps, utility.feature_matrix
        )

        states = panel.get_all_states()
        next_states = panel.get_all_next_states()

        # Train on the first feature component
        net, losses = estimator._train_component_network(
            flow=flow_features[:, 0],
            states=states,
            next_states=next_states,
            num_states=problem_spec_small.num_states,
            gamma=problem_spec_small.discount_factor,
        )

        # Average of first 5 losses vs average of last 5 losses
        early_loss = np.mean(losses[:5])
        late_loss = np.mean(losses[-5:])
        assert late_loss < early_loss, (
            f"Loss did not decrease: early={early_loss:.6f}, late={late_loss:.6f}"
        )

    def test_train_all_components_returns_correct_counts(
        self, rust_env_small, problem_spec_small
    ):
        """_train_all_components should return one net per feature plus entropy."""
        panel = simulate_panel(rust_env_small, n_individuals=20, n_periods=20, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            avi_iterations=2,
            epochs_per_avi=3,
            hidden_dim=16,
            num_hidden_layers=1,
            verbose=False,
        )
        estimator = TDCCPEstimator(config=cfg)

        ccps = estimator._estimate_ccps(
            panel, problem_spec_small.num_states, problem_spec_small.num_actions
        )
        flow_features, flow_entropy = TDCCPEstimator._compute_flow_components(
            ccps, utility.feature_matrix
        )

        feature_nets, entropy_net, loss_histories = estimator._train_all_components(
            flow_features=flow_features,
            flow_entropy=flow_entropy,
            panel=panel,
            num_states=problem_spec_small.num_states,
            gamma=problem_spec_small.discount_factor,
        )

        assert len(feature_nets) == utility.num_parameters
        assert entropy_net is not None
        assert len(loss_histories) == utility.num_parameters + 1  # features + entropy


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
            avi_iterations=3,
            epochs_per_avi=5,
            hidden_dim=16,
            num_hidden_layers=1,
            batch_size=512,
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
        assert torch.isfinite(result.parameters).all()
        assert result.policy.shape == (problem_spec_small.num_states, problem_spec_small.num_actions)
        assert result.value_function.shape == (problem_spec_small.num_states,)

    def test_estimate_returns_summary(self, rust_env_small, problem_spec_small, transitions_small):
        """The estimate() method should return an EstimationSummary."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            avi_iterations=3,
            epochs_per_avi=5,
            hidden_dim=16,
            num_hidden_layers=1,
            batch_size=512,
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
        assert summary.method == "TD-CCP Neural"
        assert len(summary.parameters) == utility.num_parameters
        assert summary.parameter_names == utility.parameter_names

    def test_policy_is_valid(self, rust_env_small, problem_spec_small, transitions_small):
        """The estimated policy should be a valid probability distribution."""
        panel = simulate_panel(rust_env_small, n_individuals=30, n_periods=30, seed=42)
        utility = LinearUtility.from_environment(rust_env_small)

        cfg = TDCCPConfig(
            avi_iterations=3,
            epochs_per_avi=5,
            hidden_dim=16,
            num_hidden_layers=1,
            batch_size=512,
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
        row_sums = result.policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


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
        rmse = torch.sqrt((diff ** 2).mean()).item()

        assert rmse < 0.5, (
            f"RMSE={rmse:.4f} exceeds 0.5 threshold. "
            f"True: {true_params.numpy()}, Est: {estimated_params.numpy()}"
        )

        # Also check that individual parameters are in a reasonable range
        assert torch.isfinite(estimated_params).all()
        assert result.log_likelihood < 0  # log-likelihood should be negative
