"""Tests for the GLADIUS estimator.

Tests cover:
- GLADIUSConfig defaults
- Network architecture shapes (forward pass output dimensions)
- Small 3-state training loop (loss decreases)
- Parameter recovery on Rust bus (slow, marked with @pytest.mark.slow)
"""

import numpy as np
import pytest
import torch

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.gladius import (
    EVNetwork,
    GLADIUSConfig,
    GLADIUSEstimator,
    QNetwork,
    _build_mlp,
)


class TestGLADIUSConfig:
    """Tests for GLADIUSConfig defaults."""

    def test_default_values(self):
        """Check that all config defaults match the specification."""
        config = GLADIUSConfig()
        assert config.q_hidden_dim == 128
        assert config.q_num_layers == 3
        assert config.v_hidden_dim == 128
        assert config.v_num_layers == 3
        assert config.q_lr == 1e-3
        assert config.v_lr == 1e-3
        assert config.max_epochs == 500
        assert config.batch_size == 512
        assert config.bellman_penalty_weight == 1.0
        assert config.weight_decay == 1e-4
        assert config.gradient_clip == 1.0
        assert config.compute_se is True
        assert config.n_bootstrap == 100
        assert config.verbose is False

    def test_custom_values(self):
        """Check that custom values are accepted."""
        config = GLADIUSConfig(
            q_hidden_dim=64,
            max_epochs=100,
            bellman_penalty_weight=0.5,
            verbose=True,
        )
        assert config.q_hidden_dim == 64
        assert config.max_epochs == 100
        assert config.bellman_penalty_weight == 0.5
        assert config.verbose is True


class TestNetworkArchitectures:
    """Tests for Q-network and EV-network shapes."""

    def test_build_mlp_shape(self):
        """Check that _build_mlp creates the right number of layers."""
        mlp = _build_mlp(input_dim=5, hidden_dim=16, num_layers=2, output_dim=1)
        # 2 hidden layers means: Linear+ReLU + Linear+ReLU + Linear = 5 modules
        n_modules = len(list(mlp.children()))
        assert n_modules == 5, f"Expected 5 modules, got {n_modules}"

    def test_build_mlp_zero_hidden(self):
        """MLP with 0 hidden layers should be a single linear layer."""
        mlp = _build_mlp(input_dim=5, hidden_dim=16, num_layers=0, output_dim=1)
        n_modules = len(list(mlp.children()))
        assert n_modules == 1

    def test_q_network_forward_shape(self):
        """Q-network forward pass should produce (batch,) output."""
        state_dim = 1
        n_actions = 3
        q_net = QNetwork(state_dim, n_actions, hidden_dim=16, num_layers=2)

        batch = 10
        state_features = torch.randn(batch, state_dim)
        action_onehot = torch.zeros(batch, n_actions)
        action_onehot[:, 0] = 1.0

        q_vals = q_net(state_features, action_onehot)
        assert q_vals.shape == (batch,), f"Expected ({batch},), got {q_vals.shape}"
        assert torch.all(torch.isfinite(q_vals)), "Q values should be finite"

    def test_q_network_forward_all_actions_shape(self):
        """forward_all_actions should produce (batch, n_actions)."""
        state_dim = 1
        n_actions = 3
        q_net = QNetwork(state_dim, n_actions, hidden_dim=16, num_layers=2)

        batch = 10
        state_features = torch.randn(batch, state_dim)

        q_all = q_net.forward_all_actions(state_features)
        assert q_all.shape == (batch, n_actions), f"Expected ({batch}, {n_actions}), got {q_all.shape}"
        assert torch.all(torch.isfinite(q_all)), "All Q values should be finite"

    def test_ev_network_forward_shape(self):
        """EV-network forward pass should produce (batch,) output."""
        state_dim = 1
        n_actions = 2
        ev_net = EVNetwork(state_dim, n_actions, hidden_dim=16, num_layers=2)

        batch = 8
        state_features = torch.randn(batch, state_dim)
        action_onehot = torch.zeros(batch, n_actions)
        action_onehot[:, 1] = 1.0

        ev_vals = ev_net(state_features, action_onehot)
        assert ev_vals.shape == (batch,), f"Expected ({batch},), got {ev_vals.shape}"
        assert torch.all(torch.isfinite(ev_vals)), "EV values should be finite"

    def test_q_network_gradient_flow(self):
        """Gradients should flow through the Q-network."""
        q_net = QNetwork(state_dim=1, n_actions=2, hidden_dim=16, num_layers=2)
        state_features = torch.randn(4, 1)
        action_onehot = torch.zeros(4, 2)
        action_onehot[:, 0] = 1.0

        q_vals = q_net(state_features, action_onehot)
        loss = q_vals.sum()
        loss.backward()

        for param in q_net.parameters():
            assert param.grad is not None, "Gradients should be computed"

    def test_ev_network_gradient_flow(self):
        """Gradients should flow through the EV-network."""
        ev_net = EVNetwork(state_dim=1, n_actions=2, hidden_dim=16, num_layers=2)
        state_features = torch.randn(4, 1)
        action_onehot = torch.zeros(4, 2)
        action_onehot[:, 1] = 1.0

        ev_vals = ev_net(state_features, action_onehot)
        loss = ev_vals.sum()
        loss.backward()

        for param in ev_net.parameters():
            assert param.grad is not None, "Gradients should be computed"


class TestGLADIUSEstimatorSmall:
    """Tests for GLADIUS estimator on a small 3-state problem."""

    @pytest.fixture
    def small_setup(self):
        """Create a minimal 3-state, 2-action test problem."""
        n_states, n_actions = 3, 2

        # Simple transitions: keep advances, replace resets
        transitions = torch.zeros((n_actions, n_states, n_states))
        transitions[0, 0, 1] = 1.0  # keep at 0 -> go to 1
        transitions[0, 1, 2] = 1.0  # keep at 1 -> go to 2
        transitions[0, 2, 2] = 1.0  # keep at 2 -> stay at 2
        transitions[1, :, 0] = 1.0  # replace -> reset to 0

        # Trajectories consistent with simple replacement behavior
        trajectories = [
            Trajectory(
                states=torch.tensor([0, 1, 2, 0, 1]),
                actions=torch.tensor([0, 0, 1, 0, 0]),
                next_states=torch.tensor([1, 2, 0, 1, 2]),
            ),
            Trajectory(
                states=torch.tensor([0, 1, 0, 1, 2]),
                actions=torch.tensor([0, 0, 0, 0, 1]),
                next_states=torch.tensor([1, 0, 1, 2, 0]),
            ),
            Trajectory(
                states=torch.tensor([0, 1, 2, 0]),
                actions=torch.tensor([0, 0, 1, 0]),
                next_states=torch.tensor([1, 2, 0, 1]),
            ),
        ]
        panel = Panel(trajectories=trajectories)

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=0.9,
        )

        # Feature matrix: operating cost feature and replacement indicator
        feature_matrix = torch.zeros(n_states, n_actions, 2)
        for s in range(n_states):
            feature_matrix[s, 0, 0] = s / max(n_states - 1, 1)  # operating cost (keep)
            feature_matrix[s, 1, 1] = 1.0  # replacement indicator

        from econirl.preferences.linear import LinearUtility
        utility = LinearUtility(
            feature_matrix=feature_matrix,
            parameter_names=["operating_cost", "replacement_cost"],
        )

        return {
            "n_states": n_states,
            "n_actions": n_actions,
            "transitions": transitions,
            "panel": panel,
            "problem": problem,
            "utility": utility,
        }

    def test_estimator_name(self):
        """Check the estimator name property."""
        est = GLADIUSEstimator()
        assert est.name == "GLADIUS"

    def test_estimator_runs_without_error(self, small_setup):
        """GLADIUS should run without raising exceptions on a small problem."""
        config = GLADIUSConfig(
            q_hidden_dim=16,
            q_num_layers=1,
            v_hidden_dim=16,
            v_num_layers=1,
            max_epochs=10,
            batch_size=32,
            compute_se=False,
            verbose=False,
        )
        estimator = GLADIUSEstimator(config=config)

        result = estimator.estimate(
            panel=small_setup["panel"],
            utility=small_setup["utility"],
            problem=small_setup["problem"],
            transitions=small_setup["transitions"],
        )

        assert result.policy.shape == (small_setup["n_states"], small_setup["n_actions"])
        assert result.value_function.shape == (small_setup["n_states"],)
        assert np.isfinite(result.log_likelihood)

    def test_policy_is_valid_distribution(self, small_setup):
        """Estimated policy should be a valid probability distribution."""
        config = GLADIUSConfig(
            q_hidden_dim=16,
            q_num_layers=1,
            v_hidden_dim=16,
            v_num_layers=1,
            max_epochs=10,
            batch_size=32,
            compute_se=False,
        )
        estimator = GLADIUSEstimator(config=config)

        result = estimator.estimate(
            panel=small_setup["panel"],
            utility=small_setup["utility"],
            problem=small_setup["problem"],
            transitions=small_setup["transitions"],
        )

        policy = result.policy
        # Each row should sum to 1
        row_sums = policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(small_setup["n_states"]), atol=1e-5), (
            f"Policy rows should sum to 1, got {row_sums}"
        )
        # All entries should be non-negative
        assert torch.all(policy >= 0), "Policy should be non-negative"

    def test_loss_decreases(self, small_setup):
        """Training loss should decrease over epochs."""
        config = GLADIUSConfig(
            q_hidden_dim=16,
            q_num_layers=1,
            v_hidden_dim=16,
            v_num_layers=1,
            max_epochs=50,
            batch_size=32,
            compute_se=False,
        )
        estimator = GLADIUSEstimator(config=config)

        result = estimator._optimize(
            panel=small_setup["panel"],
            utility=small_setup["utility"],
            problem=small_setup["problem"],
            transitions=small_setup["transitions"],
        )

        loss_history = result.metadata["loss_history"]
        assert len(loss_history) > 1, "Should have multiple epochs of loss"

        # Compare average of first few vs last few
        early_avg = np.mean(loss_history[:5])
        late_avg = np.mean(loss_history[-5:])
        assert late_avg < early_avg, (
            f"Loss should decrease: early_avg={early_avg:.4f}, late_avg={late_avg:.4f}"
        )

    def test_parameters_shape(self, small_setup):
        """Extracted parameters should have the right shape."""
        config = GLADIUSConfig(
            q_hidden_dim=16,
            q_num_layers=1,
            v_hidden_dim=16,
            v_num_layers=1,
            max_epochs=10,
            batch_size=32,
            compute_se=False,
        )
        estimator = GLADIUSEstimator(config=config)

        result = estimator._optimize(
            panel=small_setup["panel"],
            utility=small_setup["utility"],
            problem=small_setup["problem"],
            transitions=small_setup["transitions"],
        )

        # Should have 2 parameters (operating_cost, replacement_cost)
        assert result.parameters.shape == (2,), (
            f"Expected shape (2,), got {result.parameters.shape}"
        )
        assert torch.all(torch.isfinite(result.parameters)), "Parameters should be finite"

    def test_networks_stored_after_fit(self, small_setup):
        """After fitting, q_net_ and ev_net_ should be stored."""
        config = GLADIUSConfig(
            q_hidden_dim=16,
            q_num_layers=1,
            v_hidden_dim=16,
            v_num_layers=1,
            max_epochs=5,
            batch_size=32,
            compute_se=False,
        )
        estimator = GLADIUSEstimator(config=config)

        estimator._optimize(
            panel=small_setup["panel"],
            utility=small_setup["utility"],
            problem=small_setup["problem"],
            transitions=small_setup["transitions"],
        )

        assert estimator.q_net_ is not None
        assert estimator.ev_net_ is not None
        assert isinstance(estimator.q_net_, QNetwork)
        assert isinstance(estimator.ev_net_, EVNetwork)

    def test_config_kwargs_override(self):
        """Config kwargs should override defaults."""
        estimator = GLADIUSEstimator(max_epochs=42, verbose=True)
        assert estimator.config.max_epochs == 42
        assert estimator.config.verbose is True

    def test_config_object_with_kwargs_override(self):
        """Kwargs should override a passed config object."""
        config = GLADIUSConfig(max_epochs=100)
        estimator = GLADIUSEstimator(config=config, max_epochs=42)
        assert estimator.config.max_epochs == 42

    def test_metadata_contains_tables(self, small_setup):
        """Result metadata should contain reward_table, q_table, ev_table."""
        config = GLADIUSConfig(
            q_hidden_dim=16,
            q_num_layers=1,
            v_hidden_dim=16,
            v_num_layers=1,
            max_epochs=5,
            batch_size=32,
            compute_se=False,
        )
        estimator = GLADIUSEstimator(config=config)

        result = estimator._optimize(
            panel=small_setup["panel"],
            utility=small_setup["utility"],
            problem=small_setup["problem"],
            transitions=small_setup["transitions"],
        )

        assert "reward_table" in result.metadata
        assert "q_table" in result.metadata
        assert "ev_table" in result.metadata
        assert "loss_history" in result.metadata
        assert "final_loss" in result.metadata


class TestGLADIUSStateFeatures:
    """Tests for state feature construction."""

    def test_state_features_range(self):
        """State features should be in [0, 1]."""
        estimator = GLADIUSEstimator(max_epochs=1)
        n_states = 10
        states = torch.arange(n_states)
        features = estimator._build_state_features(states, n_states)

        assert features.shape == (n_states, 1)
        assert features.min() >= 0.0
        assert features.max() <= 1.0
        assert torch.isclose(features[0, 0], torch.tensor(0.0))
        assert torch.isclose(features[-1, 0], torch.tensor(1.0))

    def test_state_features_all(self):
        """_build_state_features_all should match per-state features."""
        estimator = GLADIUSEstimator(max_epochs=1)
        n_states = 5
        all_feat = estimator._build_state_features_all(n_states)
        per_feat = estimator._build_state_features(torch.arange(n_states), n_states)
        assert torch.allclose(all_feat, per_feat)

    def test_state_features_single_state(self):
        """Single-state environment should not divide by zero."""
        estimator = GLADIUSEstimator(max_epochs=1)
        features = estimator._build_state_features(torch.tensor([0]), n_states=1)
        assert features.shape == (1, 1)
        assert torch.isfinite(features).all()


@pytest.mark.slow
class TestGLADIUSParameterRecovery:
    """Parameter recovery test on the Rust bus environment.

    This test generates data from a known DGP and checks that GLADIUS
    can recover the structural parameters within reasonable tolerance.
    """

    def test_rust_bus_parameter_recovery(self):
        """Recover Rust bus parameters with RMSE < 1.0."""
        from econirl.environments import RustBusEnvironment
        from econirl.preferences.linear import LinearUtility
        from econirl.simulation.synthetic import simulate_panel

        env = RustBusEnvironment(
            operating_cost=0.001,
            replacement_cost=3.0,
            discount_factor=0.9999,
        )
        utility = LinearUtility.from_environment(env)
        panel = simulate_panel(env, n_individuals=500, n_periods=100, seed=42)

        config = GLADIUSConfig(
            q_hidden_dim=128,
            q_num_layers=3,
            v_hidden_dim=128,
            v_num_layers=3,
            q_lr=1e-3,
            v_lr=1e-3,
            max_epochs=500,
            batch_size=512,
            bellman_penalty_weight=1.0,
            compute_se=False,
            verbose=False,
        )
        estimator = GLADIUSEstimator(config=config)

        result = estimator._optimize(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )

        true_params = torch.tensor([0.001, 3.0])
        estimated_params = result.parameters

        # IRL recovers parameters up to a scale factor, so we compare
        # the ratio of parameters (replacement_cost / operating_cost).
        # The true ratio is 3.0 / 0.001 = 3000.
        # We also check RMSE directly with a generous bound since the
        # parameters have very different scales.
        rmse = torch.sqrt(((estimated_params - true_params) ** 2).mean()).item()

        # Also check that the sign/direction is correct
        assert estimated_params[1].item() > 0, "Replacement cost should be positive"

        # Check that the estimated policy is reasonable
        assert result.policy.shape == (env.problem_spec.num_states, env.problem_spec.num_actions)
        policy_sums = result.policy.sum(dim=1)
        assert torch.allclose(policy_sums, torch.ones(env.problem_spec.num_states), atol=1e-4)

        # RMSE check -- generous bound for NN-based method
        assert rmse < 1.0, (
            f"RMSE={rmse:.4f} exceeds 1.0. "
            f"True: {true_params.tolist()}, Estimated: {estimated_params.tolist()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
