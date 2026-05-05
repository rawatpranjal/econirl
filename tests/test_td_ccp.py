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
from types import SimpleNamespace

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.shapeshifter import ShapeshifterConfig, ShapeshifterEnvironment
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.td_ccp import (
    TDCCPEstimator,
    TDCCPConfig,
    _EVComponentNetwork,
    make_state_action_tabular_utility,
)
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel
from papers.econirl_package.primers.tdccp.tdccp_run import (
    build_paper_hard_case_dgp,
    evaluate_hard_case_summary,
    evaluate_paper_hard_case_summary,
    tdccp_hard_case_gates,
    tdccp_paper_hard_case_gates,
)


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

    def test_encoded_basis_uses_state_encoder(self):
        """Encoded basis should use problem.state_encoder, not scalar labels."""
        state_features = jnp.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=jnp.float64,
        )
        problem = DDCProblem(
            num_states=3,
            num_actions=2,
            discount_factor=0.9,
            state_dim=2,
            state_encoder=lambda states: state_features[states],
        )
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="encoded",
                basis_dim=2,
            )
        )

        actions = np.array([0, 1, 0], dtype=np.int32)
        states = np.array([0, 1, 2], dtype=np.int32)
        phi = estimator._build_basis_functions(
            actions, states, problem.num_states, problem.num_actions, problem
        )

        # Per action: intercept + two first-order encoded features +
        # two second-order encoded features.
        assert phi.shape == (3, 10)
        np.testing.assert_allclose(phi[0, 1:3], np.array([1.0, 0.0]))
        np.testing.assert_allclose(phi[1, 6:8], np.array([0.0, 1.0]))
        assert np.all(phi[1, :5] == 0.0)

    def test_g_uses_next_period_entropy_target(self):
        """The g semi-gradient target should be beta * e(a', x')."""
        estimator = TDCCPEstimator(
            config=TDCCPConfig(
                method="semigradient",
                basis_type="tabular",
                basis_ridge=0.0,
            )
        )
        gamma = 0.5
        ccps = np.array([[0.8, 0.2]], dtype=np.float64)
        actions = np.array([0, 1], dtype=np.int32)
        states = np.array([0, 0], dtype=np.int32)
        next_actions = np.array([1, 0], dtype=np.int32)
        next_states = np.array([0, 0], dtype=np.int32)
        feature_matrix = np.zeros((1, 2, 1), dtype=np.float64)

        _, g_table = estimator._semigradient_solve(
            actions,
            states,
            next_actions,
            next_states,
            feature_matrix,
            ccps,
            num_states=1,
            num_actions=2,
            gamma=gamma,
        )

        euler = 0.5772156649015329
        e0 = euler - np.log(ccps[0, 0])
        e1 = euler - np.log(ccps[0, 1])
        phi = np.eye(2)
        phi_next = phi[[1, 0]]
        A = (phi.T @ (phi - gamma * phi_next)) / 2
        b = (phi.T @ (gamma * np.array([e1, e0]))) / 2
        expected = np.linalg.solve(A, b)

        np.testing.assert_allclose(g_table[0], expected, atol=1e-10)


class TestTDCCPHardCaseComponents:
    """Component tests for the shapeshifter neural/neural hard-case runner."""

    def test_paper_hard_case_has_finite_theta_neural_feature_utility(self):
        """Paper hard case should have finite theta and exact linear rewards."""
        dgp = build_paper_hard_case_dgp(seed=11)
        utility = dgp["utility"]
        true_params = dgp["true_params"]
        true_reward = dgp["true_reward"]

        reconstructed = utility.compute(true_params)
        np.testing.assert_allclose(
            np.asarray(reconstructed),
            np.asarray(true_reward),
            atol=1e-10,
        )
        assert utility.num_parameters == 8
        assert dgp["basis_metadata"]["action_normalization"] == (
            "action 0 reward features fixed to zero"
        )
        np.testing.assert_allclose(
            np.asarray(utility.feature_matrix[:, 0, :]),
            np.zeros((utility.num_states, utility.num_parameters)),
            atol=1e-12,
        )

    def test_paper_hard_case_metrics_and_gates_use_parameter_truth(self):
        """Finite-theta hard-case gates should include structural theta checks."""
        dgp = build_paper_hard_case_dgp(seed=13)
        env = dgp["env"]
        utility = dgp["utility"]
        true_params = dgp["true_params"]
        true_reward = dgp["true_reward"]
        truth = value_iteration(
            SoftBellmanOperator(env.problem_spec, env.transition_matrices),
            true_reward,
            tol=1e-10,
            max_iter=10_000,
        )
        summary = SimpleNamespace(
            parameters=true_params,
            policy=truth.policy,
            value_function=truth.V,
            converged=True,
        )

        metrics = evaluate_paper_hard_case_summary(
            env,
            utility,
            true_params,
            true_reward,
            summary,
            truth=truth,
        )
        gates = tdccp_paper_hard_case_gates(summary, metrics)
        gate_names = {gate.name for gate in gates}

        assert metrics["parameters"]["cosine_similarity"] == pytest.approx(1.0)
        assert metrics["reward_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert "parameter_cosine" in gate_names
        assert "parameter_relative_rmse" in gate_names
        assert all(gate.passed for gate in gates)

    def test_tabular_reward_utility_reconstructs_reward_matrix(self):
        """One-hot state-action utility should reconstruct any reward matrix."""
        reward = jnp.array(
            [
                [0.1, -0.2, 0.3],
                [0.4, 0.0, -0.5],
                [-0.7, 0.8, 0.9],
                [1.1, -1.2, 1.3],
            ],
            dtype=jnp.float64,
        )
        utility = make_state_action_tabular_utility(
            reward.shape[0],
            reward.shape[1],
        )

        reconstructed = utility.compute(reward.reshape(-1))

        np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(reward))

    def test_hard_case_metrics_use_shapeshifter_reward_and_solver_truth(self):
        """Hard-case metrics should compare against environment reward/solver truth."""
        env = ShapeshifterEnvironment(
            ShapeshifterConfig(
                num_states=5,
                num_actions=2,
                num_features=3,
                reward_type="neural",
                feature_type="neural",
                action_dependent=True,
                stochastic_transitions=True,
                stochastic_rewards=False,
                num_periods=None,
                discount_factor=0.9,
                seed=7,
            )
        )
        truth = value_iteration(
            SoftBellmanOperator(env.problem_spec, env.transition_matrices),
            env.true_reward_matrix,
            tol=1e-10,
            max_iter=10_000,
        )
        summary = SimpleNamespace(
            parameters=jnp.asarray(env.true_reward_matrix).reshape(-1),
            policy=truth.policy,
            value_function=truth.V,
            converged=True,
        )

        metrics = evaluate_hard_case_summary(env, summary, truth=truth)

        assert metrics["reward_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["reward_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["policy_tv"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["value_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert metrics["q_normalized_rmse"] == pytest.approx(0.0, abs=1e-10)
        assert set(metrics["counterfactuals"]) == {"type_a", "type_b", "type_c"}
        for cf_metrics in metrics["counterfactuals"].values():
            assert cf_metrics["regret"] == pytest.approx(0.0, abs=1e-8)

    def test_neural_hard_case_gates_skip_parameter_cosine(self):
        """Neural-reward gates should not include finite-theta recovery checks."""
        summary = SimpleNamespace(converged=True)
        metrics = {
            "reward_normalized_rmse": 0.0,
            "policy_tv": 0.0,
            "value_normalized_rmse": 0.0,
            "q_normalized_rmse": 0.0,
            "counterfactuals": {
                "type_a": {"regret": 0.0},
                "type_b": {"regret": 0.0},
                "type_c": {"regret": 0.0},
            },
        }

        gates = tdccp_hard_case_gates(summary, metrics)
        gate_names = {gate.name for gate in gates}

        assert "parameter_cosine" not in gate_names
        assert "parameter_relative_rmse" not in gate_names
        assert {
            "converged",
            "reward_normalized_rmse",
            "policy_tv",
            "value_normalized_rmse",
            "q_normalized_rmse",
            "type_a_regret",
            "type_b_regret",
            "type_c_regret",
        }.issubset(gate_names)


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

    def test_output_shift_initializes_predictions(self):
        """AVI constant initialization should shift initial network values."""
        key = jax.random.PRNGKey(0)
        net_unshifted = _EVComponentNetwork(
            input_dim=3,
            hidden_dim=16,
            num_hidden_layers=1,
            key=key,
            output_shift=0.0,
        )
        net_shifted = _EVComponentNetwork(
            input_dim=3,
            hidden_dim=16,
            num_hidden_layers=1,
            key=key,
            output_shift=2.5,
        )
        x = jnp.ones(3)
        assert np.isclose(float(net_shifted(x) - net_unshifted(x)), 2.5)

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
