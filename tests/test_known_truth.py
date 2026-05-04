"""Tests for the compact known-truth synthetic validation harness."""

from __future__ import annotations

import math

import jax.numpy as jnp
import pytest

from experiments.known_truth import (
    DEFAULT_CELLS,
    ESTIMATOR_CONTRACTS,
    REQUIRED_ESTIMATORS,
    CounterfactualConfig,
    KnownTruthDGPConfig,
    RecoveryGateFailure,
    SimulationConfig,
    build_counterfactual,
    build_known_truth_dgp,
    check_estimator_compatibility,
    get_cell,
    known_truth_initial_params,
    make_estimator,
    policy_divergence,
    recovery_gates,
    run_estimator,
    run_pre_estimation_diagnostics,
    simulate_known_truth_panel,
    solve_counterfactual_oracle,
    solve_known_truth,
)


def test_low_dim_action_dependent_dgp_solves_and_simulates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            num_regular_states=8,
            transition_noise=0.02,
            seed=10,
        )
    )

    assert dgp.transitions.shape == (3, 9, 9)
    assert dgp.feature_matrix.shape[:2] == (9, 3)
    assert jnp.allclose(dgp.transitions.sum(axis=2), 1.0)
    assert jnp.allclose(dgp.homogeneous_reward[:, dgp.config.exit_action], 0.0)
    assert jnp.allclose(dgp.homogeneous_reward[dgp.config.absorbing_state, :], 0.0)

    solution = solve_known_truth(dgp)
    assert solution.converged
    assert solution.policy.shape == (9, 3)
    assert jnp.allclose(solution.policy.sum(axis=1), 1.0, atol=1e-5)
    assert jnp.isclose(solution.state_occupancy.sum(), 1.0, atol=1e-5)

    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=25, n_periods=12, seed=11),
    )
    assert panel.num_individuals == 25
    assert panel.num_observations == 300

    diagnostics = run_pre_estimation_diagnostics(dgp, panel)
    assert diagnostics.passed
    assert diagnostics.feature_rank == diagnostics.num_features
    assert diagnostics.is_action_dependent


def test_state_only_and_high_dim_modes_are_distinct():
    state_only = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="state_only",
            reward_dim="low",
            num_regular_states=7,
            seed=12,
        )
    )
    state_only_diag = run_pre_estimation_diagnostics(state_only)
    assert state_only_diag.passed
    assert not state_only_diag.is_action_dependent

    high_dim = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            num_regular_states=9,
            high_state_dim=6,
            high_reward_features=10,
            seed=13,
        )
    )
    assert high_dim.state_features.shape == (10, 6)
    assert high_dim.feature_matrix.shape == (10, 3, 10)
    high_dim_diag = run_pre_estimation_diagnostics(high_dim)
    assert high_dim_diag.passed
    assert high_dim_diag.is_action_dependent


def test_latent_segment_dgp_tracks_segment_truth():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="latent_segments",
            num_regular_states=8,
            high_state_dim=5,
            high_reward_features=9,
            num_segments=2,
            seed=14,
        )
    )
    assert dgp.num_segments == 2
    assert dgp.true_parameters.shape == (2, 9)
    assert dgp.reward_matrix.shape == (2, 9, 3)

    sol0 = solve_known_truth(dgp, segment_index=0)
    sol1 = solve_known_truth(dgp, segment_index=1)
    assert sol0.converged
    assert sol1.converged

    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=15),
    )
    labels = panel.metadata["segment_labels"]
    assert len(labels) == 20
    assert set(labels).issubset({0, 1})


def test_type_a_b_c_counterfactual_oracles_change_truth():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=8, seed=16, transition_noise=0.0)
    )
    baseline = solve_known_truth(dgp)

    for kind in ("type_a", "type_b", "type_c"):
        oracle = solve_counterfactual_oracle(
            dgp,
            kind,
            config=CounterfactualConfig(type_b_skip=3, type_c_action=1),
        )
        assert oracle.baseline_solution.converged
        assert oracle.counterfactual_solution.converged
        divergence = policy_divergence(
            baseline.policy,
            oracle.counterfactual_solution.policy,
        )
        assert divergence.l1 >= 0.0

    cf = build_counterfactual(
        dgp,
        "type_c",
        CounterfactualConfig(type_c_action=1, type_c_penalty=-1_000.0),
    )
    type_c_oracle = solve_counterfactual_oracle(
        dgp,
        "type_c",
        config=CounterfactualConfig(type_c_action=1, type_c_penalty=-1_000.0),
    )
    assert cf.disabled_action == 1
    regular_policy = type_c_oracle.counterfactual_solution.policy[
        : dgp.config.num_regular_states, 1
    ]
    assert float(regular_policy.max()) < 1e-3


def test_estimator_contract_registry_has_required_twelve():
    assert len(REQUIRED_ESTIMATORS) == 12
    assert "BC" not in REQUIRED_ESTIMATORS
    assert "AIRL-Het" in REQUIRED_ESTIMATORS
    for name in REQUIRED_ESTIMATORS:
        contract = ESTIMATOR_CONTRACTS[name]
        assert contract.code_path
        assert contract.paper_paths
        assert contract.recovers


def test_estimator_factories_and_compatibility_reports_are_available():
    structural_dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=6, reward_mode="action_dependent")
    )
    state_only_dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=6, reward_mode="state_only")
    )
    hetero_dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="high_dim",
            reward_mode="action_dependent",
            reward_dim="high",
            heterogeneity="latent_segments",
            num_regular_states=6,
            high_state_dim=4,
            high_reward_features=8,
        )
    )

    for name in REQUIRED_ESTIMATORS:
        dgp = hetero_dgp if name == "AIRL-Het" else structural_dgp
        estimator = make_estimator(name, dgp, smoke=True)
        assert estimator is not None

    assert check_estimator_compatibility("NFXP", structural_dgp).compatible
    assert not check_estimator_compatibility("NFXP", state_only_dgp).compatible
    assert check_estimator_compatibility("AIRL-Het", hetero_dgp).compatible


def test_nfxp_uses_universal_canonical_preset():
    cell_ids = {cell.cell_id for cell in DEFAULT_CELLS}
    assert {
        "canonical_low_action",
        "canonical_low_state_only",
        "canonical_high_action",
        "canonical_latent_segments",
    }.issubset(cell_ids)

    dgp = build_known_truth_dgp(get_cell("canonical_low_action").dgp_config)
    assert check_estimator_compatibility("NFXP", dgp).compatible
    solution = solve_known_truth(dgp)
    action_mass = (solution.state_occupancy[:, None] * solution.policy).sum(axis=0)
    assert float(action_mass.min()) > 0.10

    state_only = build_known_truth_dgp(get_cell("canonical_low_state_only").dgp_config)
    high_action = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    latent = build_known_truth_dgp(get_cell("canonical_latent_segments").dgp_config)

    assert not check_estimator_compatibility("NFXP", state_only).compatible
    assert not check_estimator_compatibility("NFXP", high_action).compatible
    assert not check_estimator_compatibility("NFXP", latent).compatible


def test_legacy_cell_ids_are_aliases_not_separate_dgps():
    legacy = get_cell("low_state_action_reward")
    canonical = get_cell("canonical_low_action")

    assert legacy.cell_id == "low_state_action_reward"
    assert legacy.dgp_config == canonical.dgp_config


def test_known_truth_initialization_is_deterministic_and_near_truth():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(num_regular_states=6, transition_noise=0.02, seed=111)
    )

    init_a = known_truth_initial_params(dgp)
    init_b = known_truth_initial_params(dgp)
    truth = dgp.homogeneous_parameters

    assert jnp.allclose(init_a, init_b)
    assert init_a.shape == truth.shape
    assert not jnp.allclose(init_a, truth)
    assert float(jnp.linalg.norm(init_a - truth)) < 0.25 * float(jnp.linalg.norm(truth))


def test_nfxp_smoke_fit_produces_known_truth_metrics():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=112,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=113),
    )

    result = run_estimator("NFXP", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["optimizer"] == "BHHH"
    assert result.summary.metadata["inner_solver"] == "hybrid"
    assert result.summary.metadata["num_inner_iterations"] > 0

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0
    assert set(metrics["counterfactuals"]) == {"type_a", "type_b", "type_c"}
    for cf_metrics in metrics["counterfactuals"].values():
        assert cf_metrics.policy.tv >= 0.0
        assert math.isfinite(cf_metrics.value_rmse)
        assert math.isfinite(cf_metrics.regret)


def test_ccp_smoke_fit_produces_known_truth_metrics_and_gates():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=312,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=313),
    )

    result = run_estimator("CCP", dgp, panel, smoke=True)

    assert result.compatibility.compatible
    assert result.summary.policy.shape == (dgp.problem.num_states, dgp.problem.num_actions)
    assert result.summary.value_function.shape == (dgp.problem.num_states,)
    assert result.summary.metadata["num_policy_iterations"] == 3
    assert result.summary.metadata["npl_converged"] in {True, False}

    metrics = result.metrics
    assert metrics["parameters"] is not None
    assert math.isfinite(metrics["parameters"].rmse)
    assert math.isfinite(metrics["reward_rmse"])
    assert math.isfinite(metrics["value_rmse"])
    assert math.isfinite(metrics["q_rmse"])
    assert metrics["policy"].tv >= 0.0

    gate_names = {
        gate.name
        for gate in recovery_gates("CCP", result.summary, metrics, smoke=False)
    }
    assert {
        "npl_iterations",
        "standard_errors_finite",
        "parameter_cosine",
        "parameter_relative_rmse",
        "policy_tv",
        "value_rmse",
        "q_rmse",
        "type_a_regret",
        "type_b_regret",
        "type_c_regret",
    }.issubset(gate_names)


def test_nfxp_failed_non_smoke_recovery_raises_hard_gate():
    dgp = build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            heterogeneity="none",
            num_regular_states=5,
            transition_noise=0.02,
            seed=212,
        )
    )
    panel = simulate_known_truth_panel(
        dgp,
        SimulationConfig(n_individuals=20, n_periods=8, seed=213),
    )

    with pytest.raises(RecoveryGateFailure):
        run_estimator("NFXP", dgp, panel, smoke=False)
