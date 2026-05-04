"""Component tests for SEES against exact known-truth DGP objects."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from econirl.estimation.sees import SEESEstimator
from experiments.known_truth import (
    KnownTruthDGP,
    KnownTruthDGPConfig,
    build_known_truth_dgp,
    get_cell,
    solve_known_truth,
)


def _low_dim_dgp() -> KnownTruthDGP:
    return build_known_truth_dgp(
        KnownTruthDGPConfig(
            state_mode="low_dim",
            reward_mode="action_dependent",
            reward_dim="low",
            num_regular_states=8,
            transition_noise=0.02,
            seed=801,
        )
    )


def _project_value(basis: jnp.ndarray, value: jnp.ndarray) -> jnp.ndarray:
    alpha, *_ = np.linalg.lstsq(
        np.asarray(basis, dtype=np.float64),
        np.asarray(value, dtype=np.float64),
        rcond=None,
    )
    return jnp.asarray(alpha, dtype=jnp.float64)


def _choice_values_from_basis(
    dgp: KnownTruthDGP,
    basis: jnp.ndarray,
    alpha: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    expected_basis = jnp.einsum(
        "ast,tk->ask",
        dgp.transitions.astype(jnp.float64),
        basis.astype(jnp.float64),
    )
    flow_u = jnp.einsum(
        "sak,k->sa",
        dgp.feature_matrix.astype(jnp.float64),
        theta.astype(jnp.float64),
    )
    continuation = dgp.problem.discount_factor * jnp.einsum(
        "ask,k->sa",
        expected_basis,
        alpha.astype(jnp.float64),
    )
    return flow_u + continuation


def test_sees_low_dimensional_basis_represents_exact_value():
    dgp = _low_dim_dgp()
    solution = solve_known_truth(dgp)
    estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(dgp.problem.num_states, dgp.problem)
    alpha = _project_value(basis, solution.V)
    projected_value = basis @ alpha

    assert estimator._last_basis_metadata["basis_source"] == "state_index"
    assert float(jnp.sqrt(jnp.mean((projected_value - solution.V) ** 2))) < 1e-8


def test_sees_high_dimensional_basis_represents_exact_value():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    solution = solve_known_truth(dgp)
    estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(dgp.problem.num_states, dgp.problem)
    alpha = _project_value(basis, solution.V)
    projected_value = basis @ alpha

    assert estimator._last_basis_metadata["basis_source"] == "encoded_state"
    assert estimator._last_basis_metadata["state_feature_dim"] == 16
    assert float(jnp.sqrt(jnp.mean((projected_value - solution.V) ** 2))) < 1e-8


def test_sees_basis_choice_values_reproduce_known_truth_q_policy_and_bellman():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)
    solution = solve_known_truth(dgp)
    estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )

    basis = estimator._build_basis(dgp.problem.num_states, dgp.problem)
    alpha = _project_value(basis, solution.V)
    q_vals = _choice_values_from_basis(
        dgp,
        basis,
        alpha,
        dgp.homogeneous_parameters,
    )
    policy = jax.nn.softmax(q_vals / dgp.problem.scale_parameter, axis=1)
    bellman_value = dgp.problem.scale_parameter * jax.scipy.special.logsumexp(
        q_vals / dgp.problem.scale_parameter,
        axis=1,
    )
    projected_value = basis @ alpha

    assert float(jnp.max(jnp.abs(q_vals - solution.Q))) < 1e-6
    assert float(jnp.max(jnp.abs(policy - solution.policy))) < 1e-6
    assert float(jnp.max(jnp.abs(projected_value - bellman_value))) < 1e-6


def test_sees_auto_basis_uses_state_encoder_for_canonical_high_action():
    dgp = build_known_truth_dgp(get_cell("canonical_high_action").dgp_config)

    encoded_estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        compute_se=False,
    )
    encoded_basis = encoded_estimator._build_basis(dgp.problem.num_states, dgp.problem)

    index_estimator = SEESEstimator(
        basis_type="bspline",
        basis_dim=dgp.problem.num_states,
        state_basis_mode="index",
        compute_se=False,
    )
    index_basis = index_estimator._build_basis(dgp.problem.num_states, dgp.problem)

    encoded_states = dgp.problem.state_encoder(jnp.arange(dgp.problem.num_states))
    assert encoded_states.shape == (dgp.problem.num_states, 16)
    assert encoded_estimator._last_basis_metadata["basis_source"] == "encoded_state"
    assert index_estimator._last_basis_metadata["basis_source"] == "state_index"
    assert not jnp.allclose(encoded_basis, index_basis)
