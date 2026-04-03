"""Tests for the MPEC (Su-Judd 2012) estimator.

Verifies that MPEC recovers the same parameters as NFXP, satisfies the
Bellman constraint at convergence, and produces valid standard errors.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation import simulate_panel
from econirl.core.bellman import SoftBellmanOperator


@pytest.fixture
def small_bus_setup():
    """Small bus environment with simulated data for quick tests."""
    env = RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        seed=42,
    )
    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=123)
    utility = LinearUtility.from_environment(env)
    return env, panel, utility


class TestMPECBasic:
    """Basic MPEC estimator functionality."""

    def test_mpec_converges(self, small_bus_setup):
        env, panel, utility = small_bus_setup
        estimator = MPECEstimator(verbose=False)
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )
        assert result.converged

    def test_mpec_returns_valid_policy(self, small_bus_setup):
        env, panel, utility = small_bus_setup
        estimator = MPECEstimator(verbose=False)
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )
        # Policy should sum to 1 across actions
        row_sums = result.policy.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

        # Policy should be non-negative
        assert float(result.policy.min()) >= 0.0

    def test_mpec_has_standard_errors(self, small_bus_setup):
        env, panel, utility = small_bus_setup
        estimator = MPECEstimator(verbose=False)
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )
        assert result.standard_errors is not None
        assert not np.any(np.isnan(np.asarray(result.standard_errors)))

    def test_bellman_constraint_satisfied(self, small_bus_setup):
        """V should satisfy V = T(V; theta) at convergence."""
        env, panel, utility = small_bus_setup
        estimator = MPECEstimator(verbose=False)
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )

        # Reconstruct the Bellman operator and check constraint
        transitions = jnp.array(env.transition_matrices, dtype=jnp.float64)
        operator = SoftBellmanOperator(env.problem_spec, transitions)
        flow_utility = jnp.array(
            utility.compute(result.parameters), dtype=jnp.float64,
        )
        V = jnp.array(result.value_function, dtype=jnp.float64)
        bellman_result = operator.apply(flow_utility, V)

        violation = float(jnp.abs(V - bellman_result.V).max())
        assert violation < 1e-4, (
            f"Bellman constraint violation = {violation:.2e}, should be < 1e-4"
        )


class TestMPECvsNFXP:
    """MPEC should recover approximately the same parameters as NFXP."""

    def test_parameter_recovery_matches_nfxp(self, small_bus_setup):
        env, panel, utility = small_bus_setup

        # NFXP estimation
        nfxp = NFXPEstimator(
            optimizer="BHHH", inner_solver="hybrid", verbose=False,
        )
        nfxp_result = nfxp.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )

        # MPEC estimation
        mpec = MPECEstimator(verbose=False)
        mpec_result = mpec.estimate(
            panel=panel,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
        )

        # Parameters should be close
        nfxp_params = np.asarray(nfxp_result.parameters)
        mpec_params = np.asarray(mpec_result.parameters)

        np.testing.assert_allclose(
            mpec_params, nfxp_params, atol=0.3, rtol=0.3,
            err_msg=(
                f"MPEC params {mpec_params} differ from "
                f"NFXP params {nfxp_params} beyond tolerance"
            ),
        )

    def test_log_likelihood_close_to_nfxp(self, small_bus_setup):
        env, panel, utility = small_bus_setup

        nfxp = NFXPEstimator(
            optimizer="BHHH", inner_solver="hybrid", verbose=False,
        )
        nfxp_result = nfxp.estimate(
            panel=panel, utility=utility,
            problem=env.problem_spec, transitions=env.transition_matrices,
        )

        mpec = MPECEstimator(verbose=False)
        mpec_result = mpec.estimate(
            panel=panel, utility=utility,
            problem=env.problem_spec, transitions=env.transition_matrices,
        )

        # Log-likelihoods should be close (both are MLE)
        np.testing.assert_allclose(
            mpec_result.log_likelihood, nfxp_result.log_likelihood,
            atol=5.0,
            err_msg=(
                f"MPEC LL={mpec_result.log_likelihood:.2f} differs from "
                f"NFXP LL={nfxp_result.log_likelihood:.2f}"
            ),
        )


class TestMPECConfig:
    """Test MPEC configuration options."""

    def test_custom_config(self, small_bus_setup):
        env, panel, utility = small_bus_setup
        config = MPECConfig(
            rho_initial=10.0,
            rho_growth=5.0,
            outer_max_iter=20,
            constraint_tol=1e-6,
        )
        estimator = MPECEstimator(config=config, verbose=False)
        result = estimator.estimate(
            panel=panel, utility=utility,
            problem=env.problem_spec, transitions=env.transition_matrices,
        )
        assert result.parameters is not None

    def test_estimator_name(self):
        estimator = MPECEstimator()
        assert "MPEC" in estimator.name
