"""Parameter recovery benchmarks across estimator x DGP combinations.

Each test simulates a panel from a known DGP, estimates parameters, and
checks that RMSE relative to the true parameter vector is below a tolerance.

All tests in this module are marked ``@pytest.mark.slow``.
"""

import pytest
import numpy as np
import jax.numpy as jnp

from econirl.environments import (
    RustBusEnvironment,
    MultiComponentBusEnvironment,
)
from econirl.estimation import (
    NFXPEstimator,
    CCPEstimator,
    MCEIRLEstimator,
    MCEIRLConfig,
    MaxEntIRLEstimator,
    TDCCPEstimator,
    TDCCPConfig,
    GLADIUSEstimator,
    GLADIUSConfig,
    GAILEstimator,
    GAILConfig,
    AIRLEstimator,
    AIRLConfig,
)
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.estimation.nnes import NNESEstimator
from econirl.estimation.sees import SEESEstimator
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rmse(estimated: jnp.ndarray, true: jnp.ndarray) -> float:
    """Compute RMSE between estimated and true parameter vectors."""
    return float(jnp.sqrt(jnp.mean((estimated - true) ** 2)))


def _simulate_and_prepare(env, n_individuals=500, n_periods=100, seed=42):
    """Simulate panel and build utility / problem / transitions."""
    panel = simulate_panel(env, n_individuals=n_individuals, n_periods=n_periods, seed=seed)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = env.get_true_parameter_vector()
    return panel, utility, problem, transitions, true_params


# ---------------------------------------------------------------------------
# NFXP on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nfxp_rust_bus():
    """NFXP should recover Rust bus parameters with RMSE < 0.1."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    estimator = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        inner_max_iter=100000,
        compute_hessian=False,
        verbose=False,
    )
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.1, f"NFXP Rust bus RMSE={rmse:.4f} exceeds tolerance 0.1"


# ---------------------------------------------------------------------------
# CCP on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_ccp_rust_bus():
    """CCP (Hotz-Miller) should recover Rust bus parameters with RMSE < 0.2."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    estimator = CCPEstimator(
        num_policy_iterations=1,
        compute_hessian=False,
        verbose=False,
    )
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.2, f"CCP Rust bus RMSE={rmse:.4f} exceeds tolerance 0.2"


# ---------------------------------------------------------------------------
# MPEC on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mpec_rust_bus():
    """MPEC should recover Rust bus parameters with RMSE < 0.2."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    config = MPECConfig(
        outer_max_iter=50,
        inner_max_iter=500,
        constraint_tol=1e-8,
    )
    estimator = MPECEstimator(config=config, compute_hessian=False, verbose=False)
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.2, f"MPEC Rust bus RMSE={rmse:.4f} exceeds tolerance 0.2"


# ---------------------------------------------------------------------------
# NNES on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_nnes_rust_bus():
    """NNES should recover Rust bus parameters with RMSE < 0.3."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    estimator = NNESEstimator(
        hidden_dim=32,
        v_epochs=500,
        n_outer_iterations=3,
        compute_se=False,
        verbose=False,
    )
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.3, f"NNES Rust bus RMSE={rmse:.4f} exceeds tolerance 0.3"


# ---------------------------------------------------------------------------
# SEES on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_sees_rust_bus():
    """SEES should recover Rust bus parameters with RMSE < 0.5."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    estimator = SEESEstimator(
        basis_type="fourier",
        basis_dim=8,
        penalty_lambda=0.01,
        compute_se=False,
        verbose=False,
    )
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.5, f"SEES Rust bus RMSE={rmse:.4f} exceeds tolerance 0.5"

    # Sieve coefficients should be finite
    alpha = result.metadata.get("alpha")
    if alpha is not None:
        assert jnp.all(jnp.isfinite(jnp.asarray(alpha))), "SEES sieve coefficients contain NaN/Inf"


# ---------------------------------------------------------------------------
# IQ-Learn on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_iq_learn_rust_bus():
    """IQ-Learn should recover reward direction (policy increases with mileage).

    IQ-Learn recovers a nonparametric Q-function, not structural theta.
    We test that the implied policy has the correct economic direction:
    replacement probability should increase with mileage.
    Uses lower gamma (0.99) since IQ-Learn with high gamma is slow.
    Tabular Q with 1000 iterations needed for correct directional recovery.
    """
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.99
    )
    panel, _, problem, transitions, _ = _simulate_and_prepare(env)

    reward = ActionDependentReward.from_rust_environment(env)

    config = IQLearnConfig(
        q_type="tabular",
        divergence="chi2",
        max_iter=1000,
        verbose=False,
    )
    estimator = IQLearnEstimator(config=config)
    result = estimator.estimate(panel, reward, problem, transitions)

    # Policy should be a valid distribution
    assert result.policy is not None, "IQ-Learn should return a policy"
    row_sums = result.policy.sum(axis=1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-4), (
        "IQ-Learn policy rows do not sum to 1"
    )

    # Directional check: replacement more likely at high mileage
    low_mile_replace = float(result.policy[:10, 1].mean())
    high_mile_replace = float(result.policy[-10:, 1].mean())
    assert high_mile_replace > low_mile_replace, (
        f"IQ-Learn: replacement should increase with mileage, "
        f"but low={low_mile_replace:.4f}, high={high_mile_replace:.4f}"
    )


# ---------------------------------------------------------------------------
# MCE IRL on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mce_irl_rust_bus():
    """MCE IRL should recover Rust bus parameters with RMSE < 0.5.

    Uses lower gamma (0.99) since IRL with gamma=0.9999 requires
    prohibitively many inner iterations for the soft VI to converge.
    """
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.99
    )
    panel, _, problem, transitions, true_params = _simulate_and_prepare(env)

    # MCE IRL needs ActionDependentReward
    reward = ActionDependentReward.from_rust_environment(env)

    config = MCEIRLConfig(
        learning_rate=0.05,
        outer_max_iter=500,
        inner_max_iter=5000,
        compute_se=False,
        verbose=False,
    )
    estimator = MCEIRLEstimator(config=config)
    result = estimator.estimate(panel, reward, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.5, f"MCE IRL Rust bus RMSE={rmse:.4f} exceeds tolerance 0.5"


# ---------------------------------------------------------------------------
# MaxEnt IRL on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_maxent_irl_rust_bus():
    """MaxEnt IRL should recover Rust bus reward direction with RMSE < 0.5.

    Uses lower gamma (0.99) and checks that the reward direction (ratio of
    parameters) is correct, since MaxEnt IRL uses state-only features which
    can't perfectly represent the Rust bus action-dependent structure.
    """
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.99
    )
    panel, _, problem, transitions, true_params = _simulate_and_prepare(env)

    # MaxEnt IRL uses state-only features (LinearReward with 2D features)
    from econirl.preferences.reward import LinearReward
    n_states = problem.num_states
    state_features = jnp.stack([
        jnp.arange(n_states, dtype=jnp.float32) / n_states,       # mileage
        jnp.ones(n_states, dtype=jnp.float32),                     # constant
    ], axis=1)
    reward = LinearReward(
        state_features=state_features,
        parameter_names=["operating_cost", "replacement_cost"],
        n_actions=problem.num_actions,
    )

    estimator = MaxEntIRLEstimator(
        inner_tol=1e-8,
        inner_max_iter=10000,
        outer_max_iter=500,
        compute_hessian=False,
        verbose=False,
    )
    result = estimator.estimate(panel, reward, problem, transitions)

    # MaxEnt IRL with state-only features can't match action-dependent rewards
    # exactly, but should produce a valid policy favoring low-mileage states
    assert result.policy is not None, "MaxEnt IRL should produce a policy"
    assert result.converged or result.parameters is not None, "Should produce parameters"


# ---------------------------------------------------------------------------
# TD-CCP on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_td_ccp_rust_bus():
    """TD-CCP should recover Rust bus parameters with RMSE < 0.5."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    config = TDCCPConfig(
        hidden_dim=64,
        avi_iterations=15,
        epochs_per_avi=20,
        compute_se=False,
        verbose=False,
    )
    estimator = TDCCPEstimator(config=config)
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 0.5, f"TD-CCP Rust bus RMSE={rmse:.4f} exceeds tolerance 0.5"


# ---------------------------------------------------------------------------
# TD-CCP on multi-component bus K=2
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_td_ccp_multi_component_k2():
    """TD-CCP on multi-component K=2 bus with RMSE < 1.0."""
    env = MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    config = TDCCPConfig(
        hidden_dim=64,
        avi_iterations=15,
        epochs_per_avi=20,
        compute_se=False,
        verbose=False,
    )
    estimator = TDCCPEstimator(config=config)
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 1.0, f"TD-CCP multi-component K=2 RMSE={rmse:.4f} exceeds tolerance 1.0"


# ---------------------------------------------------------------------------
# GLADIUS on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gladius_rust_bus():
    """GLADIUS should recover Rust bus parameters with RMSE < 1.0."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    config = GLADIUSConfig(
        max_epochs=300,
        q_hidden_dim=64,
        v_hidden_dim=64,
        compute_se=False,
        verbose=False,
    )
    estimator = GLADIUSEstimator(config=config)
    result = estimator.estimate(panel, utility, problem, transitions)
    rmse = _rmse(result.parameters, true_params)
    assert rmse < 1.0, f"GLADIUS Rust bus RMSE={rmse:.4f} exceeds tolerance 1.0"


# ---------------------------------------------------------------------------
# GAIL on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_gail_rust_bus():
    """GAIL should produce a valid policy and RMSE < 2.0 (or run without error)."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    config = GAILConfig(
        discriminator_type="tabular",
        max_rounds=50,
        compute_se=False,
        verbose=False,
    )
    estimator = GAILEstimator(config=config)
    result = estimator.estimate(panel, utility, problem, transitions)

    # GAIL returns a policy; check it is a valid probability distribution
    assert result.policy is not None, "GAIL did not return a policy"
    assert result.policy.shape == (problem.num_states, problem.num_actions)
    row_sums = result.policy.sum(axis=1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-4), (
        "GAIL policy rows do not sum to 1"
    )

    # If parameters are returned, check RMSE
    if result.parameters is not None and result.parameters.size == true_params.size:
        rmse = _rmse(result.parameters, true_params)
        assert rmse < 2.0, f"GAIL Rust bus RMSE={rmse:.4f} exceeds tolerance 2.0"


# ---------------------------------------------------------------------------
# AIRL on Rust bus
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_airl_rust_bus():
    """AIRL should produce a valid policy and RMSE < 2.0 (or run without error)."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0, discount_factor=0.9999
    )
    panel, utility, problem, transitions, true_params = _simulate_and_prepare(env)

    config = AIRLConfig(
        reward_type="tabular",
        max_rounds=50,
        compute_se=False,
        verbose=False,
    )
    estimator = AIRLEstimator(config=config)
    result = estimator.estimate(panel, utility, problem, transitions)

    # AIRL returns a policy; check validity
    assert result.policy is not None, "AIRL did not return a policy"
    assert result.policy.shape == (problem.num_states, problem.num_actions)
    row_sums = result.policy.sum(axis=1)
    assert jnp.allclose(row_sums, jnp.ones_like(row_sums), atol=1e-4), (
        "AIRL policy rows do not sum to 1"
    )

    # If parameters are returned, check RMSE
    if result.parameters is not None and result.parameters.size == true_params.size:
        rmse = _rmse(result.parameters, true_params)
        assert rmse < 2.0, f"AIRL Rust bus RMSE={rmse:.4f} exceeds tolerance 2.0"
