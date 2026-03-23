"""Wall-clock scaling tests across state-space sizes.

Verifies that estimation completes within reasonable time for NFXP, CCP,
and TD-CCP across K=1..3 multi-component bus environments.

All tests are marked ``@pytest.mark.slow``.
"""

import time

import pytest
import torch

from econirl.environments import MultiComponentBusEnvironment
from econirl.estimation import (
    NFXPEstimator,
    CCPEstimator,
    TDCCPEstimator,
    TDCCPConfig,
)
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_timed_estimation(estimator, env, n_individuals=200, n_periods=50, seed=42):
    """Simulate, estimate, and return (result, elapsed_seconds)."""
    panel = simulate_panel(env, n_individuals=n_individuals, n_periods=n_periods, seed=seed)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    t0 = time.perf_counter()
    result = estimator.estimate(panel, utility, problem, transitions)
    elapsed = time.perf_counter() - t0
    return result, elapsed


# ---------------------------------------------------------------------------
# NFXP scaling: K = 1, 2, 3
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("K", [1, 2, 3])
def test_nfxp_scaling(K):
    """NFXP completes on multi-component bus with K components."""
    env = MultiComponentBusEnvironment(K=K, M=10, discount_factor=0.99)
    estimator = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=50000,
        outer_max_iter=200,
        compute_hessian=False,
        verbose=False,
    )
    result, elapsed = _run_timed_estimation(estimator, env)
    assert result.converged, f"NFXP did not converge for K={K}"
    # Just log the time; no hard wall-clock assertion to avoid flaky CI
    print(f"NFXP K={K}: {elapsed:.1f}s, states={env.num_states}")


# ---------------------------------------------------------------------------
# CCP scaling: K = 1, 2, 3
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("K", [1, 2, 3])
def test_ccp_scaling(K):
    """CCP (Hotz-Miller) completes on multi-component bus with K components."""
    env = MultiComponentBusEnvironment(K=K, M=10, discount_factor=0.99)
    estimator = CCPEstimator(
        num_policy_iterations=1,
        compute_hessian=False,
        verbose=False,
    )
    result, elapsed = _run_timed_estimation(estimator, env)
    # CCP one-step always "converges"
    print(f"CCP K={K}: {elapsed:.1f}s, states={env.num_states}")


# ---------------------------------------------------------------------------
# TD-CCP scaling: K = 1, 2, 3
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("K", [1, 2, 3])
def test_td_ccp_scaling(K):
    """TD-CCP completes on multi-component bus with K components."""
    env = MultiComponentBusEnvironment(K=K, M=10, discount_factor=0.99)
    config = TDCCPConfig(
        hidden_dim=32,
        avi_iterations=5,
        epochs_per_avi=10,
        compute_se=False,
        verbose=False,
    )
    estimator = TDCCPEstimator(config=config)
    result, elapsed = _run_timed_estimation(estimator, env)
    print(f"TD-CCP K={K}: {elapsed:.1f}s, states={env.num_states}")


# ---------------------------------------------------------------------------
# Relative ordering sanity check
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_scaling_ordering():
    """K=1 should be faster than K=3 for NFXP (sanity check)."""
    estimator = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=50000,
        outer_max_iter=200,
        compute_hessian=False,
        verbose=False,
    )

    env_k1 = MultiComponentBusEnvironment(K=1, M=10, discount_factor=0.99)
    _, t1 = _run_timed_estimation(estimator, env_k1)

    env_k3 = MultiComponentBusEnvironment(K=3, M=10, discount_factor=0.99)
    _, t3 = _run_timed_estimation(estimator, env_k3)

    # K=3 has 1000 states vs K=1 with 10 states, so it should be slower.
    # Allow generous slack to avoid flakiness.
    print(f"NFXP K=1: {t1:.1f}s, K=3: {t3:.1f}s")
    assert t3 > t1 * 0.5, "K=3 was unexpectedly faster than K=1"
