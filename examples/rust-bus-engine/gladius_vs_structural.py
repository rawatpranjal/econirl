#!/usr/bin/env python3
"""
GLADIUS vs Structural Estimators on Rust Bus Engine
====================================================

Compares GLADIUS (neural network IRL) against structural estimators
(NFXP, CCP) and MCE-IRL on the Rust (1987) bus engine replacement
problem with known ground truth.

This example demonstrates the corrected GLADIUS algorithm from
Kang et al. (2025) with alternating zeta/Q optimization, learning
rate decay, and optional Tikhonov annealing.

Usage:
    python examples/rust-bus-engine/gladius_vs_structural.py
"""

import time

import numpy as np
import jax.numpy as jnp

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration

# ── Configuration ──
# Uses parameters from Kang et al. (2025) Section 7 experiments.
# The lower discount factor (0.95 vs 0.9999) keeps Q-values on a
# manageable scale for neural network estimation. With beta=0.9999
# the Q-values are order 10000 and tiny reward differences get lost
# in numerical noise.
N_INDIVIDUALS = 500
N_PERIODS = 100
SEED = 42

TRUE_OC = 0.01
TRUE_RC = 3.0


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a = jnp.asarray(a, dtype=jnp.float64).flatten()
    b = jnp.asarray(b, dtype=jnp.float64).flatten()
    dot = jnp.dot(a, b)
    norm = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    if norm < 1e-15:
        return float("nan")
    return float(dot / norm)


def mape(estimated, true):
    """Mean absolute percentage error."""
    estimated = np.asarray(estimated, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mask = np.abs(true) > 1e-12
    return float(np.mean(np.abs((estimated[mask] - true[mask]) / true[mask])) * 100)


def main():
    # ── Setup ──
    env = RustBusEnvironment(
        operating_cost=TRUE_OC, replacement_cost=TRUE_RC,
        num_mileage_bins=90, discount_factor=0.95,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.array([TRUE_OC, TRUE_RC])

    panel = simulate_panel(
        env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED
    )
    n_obs = sum(len(t.states) for t in panel.trajectories)

    # Compute true policy
    operator = SoftBellmanOperator(
        problem, jnp.asarray(transitions, dtype=jnp.float64)
    )
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy = true_result.policy

    print(f"Environment: {problem.num_states} states, "
          f"{problem.num_actions} actions, gamma={problem.discount_factor}")
    print(f"True params: operating_cost={TRUE_OC}, replacement_cost={TRUE_RC}")
    print(f"Simulated: {N_INDIVIDUALS} x {N_PERIODS} = {n_obs:,} observations")
    print()

    results = []

    # ── NFXP ──
    try:
        from econirl.estimation.nfxp import NFXPEstimator
        print("Running NFXP...")
        t0 = time.time()
        nfxp = NFXPEstimator(se_method="robust", verbose=False)
        nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
        dt = time.time() - t0
        results.append(("NFXP", nfxp_result, dt))
    except Exception as e:
        print(f"  NFXP failed: {e}")

    # ── CCP ──
    try:
        from econirl.estimation.ccp import CCPEstimator
        print("Running CCP...")
        t0 = time.time()
        ccp = CCPEstimator(num_policy_iterations=20, verbose=False)
        ccp_result = ccp.estimate(panel, utility, problem, transitions)
        dt = time.time() - t0
        results.append(("CCP", ccp_result, dt))
    except Exception as e:
        print(f"  CCP failed: {e}")

    # ── MCE-IRL ──
    try:
        from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
        print("Running MCE-IRL...")
        t0 = time.time()
        mce = MCEIRLEstimator(
            config=MCEIRLConfig(
                learning_rate=0.1, outer_max_iter=200,
                inner_max_iter=200, outer_tol=1e-5,
            ),
            verbose=False,
        )
        mce_result = mce.estimate(panel, utility, problem, transitions)
        dt = time.time() - t0
        results.append(("MCE-IRL", mce_result, dt))
    except Exception as e:
        print(f"  MCE-IRL failed: {e}")

    # ── GLADIUS (new algorithm) ──
    try:
        from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
        print("Running GLADIUS (alternating, LR decay)...")
        t0 = time.time()
        gladius = GLADIUSEstimator(config=GLADIUSConfig(
            q_hidden_dim=64, v_hidden_dim=64,
            q_num_layers=2, v_num_layers=2,
            max_epochs=500, batch_size=256,
            alternating_updates=True,
            lr_decay_rate=0.001,
            bellman_penalty_weight=1.0,
            verbose=True,
        ))
        gladius_result = gladius.estimate(panel, utility, problem, transitions)
        dt = time.time() - t0
        results.append(("GLADIUS", gladius_result, dt))
    except Exception as e:
        print(f"  GLADIUS failed: {e}")

    # ── GLADIUS with Tikhonov ──
    try:
        from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
        print("Running GLADIUS (Tikhonov annealing)...")
        t0 = time.time()
        gladius_tik = GLADIUSEstimator(config=GLADIUSConfig(
            q_hidden_dim=64, v_hidden_dim=64,
            q_num_layers=2, v_num_layers=2,
            max_epochs=500, batch_size=256,
            alternating_updates=True,
            lr_decay_rate=0.001,
            tikhonov_annealing=True,
            tikhonov_initial_weight=100.0,
            verbose=True,
        ))
        gladius_tik_result = gladius_tik.estimate(
            panel, utility, problem, transitions
        )
        dt = time.time() - t0
        results.append(("GLADIUS-Tik", gladius_tik_result, dt))
    except Exception as e:
        print(f"  GLADIUS-Tik failed: {e}")

    # ── Results table ──
    print()
    print("=" * 80)
    print(f"{'Estimator':<18} {'OC':>10} {'RC':>10} {'MAPE%':>8} "
          f"{'Cosine':>8} {'MaxPDiff':>10} {'Time':>8}")
    print("-" * 80)

    for name, result, dt in results:
        params = np.asarray(result.parameters)
        cos = cosine_sim(params, true_params)
        m = mape(params, np.array([TRUE_OC, TRUE_RC]))
        max_pdiff = float(
            jnp.max(jnp.abs(jnp.asarray(result.policy) - true_policy))
        )
        print(f"{name:<18} {params[0]:>10.6f} {params[1]:>10.4f} "
              f"{m:>7.1f}% {cos:>8.4f} {max_pdiff:>10.6f} {dt:>7.1f}s")

    print("-" * 80)
    print(f"{'TRUE':<18} {TRUE_OC:>10.6f} {TRUE_RC:>10.4f}")
    print()


if __name__ == "__main__":
    main()
