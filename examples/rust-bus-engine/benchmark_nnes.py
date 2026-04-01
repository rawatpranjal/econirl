#!/usr/bin/env python3
"""
NNES-NPL vs NNES-NFXP Benchmark on Rust Bus Engine
====================================================

Compares the two NNES variants (NPL Bellman vs NFXP Bellman) against
NFXP and CCP-NPL baselines on simulated Rust (1987) bus data with
known true parameters.

The NPL variant has the zero Jacobian property (Neyman orthogonality),
so V-approximation errors should not bias the structural parameter
estimates. The NFXP variant lacks this property and may produce biased
estimates when the V-network is not fully converged.

Usage:
    python examples/rust-bus-engine/benchmark_nnes.py
"""

import time

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ── Configuration ──
N_INDIVIDUALS = 200
N_PERIODS = 100
SEED = 42
TRUE_OC = 0.001
TRUE_RC = 3.0


def cosine_sim(a, b):
    a = jnp.asarray(a, dtype=jnp.float64).flatten()
    b = jnp.asarray(b, dtype=jnp.float64).flatten()
    dot = jnp.dot(a, b)
    norm = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    if norm < 1e-15:
        return float("nan")
    return float(dot / norm)


def setup():
    """Create environment, simulate data, compute true policy."""
    env = RustBusEnvironment(
        operating_cost=TRUE_OC, replacement_cost=TRUE_RC,
        num_mileage_bins=90, discount_factor=0.9999,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.array([TRUE_OC, TRUE_RC])

    panel = simulate_panel(
        env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED,
    )
    n_obs = sum(len(t.states) for t in panel.trajectories)

    # True policy for comparison
    operator = SoftBellmanOperator(
        problem, jnp.asarray(transitions, dtype=jnp.float64),
    )
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix",
    )
    true_policy = true_result.policy

    print(f"Environment: 90 states, 2 actions, gamma=0.9999")
    print(f"True params: operating_cost={TRUE_OC}, replacement_cost={TRUE_RC}")
    print(f"Simulated: {N_INDIVIDUALS} x {N_PERIODS} = {n_obs:,} observations")
    print()

    return env, utility, problem, transitions, panel, true_params, true_policy


def run_estimator(name, estimator, panel, utility, problem, transitions, true_params, true_policy):
    """Run one estimator and print results."""
    print(f"{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        result = estimator.estimate(panel, utility, problem, transitions)
        elapsed = time.time() - t0

        params = result.parameters
        se = result.standard_errors
        cos = cosine_sim(params, true_params)
        max_diff = float(jnp.max(jnp.abs(result.policy - true_policy)))

        print(f"  Time:        {elapsed:.1f}s")
        print(f"  Converged:   {result.converged}")
        print(f"  theta_c:     {float(params[0]):.6f}  (true: {TRUE_OC})")
        print(f"  RC:          {float(params[1]):.4f}  (true: {TRUE_RC})")
        print(f"  SE(theta_c): {float(se[0]):.6f}")
        print(f"  SE(RC):      {float(se[1]):.4f}")
        print(f"  Log-lik:     {result.log_likelihood:.2f}")
        print(f"  Cos sim:     {cos:.6f}")
        print(f"  Max policy:  {max_diff:.6f}")
        print()
        return {
            "name": name, "params": params, "se": se,
            "ll": result.log_likelihood, "cos": cos, "max_diff": max_diff,
            "time": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAILED after {elapsed:.1f}s: {e}")
        print()
        return None


def main():
    env, utility, problem, transitions, panel, true_params, true_policy = setup()

    results = []

    # 1. NFXP (gold standard)
    from econirl.estimation.nfxp import NFXPEstimator
    r = run_estimator(
        "NFXP-NK (gold standard)",
        NFXPEstimator(
            optimizer="BHHH", inner_solver="policy",
            inner_tol=1e-12, inner_max_iter=200,
            compute_hessian=True, outer_tol=1e-3, verbose=False,
        ),
        panel, utility, problem, transitions, true_params, true_policy,
    )
    if r:
        results.append(r)

    # 2. CCP-NPL
    from econirl.estimation.ccp import CCPEstimator
    r = run_estimator(
        "CCP-NPL (K=20)",
        CCPEstimator(
            num_policy_iterations=20, compute_hessian=True, verbose=False,
        ),
        panel, utility, problem, transitions, true_params, true_policy,
    )
    if r:
        results.append(r)

    # 3. NNES-NPL (correct variant)
    from econirl.estimation.nnes import NNESEstimator
    r = run_estimator(
        "NNES-NPL (Neyman orthogonal)",
        NNESEstimator(
            hidden_dim=64, num_layers=2,
            v_epochs=500, outer_max_iter=200,
            n_outer_iterations=3, verbose=False,
        ),
        panel, utility, problem, transitions, true_params, true_policy,
    )
    if r:
        results.append(r)

    # 4. NNES-NFXP (legacy variant)
    from econirl.estimation.nnes import NNESNFXPEstimator
    r = run_estimator(
        "NNES-NFXP (no orthogonality)",
        NNESNFXPEstimator(
            hidden_dim=64, num_layers=2,
            v_epochs=500, outer_max_iter=200,
            n_outer_iterations=3, verbose=False,
        ),
        panel, utility, problem, transitions, true_params, true_policy,
    )
    if r:
        results.append(r)

    # 5. NNES-NPL with early stopping (test robustness)
    r = run_estimator(
        "NNES-NPL (early stop, 50 epochs)",
        NNESEstimator(
            hidden_dim=64, num_layers=2,
            v_epochs=50, outer_max_iter=200,
            n_outer_iterations=3, verbose=False,
        ),
        panel, utility, problem, transitions, true_params, true_policy,
    )
    if r:
        results.append(r)

    # 6. NNES-NFXP with early stopping (should show bias)
    r = run_estimator(
        "NNES-NFXP (early stop, 50 epochs)",
        NNESNFXPEstimator(
            hidden_dim=64, num_layers=2,
            v_epochs=50, outer_max_iter=200,
            n_outer_iterations=3, verbose=False,
        ),
        panel, utility, problem, transitions, true_params, true_policy,
    )
    if r:
        results.append(r)

    # Summary table
    if results:
        print(f"\n{'='*80}")
        print("  SUMMARY")
        print(f"{'='*80}")
        print(f"{'Estimator':<35} {'theta_c':>10} {'RC':>8} {'Cos':>8} {'Time':>7}")
        print(f"{'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*7}")
        for r in results:
            print(
                f"{r['name']:<35} "
                f"{float(r['params'][0]):>10.6f} "
                f"{float(r['params'][1]):>8.4f} "
                f"{r['cos']:>8.4f} "
                f"{r['time']:>6.1f}s"
            )
        print(f"{'True':.<35} {TRUE_OC:>10.6f} {TRUE_RC:>8.4f}")


if __name__ == "__main__":
    main()
