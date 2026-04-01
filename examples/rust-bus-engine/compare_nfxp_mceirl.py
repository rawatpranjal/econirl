#!/usr/bin/env python3
"""
NFXP vs MCE-IRL Equivalence on Rust Bus Engine
================================================

Demonstrates that NFXP (structural MLE) and MCE-IRL (maximum causal entropy
inverse RL) recover identical parameters on the Rust (1987) bus engine
replacement problem.

Both methods maximize the same DDC log-likelihood:
    max_θ Σ_t log π_θ(a_t | s_t)
where π_θ is the logit choice probability from the soft Bellman equation.

Experiments:
    1. Simulated data (known true params) — both recover ground truth
    2. Original Rust (1987) data — both agree on estimated params

Usage:
    python examples/rust-bus-engine/compare_nfxp_mceirl.py
"""

import time

import jax.numpy as jnp
import numpy as np

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator, estimate_transitions_from_panel
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


def run_nfxp(panel, utility, problem, transitions, verbose=False):
    """Run NFXP with BHHH optimizer and analytical gradient."""
    estimator = NFXPEstimator(
        optimizer="BHHH",
        inner_solver="policy",
        inner_tol=1e-12,
        inner_max_iter=200,
        compute_hessian=True,
        outer_tol=1e-3,
        verbose=verbose,
    )
    return estimator.estimate(panel, utility, problem, transitions)


def run_mce_irl(panel, utility, problem, transitions, verbose=False):
    """Run MCE-IRL with L-BFGS-B, tuned for gamma=0.9999."""
    config = MCEIRLConfig(
        optimizer="L-BFGS-B",
        inner_solver="policy",  # Matrix solve, same as NFXP — essential for gamma=0.9999
        inner_max_iter=200,
        inner_tol=1e-6,  # Policy convergence tolerance (float32 precision limit ~1e-7)
        outer_max_iter=500,
        outer_tol=1e-6,
        svf_max_iter=10000,
        svf_tol=1e-8,
        compute_se=True,
        se_method="hessian",
        verbose=verbose,
    )
    estimator = MCEIRLEstimator(config=config)
    return estimator.estimate(panel, utility, problem, transitions)


def cosine_sim(a, b):
    """Cosine similarity between two parameter vectors."""
    a, b = jnp.asarray(a, dtype=jnp.float64), jnp.asarray(b, dtype=jnp.float64)
    return (a @ b / (jnp.linalg.norm(a) * jnp.linalg.norm(b))).item()


def compare_results(title, result_nfxp, result_mce, param_names, true_params=None):
    """Print a formatted comparison table."""
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print(f"{'=' * 72}")

    p_nfxp = result_nfxp.parameters
    p_mce = result_mce.parameters
    se_nfxp = result_nfxp.standard_errors
    se_mce = result_mce.standard_errors

    # Parameter table
    header = f"  {'Parameter':<20} {'NFXP':>12} {'MCE-IRL':>12}"
    if true_params is not None:
        header += f" {'True':>12}"
    print(header)
    print("  " + "-" * len(header.strip()))

    for i, name in enumerate(param_names):
        nfxp_val = p_nfxp[i].item()
        mce_val = p_mce[i].item()
        line = f"  {name:<20} {nfxp_val:>12.6f} {mce_val:>12.6f}"
        if true_params is not None:
            line += f" {true_params[i]:>12.6f}"
        print(line)

    if se_nfxp is not None and se_mce is not None:
        print()
        for i, name in enumerate(param_names):
            se_n = se_nfxp[i].item() if i < len(se_nfxp) else float("nan")
            se_m = se_mce[i].item() if i < len(se_mce) else float("nan")
            pad = " " * max(0, 13 - len(name))
            print(f"  SE({name}){pad} {se_n:>12.6f} {se_m:>12.6f}")

    # Summary metrics
    ll_nfxp = result_nfxp.log_likelihood
    ll_mce = result_mce.log_likelihood
    cos = cosine_sim(p_nfxp, p_mce)
    policy_diff = float("nan")
    if result_nfxp.policy is not None and result_mce.policy is not None:
        policy_diff = float(jnp.abs(result_nfxp.policy - result_mce.policy).max())

    print(f"\n  {'Metric':<30} {'Value':>15}")
    print("  " + "-" * 47)
    print(f"  {'LL (NFXP)':<30} {ll_nfxp:>15.4f}")
    print(f"  {'LL (MCE-IRL)':<30} {ll_mce:>15.4f}")
    print(f"  {'|LL difference|':<30} {abs(ll_nfxp - ll_mce):>15.4f}")
    print(f"  {'Cosine similarity':<30} {cos:>15.6f}")
    if not np.isnan(policy_diff):
        print(f"  {'Max |policy difference|':<30} {policy_diff:>15.6f}")
    print(f"  {'NFXP converged':<30} {str(result_nfxp.converged):>15}")
    print(f"  {'MCE-IRL converged':<30} {str(result_mce.converged):>15}")

    if true_params is not None:
        true_t = jnp.asarray(true_params, dtype=jnp.float64)
        cos_nfxp = cosine_sim(p_nfxp, true_t)
        cos_mce = cosine_sim(p_mce, true_t)
        print(f"\n  {'Cosine to true (NFXP)':<30} {cos_nfxp:>15.6f}")
        print(f"  {'Cosine to true (MCE-IRL)':<30} {cos_mce:>15.6f}")

    # Verdict
    ll_ok = abs(ll_nfxp - ll_mce) < 1.0
    cos_ok = cos > 0.999
    pol_ok = np.isnan(policy_diff) or policy_diff < 0.01
    all_ok = ll_ok and cos_ok and pol_ok

    print(f"\n  Checks: |dLL|<1.0 {'PASS' if ll_ok else 'FAIL'}  "
          f"cos>0.999 {'PASS' if cos_ok else 'FAIL'}  "
          f"max|dpolicy|<0.01 {'PASS' if pol_ok else 'FAIL'}")
    if all_ok:
        print("  >>> EQUIVALENCE DEMONSTRATED <<<")
    else:
        print("  >>> DIFFERENCES DETECTED <<<")

    return all_ok


def main():
    print("=" * 72)
    print("  NFXP vs MCE-IRL Equivalence — Rust (1987) Bus Engine")
    print("=" * 72)

    # ==================================================================
    # Experiment 1: Simulated Data (known true parameters)
    # ==================================================================
    print("\n" + "=" * 72)
    print("  EXPERIMENT 1: Simulated Data")
    print("  True params: operating_cost=0.001, replacement_cost=3.0")
    print("=" * 72)

    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=90,
        discount_factor=0.9999,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = [0.001, 3.0]

    print("\nSimulating panel: 200 individuals x 100 periods...")
    panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)
    n_obs = sum(len(t.states) for t in panel.trajectories)
    print(f"  {n_obs:,} observations")

    print("\n--- Running NFXP ---")
    t0 = time.time()
    result_nfxp = run_nfxp(panel, utility, problem, transitions, verbose=True)
    t_nfxp = time.time() - t0
    print(f"  Time: {t_nfxp:.1f}s")

    print("\n--- Running MCE-IRL ---")
    t0 = time.time()
    result_mce = run_mce_irl(panel, utility, problem, transitions, verbose=True)
    t_mce = time.time() - t0
    print(f"  Time: {t_mce:.1f}s")

    ok1 = compare_results(
        "Experiment 1: Simulated Data — Parameter Recovery",
        result_nfxp, result_mce,
        env.parameter_names, true_params,
    )

    # ==================================================================
    # Experiment 2: Original Rust (1987) Data
    # ==================================================================
    print("\n\n" + "=" * 72)
    print("  EXPERIMENT 2: Original Rust (1987) Data")
    print("=" * 72)

    print("\nLoading original data...")
    df = load_rust_bus(original=True)
    panel_orig = load_rust_bus(original=True, as_panel=True)
    print(f"  {len(df):,} observations, {df['bus_id'].nunique()} buses")

    env_orig = RustBusEnvironment(num_mileage_bins=90, discount_factor=0.9999)
    utility_orig = LinearUtility.from_environment(env_orig)
    problem_orig = env_orig.problem_spec
    transitions_orig = estimate_transitions_from_panel(
        panel_orig, num_states=90, max_increment=2,
    ).astype(jnp.float32)  # Ensure float32 for consistent dtype with reward computation

    print("\n--- Running NFXP ---")
    t0 = time.time()
    result_nfxp_orig = run_nfxp(
        panel_orig, utility_orig, problem_orig, transitions_orig, verbose=True,
    )
    t_nfxp = time.time() - t0
    print(f"  Time: {t_nfxp:.1f}s")

    print("\n--- Running MCE-IRL ---")
    t0 = time.time()
    result_mce_orig = run_mce_irl(
        panel_orig, utility_orig, problem_orig, transitions_orig, verbose=True,
    )
    t_mce = time.time() - t0
    print(f"  Time: {t_mce:.1f}s")

    ok2 = compare_results(
        "Experiment 2: Original Rust (1987) Data — Cross-Validation",
        result_nfxp_orig, result_mce_orig,
        env_orig.parameter_names,
    )

    # Rust-style parameterization for reference
    op_nfxp = result_nfxp_orig.parameters[0].item()
    rc_nfxp = result_nfxp_orig.parameters[1].item()
    op_mce = result_mce_orig.parameters[0].item()
    rc_mce = result_mce_orig.parameters[1].item()
    print(f"\n  Rust parameterization:")
    print(f"    NFXP:    c = {op_nfxp / 0.001:.4f}, RC = {rc_nfxp:.4f}")
    print(f"    MCE-IRL: c = {op_mce / 0.001:.4f}, RC = {rc_mce:.4f}")

    # ==================================================================
    # Final Summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("  FINAL SUMMARY")
    print("=" * 72)
    print(f"  Experiment 1 (simulated):  {'PASS' if ok1 else 'FAIL'}")
    print(f"  Experiment 2 (original):   {'PASS' if ok2 else 'FAIL'}")
    if ok1 and ok2:
        print("\n  Both experiments confirm NFXP-MCE IRL equivalence.")
    print()


if __name__ == "__main__":
    main()
