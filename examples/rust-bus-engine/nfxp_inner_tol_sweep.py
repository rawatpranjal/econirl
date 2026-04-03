#!/usr/bin/env python3
"""
NFXP Inner Loop Tolerance Sweep
================================

Does NFXP remain unbiased and efficient when the Bellman equation is
solved approximately? This experiment answers the question by sweeping
the inner loop convergence tolerance from 1e-12 (machine precision) to
1e-1 (very approximate) and comparing the resulting structural parameter
estimates.

Motivation: John's question about whether a neural extension of NFXP
can get away with approximate Bellman solves. If the parameter estimates
degrade smoothly and only at very loose tolerances, approximate solvers
(including neural ones) may be viable. If they break sharply, the inner
solve must be tight for efficiency.

Setup:
    - Rust (1987) bus engine replacement model
    - 90 mileage bins, 2 actions, beta = 0.9999
    - True parameters: theta_c = 0.001, RC = 3.0
    - 500 individuals x 100 periods (simulated)
    - NFXP with BHHH optimizer, value iteration inner solver
    - Inner tolerance swept: 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1

Usage:
    python examples/rust-bus-engine/nfxp_inner_tol_sweep.py
"""

import json
import os
import time

import jax.numpy as jnp
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

# ── Configuration ──
N_INDIVIDUALS = 500
N_PERIODS = 100
SEED = 42
TRUE_THETA_C = 0.001
TRUE_RC = 3.0

INNER_TOLERANCES = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "nfxp_inner_tol_results.json")


def main():
    # Set up environment and simulate data once
    env = RustBusEnvironment(
        operating_cost=TRUE_THETA_C,
        replacement_cost=TRUE_RC,
        num_mileage_bins=90,
        discount_factor=0.9999,
        seed=SEED,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    print(f"Environment: 90 states, 2 actions, beta = 0.9999")
    print(f"True parameters: theta_c = {TRUE_THETA_C}, RC = {TRUE_RC}")
    print(f"Simulating panel: {N_INDIVIDUALS} individuals x {N_PERIODS} periods...")

    panel = simulate_panel(env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)
    n_obs = sum(len(t.states) for t in panel.trajectories)
    print(f"Panel: {n_obs:,} observations\n")

    results = []

    # Header
    print(f"{'inner_tol':>12s}  {'theta_c':>10s}  {'RC':>10s}  "
          f"{'SE(tc)':>10s}  {'SE(RC)':>10s}  {'LL':>14s}  "
          f"{'outer_iter':>10s}  {'time_s':>8s}  {'converged':>9s}")
    print("-" * 110)

    for tol in INNER_TOLERANCES:
        estimator = NFXPEstimator(
            optimizer="BHHH",
            inner_solver="value",
            inner_tol=tol,
            inner_max_iter=100000,
            outer_tol=1e-6,
            outer_max_iter=1000,
            se_method="asymptotic",
            verbose=False,
        )

        t0 = time.time()
        try:
            result = estimator.estimate(panel, utility, problem, transitions)
            elapsed = time.time() - t0

            theta_c_hat = float(result.parameters[0])
            rc_hat = float(result.parameters[1])
            se_tc = float(result.standard_errors[0]) if result.standard_errors is not None else float("nan")
            se_rc = float(result.standard_errors[1]) if result.standard_errors is not None else float("nan")
            ll = float(result.log_likelihood)
            converged = bool(result.converged)
            n_outer = int(result.convergence_info.get("num_iterations", 0)) if hasattr(result, "convergence_info") and result.convergence_info else 0

            row = {
                "inner_tol": tol,
                "theta_c": theta_c_hat,
                "RC": rc_hat,
                "SE_theta_c": se_tc,
                "SE_RC": se_rc,
                "log_likelihood": ll,
                "outer_iterations": n_outer,
                "time_seconds": round(elapsed, 2),
                "converged": converged,
                "bias_theta_c": theta_c_hat - TRUE_THETA_C,
                "bias_RC": rc_hat - TRUE_RC,
            }

            print(f"{tol:>12.0e}  {theta_c_hat:>10.6f}  {rc_hat:>10.4f}  "
                  f"{se_tc:>10.6f}  {se_rc:>10.4f}  {ll:>14.2f}  "
                  f"{n_outer:>10d}  {elapsed:>8.1f}  {str(converged):>9s}")

        except Exception as e:
            elapsed = time.time() - t0
            row = {
                "inner_tol": tol,
                "theta_c": None,
                "RC": None,
                "SE_theta_c": None,
                "SE_RC": None,
                "log_likelihood": None,
                "outer_iterations": None,
                "time_seconds": round(elapsed, 2),
                "converged": False,
                "bias_theta_c": None,
                "bias_RC": None,
                "error": str(e),
            }
            print(f"{tol:>12.0e}  {'FAILED':>10s}  {'':>10s}  "
                  f"{'':>10s}  {'':>10s}  {'':>14s}  "
                  f"{'':>10s}  {elapsed:>8.1f}  {'False':>9s}  {e}")

        results.append(row)

    # Summary
    print("\n" + "=" * 110)
    print("Summary: Bias relative to true parameters")
    print(f"{'inner_tol':>12s}  {'bias(theta_c)':>14s}  {'bias(RC)':>14s}  {'pct_bias(tc)':>14s}  {'pct_bias(RC)':>14s}")
    print("-" * 72)
    for r in results:
        if r["theta_c"] is not None:
            pct_tc = 100 * r["bias_theta_c"] / TRUE_THETA_C
            pct_rc = 100 * r["bias_RC"] / TRUE_RC
            print(f"{r['inner_tol']:>12.0e}  {r['bias_theta_c']:>14.8f}  {r['bias_RC']:>14.6f}  {pct_tc:>13.4f}%  {pct_rc:>13.4f}%")

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
