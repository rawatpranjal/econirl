"""SEES vs NFXP Monte Carlo comparison on large state space.

Runs 5 Monte Carlo replications on a Rust bus engine with 500 mileage
bins and high discount factor. SEES avoids the inner fixed-point loop
by approximating V(s) with a small Fourier basis, making it faster
when the state space is large and contraction is slow.
"""

import time
import numpy as np
import json

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.sees import SEESEstimator, SEESConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

N_REPS = 5
N_INDIVIDUALS = 200
N_PERIODS = 100
TRUE_OC = 0.001
TRUE_RC = 3.0
DISCOUNT = 0.9999
N_BINS = 200
SEED_BASE = 42


def run_one_rep(rep_id: int):
    """Run one Monte Carlo replication."""
    env = RustBusEnvironment(
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        num_mileage_bins=N_BINS,
        discount_factor=DISCOUNT,
        seed=SEED_BASE + rep_id,
    )
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)

    panel = simulate_panel(
        env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS,
        seed=SEED_BASE + rep_id + 1000,
    )

    results = {}

    # NFXP with hybrid solver
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-12,
        inner_max_iter=300000,
        switch_tol=1e-3,
        outer_max_iter=200,
        compute_hessian=False,
        verbose=False,
    )
    t0 = time.time()
    nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0

    results["NFXP"] = {
        "params": np.asarray(nfxp_result.parameters).tolist(),
        "ll": float(nfxp_result.log_likelihood),
        "time": nfxp_time,
        "iterations": int(nfxp_result.num_iterations),
        "converged": bool(nfxp_result.converged),
    }

    # SEES with Fourier basis
    sees = SEESEstimator(
        basis_type="fourier",
        basis_dim=8,
        penalty_weight=10.0,
        max_iter=500,
        compute_se=False,
        verbose=False,
    )
    t0 = time.time()
    sees_result = sees.estimate(panel, utility, problem, transitions)
    sees_time = time.time() - t0

    bellman_viol = sees_result.metadata.get("bellman_violation", float("nan"))
    results["SEES"] = {
        "params": np.asarray(sees_result.parameters).tolist(),
        "ll": float(sees_result.log_likelihood),
        "time": sees_time,
        "iterations": int(sees_result.num_iterations),
        "converged": bool(sees_result.converged),
        "bellman_violation": bellman_viol,
        "basis_dim": 8,
    }

    return results


def main():
    all_results = []
    true_params = np.array([TRUE_OC, TRUE_RC])

    for rep in range(N_REPS):
        print(f"--- Rep {rep + 1}/{N_REPS} ---")
        res = run_one_rep(rep)
        all_results.append(res)
        for name in ["NFXP", "SEES"]:
            p = np.array(res[name]["params"])
            print(f"  {name}: params={p}, LL={res[name]['ll']:.2f}, "
                  f"time={res[name]['time']:.1f}s, converged={res[name]['converged']}")

    # Summarize
    print("\n" + "=" * 70)
    print(f"SEES vs NFXP: {N_REPS} Monte Carlo reps")
    print(f"Environment: Rust bus, {N_BINS} bins, beta={DISCOUNT}")
    print(f"Data: {N_INDIVIDUALS} buses x {N_PERIODS} periods")
    print("=" * 70)

    header = f"{'Metric':<30} {'NFXP':>15} {'SEES':>15}"
    print(header)
    print("-" * len(header))

    stats = {}
    for name in ["NFXP", "SEES"]:
        params = np.array([r[name]["params"] for r in all_results])
        bias = params.mean(axis=0) - true_params
        rmse = np.sqrt(((params - true_params) ** 2).mean(axis=0))
        lls = [r[name]["ll"] for r in all_results]
        times = [r[name]["time"] for r in all_results]
        stats[name] = {
            "bias_oc": bias[0], "bias_rc": bias[1],
            "rmse_oc": rmse[0], "rmse_rc": rmse[1],
            "mean_ll": np.mean(lls), "mean_time": np.mean(times),
        }

    for metric, key in [
        ("Mean bias (theta_c)", "bias_oc"),
        ("Mean bias (RC)", "bias_rc"),
        ("RMSE (theta_c)", "rmse_oc"),
        ("RMSE (RC)", "rmse_rc"),
        ("Mean LL", "mean_ll"),
        ("Mean time (s)", "mean_time"),
    ]:
        nv = stats["NFXP"][key]
        sv = stats["SEES"][key]
        if "LL" in metric:
            print(f"{metric:<30} {nv:>15.2f} {sv:>15.2f}")
        elif "time" in metric:
            print(f"{metric:<30} {nv:>15.1f} {sv:>15.1f}")
        elif "bias" in key and "oc" in key:
            print(f"{metric:<30} {nv:>15.6f} {sv:>15.6f}")
        else:
            print(f"{metric:<30} {nv:>15.4f} {sv:>15.4f}")

    speedup = stats["NFXP"]["mean_time"] / max(stats["SEES"]["mean_time"], 0.01)
    print(f"{'Speedup (NFXP/SEES)':<30} {'':>15} {speedup:>15.1f}x")

    with open("sees_vs_nfxp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to sees_vs_nfxp_results.json")


if __name__ == "__main__":
    main()
