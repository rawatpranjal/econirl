"""MPEC vs NFXP Monte Carlo comparison (Su & Judd 2012 style).

Runs 5 Monte Carlo replications on the Rust bus engine and compares
NFXP (SA-then-NK hybrid) against MPEC (SLSQP with native constraints).
Both recover the same MLE; the comparison is about computation.
"""

import time
import numpy as np
import json

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mpec import MPECEstimator, MPECConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

N_REPS = 5
N_INDIVIDUALS = 200
N_PERIODS = 100
TRUE_OC = 0.001
TRUE_RC = 3.0
DISCOUNT = 0.9999
N_BINS = 90
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

    true_params = np.array([TRUE_OC, TRUE_RC])
    results = {}

    # NFXP with hybrid (SA then NK) solver
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-12,
        inner_max_iter=100000,
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

    # MPEC with SLSQP
    mpec = MPECEstimator(
        config=MPECConfig(solver="slsqp", max_iter=500, constraint_tol=1e-8),
        compute_hessian=False,
        verbose=False,
    )
    t0 = time.time()
    mpec_result = mpec.estimate(panel, utility, problem, transitions)
    mpec_time = time.time() - t0

    constraint_viol = mpec_result.metadata.get("final_constraint_violation", float("nan"))
    results["MPEC"] = {
        "params": np.asarray(mpec_result.parameters).tolist(),
        "ll": float(mpec_result.log_likelihood),
        "time": mpec_time,
        "iterations": int(mpec_result.num_iterations),
        "converged": bool(mpec_result.converged),
        "constraint_violation": constraint_viol,
    }

    return results


def main():
    all_results = []
    true_params = np.array([TRUE_OC, TRUE_RC])

    for rep in range(N_REPS):
        print(f"--- Rep {rep + 1}/{N_REPS} ---")
        res = run_one_rep(rep)
        all_results.append(res)
        for name in ["NFXP", "MPEC"]:
            p = np.array(res[name]["params"])
            print(f"  {name}: params={p}, LL={res[name]['ll']:.2f}, "
                  f"time={res[name]['time']:.1f}s, converged={res[name]['converged']}")

    # Summarize
    print("\n" + "=" * 70)
    print(f"MPEC vs NFXP: {N_REPS} Monte Carlo reps")
    print(f"Environment: Rust bus, {N_BINS} bins, beta={DISCOUNT}")
    print(f"Data: {N_INDIVIDUALS} buses x {N_PERIODS} periods = "
          f"{N_INDIVIDUALS * N_PERIODS} obs per rep")
    print("=" * 70)

    header = f"{'Metric':<30} {'NFXP':>15} {'MPEC':>15}"
    print(header)
    print("-" * len(header))

    for name in ["NFXP", "MPEC"]:
        params = np.array([r[name]["params"] for r in all_results])
        bias = params.mean(axis=0) - true_params
        rmse = np.sqrt(((params - true_params) ** 2).mean(axis=0))
        lls = [r[name]["ll"] for r in all_results]
        times = [r[name]["time"] for r in all_results]
        iters = [r[name]["iterations"] for r in all_results]

        if name == "NFXP":
            nfxp_stats = {
                "bias_oc": bias[0], "bias_rc": bias[1],
                "rmse_oc": rmse[0], "rmse_rc": rmse[1],
                "mean_ll": np.mean(lls), "mean_time": np.mean(times),
                "mean_iters": np.mean(iters),
            }
        else:
            mpec_stats = {
                "bias_oc": bias[0], "bias_rc": bias[1],
                "rmse_oc": rmse[0], "rmse_rc": rmse[1],
                "mean_ll": np.mean(lls), "mean_time": np.mean(times),
                "mean_iters": np.mean(iters),
            }

    print(f"{'Mean bias (theta_c)':<30} {nfxp_stats['bias_oc']:>15.6f} {mpec_stats['bias_oc']:>15.6f}")
    print(f"{'Mean bias (RC)':<30} {nfxp_stats['bias_rc']:>15.4f} {mpec_stats['bias_rc']:>15.4f}")
    print(f"{'RMSE (theta_c)':<30} {nfxp_stats['rmse_oc']:>15.6f} {mpec_stats['rmse_oc']:>15.6f}")
    print(f"{'RMSE (RC)':<30} {nfxp_stats['rmse_rc']:>15.4f} {mpec_stats['rmse_rc']:>15.4f}")
    print(f"{'Mean LL':<30} {nfxp_stats['mean_ll']:>15.2f} {mpec_stats['mean_ll']:>15.2f}")
    print(f"{'Mean time (s)':<30} {nfxp_stats['mean_time']:>15.1f} {mpec_stats['mean_time']:>15.1f}")
    print(f"{'Mean iterations':<30} {nfxp_stats['mean_iters']:>15.1f} {mpec_stats['mean_iters']:>15.1f}")

    # Save results
    with open("mpec_vs_nfxp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to mpec_vs_nfxp_results.json")


if __name__ == "__main__":
    main()
