"""NNES vs NFXP Monte Carlo comparison.

Runs 5 Monte Carlo replications on a multi-component bus environment.
NFXP is the oracle (knows the tabular structure and solves the Bellman
equation exactly). NNES learns V(s) via a neural network and achieves
comparable precision through the NPL zero-Jacobian property.
"""

import time
import numpy as np
import json

from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.nnes import NNESEstimator, NNESConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

N_REPS = 5
N_INDIVIDUALS = 200
N_PERIODS = 100
DISCOUNT = 0.95
SEED_BASE = 42


def run_one_rep(rep_id: int):
    """Run one Monte Carlo replication."""
    env = MultiComponentBusEnvironment(
        K=2,
        M=10,
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

    true_params = np.array([p for p in env.true_parameters.values()])
    results = {}

    # NFXP oracle (exact Bellman solve)
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-12,
        inner_max_iter=100000,
        switch_tol=1e-3,
        outer_max_iter=200,
        compute_hessian=True,
        verbose=False,
    )
    t0 = time.time()
    nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0

    results["NFXP"] = {
        "params": np.asarray(nfxp_result.parameters).tolist(),
        "ll": float(nfxp_result.log_likelihood),
        "time": nfxp_time,
        "converged": bool(nfxp_result.converged),
        "has_se": nfxp_result.standard_errors is not None,
    }

    # NNES with NPL variant
    nnes = NNESEstimator(
        hidden_dim=32,
        num_layers=2,
        v_lr=1e-3,
        v_epochs=500,
        n_outer_iterations=3,
        compute_se=True,
        verbose=False,
        seed=rep_id,
    )
    t0 = time.time()
    nnes_result = nnes.estimate(panel, utility, problem, transitions)
    nnes_time = time.time() - t0

    results["NNES"] = {
        "params": np.asarray(nnes_result.parameters).tolist(),
        "ll": float(nnes_result.log_likelihood),
        "time": nnes_time,
        "converged": bool(nnes_result.converged),
        "has_se": nnes_result.standard_errors is not None,
    }

    return results, true_params


def main():
    all_results = []
    true_params = None

    for rep in range(N_REPS):
        print(f"--- Rep {rep + 1}/{N_REPS} ---")
        res, tp = run_one_rep(rep)
        all_results.append(res)
        true_params = tp
        for name in ["NFXP", "NNES"]:
            p = np.array(res[name]["params"])
            print(f"  {name}: params={p}, LL={res[name]['ll']:.2f}, "
                  f"time={res[name]['time']:.1f}s")

    n_params = len(true_params)
    param_names = [f"p{i}" for i in range(n_params)]

    print("\n" + "=" * 70)
    print(f"NNES vs NFXP: {N_REPS} Monte Carlo reps")
    print(f"Environment: MultiComponentBus, 2 components x 10 bins = "
          f"{10**2} states, beta={DISCOUNT}")
    print(f"Data: {N_INDIVIDUALS} buses x {N_PERIODS} periods")
    print(f"True params: {true_params}")
    print("=" * 70)

    header = f"{'Metric':<30} {'NFXP':>15} {'NNES':>15}"
    print(header)
    print("-" * len(header))

    for name in ["NFXP", "NNES"]:
        params = np.array([r[name]["params"] for r in all_results])
        bias = params.mean(axis=0) - true_params
        rmse = np.sqrt(((params - true_params) ** 2).mean(axis=0))
        total_rmse = np.sqrt((rmse ** 2).mean())
        lls = [r[name]["ll"] for r in all_results]
        times = [r[name]["time"] for r in all_results]

        if name == "NFXP":
            nfxp_s = {"total_rmse": total_rmse, "mean_ll": np.mean(lls),
                       "mean_time": np.mean(times), "bias": bias, "rmse": rmse}
        else:
            nnes_s = {"total_rmse": total_rmse, "mean_ll": np.mean(lls),
                       "mean_time": np.mean(times), "bias": bias, "rmse": rmse}

    for i, pname in enumerate(param_names):
        print(f"{'Bias (' + pname + ')':<30} {nfxp_s['bias'][i]:>15.4f} {nnes_s['bias'][i]:>15.4f}")
    for i, pname in enumerate(param_names):
        print(f"{'RMSE (' + pname + ')':<30} {nfxp_s['rmse'][i]:>15.4f} {nnes_s['rmse'][i]:>15.4f}")
    print(f"{'Total RMSE':<30} {nfxp_s['total_rmse']:>15.4f} {nnes_s['total_rmse']:>15.4f}")
    print(f"{'Mean LL':<30} {nfxp_s['mean_ll']:>15.2f} {nnes_s['mean_ll']:>15.2f}")
    print(f"{'Mean time (s)':<30} {nfxp_s['mean_time']:>15.1f} {nnes_s['mean_time']:>15.1f}")

    with open("nnes_vs_nfxp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to nnes_vs_nfxp_results.json")


if __name__ == "__main__":
    main()
