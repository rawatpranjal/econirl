"""TD-CCP primer companion script.

Runs TD-CCP on the Rust bus engine and writes results to JSON
for auto-inclusion in the primer tex file.

Usage:
    python papers/primers/tdccp_demo.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


TRUE_OC = 0.001
TRUE_RC = 3.0
N_BINS = 90
BETA = 0.95
N_INDIVIDUALS = 500
N_PERIODS = 100


def main():
    env = RustBusEnvironment(
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        num_mileage_bins=N_BINS,
        discount_factor=BETA,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.array([TRUE_OC, TRUE_RC])

    panel = simulate_panel(env, n_individuals=N_INDIVIDUALS,
                           n_periods=N_PERIODS, seed=42)
    n_obs = sum(len(t.states) for t in panel.trajectories)

    # Pre-estimation diagnostics
    features = np.array(utility.feature_matrix)
    n_states = problem.num_states
    n_actions = problem.num_actions
    n_features = features.shape[2]
    flat_features = features.reshape(n_states * n_actions, n_features)
    rank = int(np.linalg.matrix_rank(flat_features))
    nonzero_mask = np.any(flat_features != 0, axis=1)
    cond = float(np.linalg.cond(flat_features[nonzero_mask]))
    covered = int(len(np.unique(panel.get_all_states())))

    # True policy
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
    true_policy = np.array(true_result.policy)

    results = {}

    # NFXP (baseline for comparison)
    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust", verbose=False)
    nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    nfxp_p = np.array(nfxp_result.parameters)
    nfxp_se = np.array(nfxp_result.standard_errors)
    results["nfxp"] = {
        "operating_cost": round(float(nfxp_p[0]), 6),
        "replacement_cost": round(float(nfxp_p[1]), 4),
        "se_oc": round(float(nfxp_se[0]), 6),
        "se_rc": round(float(nfxp_se[1]), 4),
        "ll": round(float(nfxp_result.log_likelihood), 2),
        "time": round(nfxp_time, 2),
    }

    # TD-CCP (semi-gradient, the default and recommended method)
    t0 = time.time()
    tdccp = TDCCPEstimator(config=TDCCPConfig(
        method="semigradient",
        basis_dim=8,
        cross_fitting=False,
        robust_se=True,
        verbose=False,
    ))
    tdccp_result = tdccp.estimate(panel, utility, problem, transitions)
    tdccp_time = time.time() - t0
    tdccp_p = np.array(tdccp_result.parameters)
    tdccp_se = np.array(tdccp_result.standard_errors)

    est_policy = np.array(tdccp_result.policy)
    est_actions = np.argmax(est_policy, axis=1)
    true_actions = np.argmax(true_policy, axis=1)
    policy_acc = float(np.mean(est_actions == true_actions))

    results["tdccp"] = {
        "operating_cost": round(float(tdccp_p[0]), 6),
        "replacement_cost": round(float(tdccp_p[1]), 4),
        "se_oc": round(float(tdccp_se[0]), 6),
        "se_rc": round(float(tdccp_se[1]), 4),
        "ll": round(float(tdccp_result.log_likelihood), 2),
        "time": round(tdccp_time, 2),
        "policy_accuracy": round(policy_acc, 4),
    }

    out = {
        "estimator": "TD-CCP",
        "paper": "Adusumilli and Eckardt (2025)",
        "environment": {
            "name": "Rust Bus Engine",
            "n_states": n_states,
            "n_actions": n_actions,
            "n_bins": N_BINS,
            "beta": BETA,
            "n_individuals": N_INDIVIDUALS,
            "n_periods": N_PERIODS,
            "n_observations": n_obs,
        },
        "diagnostics": {
            "feature_rank": rank,
            "n_features": n_features,
            "condition_number": round(cond, 1),
            "state_coverage": covered,
        },
        "true_params": {
            "operating_cost": TRUE_OC,
            "replacement_cost": TRUE_RC,
        },
        "results": results,
    }

    path = Path(__file__).parent / "tables" / "tdccp_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"TD-CCP primer results written to {path}")
    print(f"\n  {'':>5} {'True':>10} {'NFXP':>10} {'TD-CCP':>10}")
    print(f"  {'-'*38}")
    print(f"  {'OC':>5} {TRUE_OC:>10} "
          f"{results['nfxp']['operating_cost']:>10} "
          f"{results['tdccp']['operating_cost']:>10}")
    print(f"  {'RC':>5} {TRUE_RC:>10} "
          f"{results['nfxp']['replacement_cost']:>10} "
          f"{results['tdccp']['replacement_cost']:>10}")
    print(f"\n  TD-CCP policy accuracy: {policy_acc:.2%}")
    print(f"  TD-CCP time: {tdccp_time:.2f}s, NFXP time: {nfxp_time:.2f}s")


if __name__ == "__main__":
    main()
