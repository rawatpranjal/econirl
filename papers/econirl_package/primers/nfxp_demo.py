"""NFXP-NK primer companion script.

Runs NFXP on the Rust bus engine and writes results to JSON
for auto-inclusion in the primer tex file.

Usage:
    python papers/primers/nfxp_demo.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
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

    # NFXP estimation
    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust", verbose=False)
    result = nfxp.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0

    est_params = np.array(result.parameters)
    est_se = np.array(result.standard_errors)
    est_policy = np.array(result.policy)

    # Policy accuracy
    est_actions = np.argmax(est_policy, axis=1)
    true_actions = np.argmax(true_policy, axis=1)
    policy_acc = float(np.mean(est_actions == true_actions))

    # Max policy difference
    max_policy_diff = float(np.max(np.abs(est_policy - true_policy)))

    out = {
        "estimator": "NFXP-NK",
        "paper": "Rust (1987), Iskhakov et al. (2016)",
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
        "estimated_params": {
            "operating_cost": round(float(est_params[0]), 6),
            "replacement_cost": round(float(est_params[1]), 4),
        },
        "standard_errors": {
            "operating_cost": round(float(est_se[0]), 6),
            "replacement_cost": round(float(est_se[1]), 4),
        },
        "log_likelihood": round(float(result.log_likelihood), 2),
        "converged": bool(result.converged),
        "time_seconds": round(elapsed, 2),
        "policy_accuracy": round(policy_acc, 4),
        "max_policy_diff": round(max_policy_diff, 6),
    }

    path = Path(__file__).parent / "tables" / "nfxp_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"NFXP primer results written to {path}")
    print(f"  OC: {out['estimated_params']['operating_cost']} "
          f"(true: {TRUE_OC}, SE: {out['standard_errors']['operating_cost']})")
    print(f"  RC: {out['estimated_params']['replacement_cost']} "
          f"(true: {TRUE_RC}, SE: {out['standard_errors']['replacement_cost']})")
    print(f"  LL: {out['log_likelihood']}, Time: {out['time_seconds']}s")
    print(f"  Policy accuracy: {out['policy_accuracy']:.2%}")


if __name__ == "__main__":
    main()
