"""GCL primer companion script.

Runs GCL alongside MCE-IRL on a 5x5 gridworld and writes
results to JSON for auto-inclusion in the primer tex file.

Usage:
    python papers/primers/gcl_demo.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.gridworld import GridworldEnvironment
from econirl.contrib.gcl import GCLEstimator, GCLConfig
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


GRID_SIZE = 5


def main():
    env = GridworldEnvironment(grid_size=GRID_SIZE, discount_factor=0.99)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = env.get_true_parameter_vector()

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

    panel = env.generate_panel(n_individuals=200, n_periods=50, seed=42)
    n_obs = panel.num_observations

    results = {}

    # MCE-IRL (baseline)
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        learning_rate=0.05, outer_max_iter=300, verbose=False,
    ))
    mce_result = mce.estimate(panel, utility, problem, transitions)
    mce_time = time.time() - t0
    mce_policy = np.array(mce_result.policy)
    mce_acc = float(np.mean(
        np.argmax(mce_policy, axis=1) == np.argmax(true_policy, axis=1)
    ))
    results["mce_irl"] = {
        "policy_accuracy": round(mce_acc, 4),
        "time": round(mce_time, 2),
    }

    # GCL (neural cost)
    t0 = time.time()
    gcl = GCLEstimator(config=GCLConfig(
        hidden_dims=[32, 32],
        cost_lr=0.001,
        max_iterations=100,
        n_sample_trajectories=50,
        verbose=False,
    ))
    gcl_result = gcl.estimate(panel, utility, problem, transitions)
    gcl_time = time.time() - t0
    gcl_policy = np.array(gcl_result.policy)
    gcl_acc = float(np.mean(
        np.argmax(gcl_policy, axis=1) == np.argmax(true_policy, axis=1)
    ))
    results["gcl"] = {
        "policy_accuracy": round(gcl_acc, 4),
        "time": round(gcl_time, 2),
    }

    out = {
        "estimator": "GCL",
        "paper": "Finn, Levine, and Abbeel (2016)",
        "environment": {
            "name": "Gridworld",
            "grid_size": GRID_SIZE,
            "n_states": problem.num_states,
            "n_actions": problem.num_actions,
            "n_observations": n_obs,
            "beta": 0.99,
        },
        "results": results,
    }

    path = Path(__file__).parent / "tables" / "gcl_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"GCL primer results written to {path}")
    print(f"  MCE-IRL: policy_acc={mce_acc:.2%}, time={mce_time:.1f}s")
    print(f"  GCL:     policy_acc={gcl_acc:.2%}, time={gcl_time:.1f}s")


if __name__ == "__main__":
    main()
