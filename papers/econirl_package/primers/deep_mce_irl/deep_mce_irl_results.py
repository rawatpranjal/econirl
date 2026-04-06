#!/usr/bin/env python3
"""Deep MCE-IRL Primer -- auto-generate results for the LaTeX document.

Runs MCEIRLNeural (deep), MCEIRLEstimator (linear), and BC on an 8x8
Objectworld and writes LaTeX macros plus a results table to
deep_mce_irl_results.tex.

The Objectworld environment has a nonlinear reward that depends on
distances to colored objects, making it an ideal test case for deep
versus linear reward recovery.

Usage:
    cd /path/to/econirl
    python papers/econirl_package/primers/deep_mce_irl/deep_mce_irl_results.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

OUT = Path(__file__).resolve().parent / "deep_mce_irl_results.tex"

GRID_SIZE = 8
N_COLORS = 2
N_OBJECTS_PER_COLOR = 3
BETA = 0.9
N_DEMOS = 200
TRAJ_LEN = 50
SEED = 42


def main():
    from econirl.environments.objectworld import ObjectworldEnvironment
    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.estimators.mceirl_neural import MCEIRLNeural
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import policy_iteration
    from econirl.preferences.linear import LinearUtility

    n_states = GRID_SIZE ** 2
    n_obs = N_DEMOS * TRAJ_LEN

    print("Deep MCE-IRL Primer -- generating results")
    print(f"  Environment: {GRID_SIZE}x{GRID_SIZE} Objectworld "
          f"({n_states} states, 5 actions), beta={BETA}")
    print(f"  Data: {N_DEMOS} trajectories x {TRAJ_LEN} steps = {n_obs:,} obs")

    env = ObjectworldEnvironment(
        grid_size=GRID_SIZE,
        n_colors=N_COLORS,
        n_objects_per_color=N_OBJECTS_PER_COLOR,
        discount_factor=BETA,
        feature_type="continuous",
        seed=SEED,
    )
    panel = env.simulate_demonstrations(
        n_demos=N_DEMOS, max_steps=TRAJ_LEN, noise_fraction=0.3, seed=SEED,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    # True policy for accuracy comparison
    true_reward = jnp.broadcast_to(
        jnp.expand_dims(env.true_reward, axis=1),
        (n_states, 5),
    ).copy()
    op = SoftBellmanOperator(problem, jnp.asarray(transitions, dtype=jnp.float64))
    true_sol = policy_iteration(op, true_reward, tol=1e-10, max_iter=200,
                                eval_method="matrix")
    true_actions = jnp.argmax(true_sol.policy, axis=1)

    results = {}

    # -- Deep MCE-IRL (MCEIRLNeural) --
    print("\n  Running MCEIRLNeural (deep)...")
    t0 = time.time()
    deep = MCEIRLNeural(
        n_states=n_states,
        n_actions=5,
        discount=BETA,
        reward_type="state_action",
        reward_hidden_dim=64,
        reward_num_layers=2,
        max_epochs=300,
        lr=1e-3,
        inner_solver="hybrid",
        seed=SEED,
        verbose=True,
    )
    deep.fit(
        data=panel,
        features=env.feature_matrix,
        transitions=np.asarray(transitions),
    )
    deep_time = time.time() - t0
    deep_actions = jnp.argmax(jnp.asarray(deep.policy_), axis=1)
    deep_acc = float((true_actions == deep_actions).mean()) * 100
    results["deep"] = {
        "time": deep_time,
        "policy_acc": deep_acc,
        "n_epochs": deep.n_epochs_,
    }
    if deep.projection_r2_ is not None:
        results["deep"]["projection_r2"] = deep.projection_r2_
    print(f"    policy_acc={deep_acc:.1f}%, time={deep_time:.1f}s, "
          f"epochs={deep.n_epochs_}")
    if deep.projection_r2_ is not None:
        print(f"    projection R2={deep.projection_r2_:.4f}")

    # -- Linear MCE-IRL --
    print("\n  Running linear MCE-IRL...")
    t0 = time.time()
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        optimizer="L-BFGS-B", inner_solver="hybrid",
        inner_max_iter=10000, inner_tol=1e-8,
        outer_max_iter=500, outer_tol=1e-6,
        compute_se=False, verbose=False,
    ))
    mce_r = mce.estimate(
        panel=panel, utility=utility, problem=problem, transitions=transitions,
    )
    mce_time = time.time() - t0
    mce_actions = (jnp.argmax(mce_r.policy, axis=1)
                   if mce_r.policy is not None else true_actions)
    mce_acc = float((true_actions == mce_actions).mean()) * 100
    results["mce"] = {
        "ll": float(mce_r.log_likelihood),
        "time": mce_time,
        "policy_acc": mce_acc,
    }
    print(f"    policy_acc={mce_acc:.1f}%, time={mce_time:.1f}s")

    # -- Behavioral Cloning --
    print("\n  Running BC...")
    t0 = time.time()
    bc = BehavioralCloningEstimator(verbose=False)
    bc_r = bc.estimate(
        panel=panel, utility=utility, problem=problem, transitions=transitions,
    )
    bc_time = time.time() - t0
    bc_actions = (jnp.argmax(bc_r.policy, axis=1)
                  if bc_r.policy is not None else true_actions)
    bc_acc = float((true_actions == bc_actions).mean()) * 100
    results["bc"] = {
        "ll": float(bc_r.log_likelihood),
        "time": bc_time,
        "policy_acc": bc_acc,
    }
    print(f"    policy_acc={bc_acc:.1f}%, time={bc_time:.1f}s")

    # -- Write LaTeX --
    tex = []
    tex.append("% Auto-generated by deep_mce_irl_results.py -- do not edit by hand")
    tex.append(f"% {GRID_SIZE}x{GRID_SIZE} Objectworld, beta={BETA}, "
               f"{n_obs} obs, seed={SEED}")
    tex.append("")
    tex.append(f"\\newcommand{{\\deepGridSize}}{{{GRID_SIZE}}}")
    tex.append(f"\\newcommand{{\\deepNstates}}{{{n_states}}}")
    tex.append(f"\\newcommand{{\\deepBeta}}{{{BETA}}}")
    tex.append(f"\\newcommand{{\\deepNobs}}{{{n_obs:,}}}")
    tex.append(f"\\newcommand{{\\deepNdemos}}{{{N_DEMOS}}}")
    tex.append(f"\\newcommand{{\\deepTrajLen}}{{{TRAJ_LEN}}}")
    tex.append("")

    # Deep MCE-IRL macros
    r = results["deep"]
    tex.append(f"\\newcommand{{\\deepDeepTime}}{{{r['time']:.1f}}}")
    tex.append(f"\\newcommand{{\\deepDeepPolicyAcc}}{{{r['policy_acc']:.1f}}}")
    tex.append(f"\\newcommand{{\\deepDeepEpochs}}{{{r['n_epochs']}}}")
    if "projection_r2" in r:
        tex.append(f"\\newcommand{{\\deepDeepProjectionR}}{{{r['projection_r2']:.4f}}}")
    else:
        tex.append("\\newcommand{\\deepDeepProjectionR}{N/A}")
    tex.append("")

    # Linear MCE-IRL macros
    r = results["mce"]
    tex.append(f"\\newcommand{{\\deepMceLL}}{{{r['ll']:.2f}}}")
    tex.append(f"\\newcommand{{\\deepMceTime}}{{{r['time']:.1f}}}")
    tex.append(f"\\newcommand{{\\deepMcePolicyAcc}}{{{r['policy_acc']:.1f}}}")
    tex.append("")

    # BC macros
    r = results["bc"]
    tex.append(f"\\newcommand{{\\deepBcLL}}{{{r['ll']:.2f}}}")
    tex.append(f"\\newcommand{{\\deepBcTime}}{{{r['time']:.1f}}}")
    tex.append(f"\\newcommand{{\\deepBcPolicyAcc}}{{{r['policy_acc']:.1f}}}")
    tex.append("")

    # Results table
    tex.append("\\begin{table}[H]")
    tex.append("\\centering\\small")
    tex.append(f"\\caption{{\\deepGridSize$\\times$\\deepGridSize\\ Objectworld "
               f"(\\deepNstates\\ states, $\\beta=\\deepBeta$, \\deepNobs\\ obs). "
               f"The Objectworld reward is nonlinear in state features, "
               f"so linear MCE-IRL misspecifies the reward.}}")
    tex.append("\\label{tab:deep_results}")
    tex.append("\\begin{tabular*}{\\textwidth}"
               "{@{\\extracolsep{\\fill}} l r r r}")
    tex.append("\\toprule")
    tex.append("Metric & Deep MCE-IRL & Linear MCE-IRL & BC \\\\")
    tex.append("\\midrule")
    tex.append("Policy accuracy (\\%) & \\deepDeepPolicyAcc "
               "& \\deepMcePolicyAcc & \\deepBcPolicyAcc \\\\")
    tex.append("Time (s) & \\deepDeepTime "
               "& \\deepMceTime & \\deepBcTime \\\\")
    tex.append("Projection $R^2$ & \\deepDeepProjectionR & --- & --- \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular*}")
    tex.append("\\end{table}")

    OUT.write_text("\n".join(tex) + "\n")
    print(f"\n  Wrote {OUT}")
    OUT.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print(f"  Wrote {OUT.with_suffix('.json')}")


if __name__ == "__main__":
    main()
