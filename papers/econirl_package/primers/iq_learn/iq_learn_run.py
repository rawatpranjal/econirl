#!/usr/bin/env python3
"""IQ-Learn Primer — low-data stability benchmark.

Key result: With only 5 expert trajectories, IQ-Learn achieves stable high
policy accuracy across seeds. GAIL collapses (high variance from adversarial
instability). MCE-IRL matches accuracy but is slower due to inner VI at
beta=0.9999. BC provides the floor (no MDP structure).

Usage:  python iq_learn_run.py
Output: iq_learn_results.tex, iq_learn_results.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "src"))

import jax.numpy as jnp

PRIMER_DIR = Path(__file__).resolve().parent
OUT_TEX = PRIMER_DIR / "iq_learn_results.tex"
OUT_JSON = PRIMER_DIR / "iq_learn_results.json"

SEEDS = [42, 7, 123]
N_EXPERT_TRAJS = 5
N_PERIODS = 100
N_ORACLE_TRAJS = 500


def cosine_similarity(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def policy_accuracy(pi_hat, pi_oracle):
    """Fraction of states where greedy action matches oracle."""
    greedy_hat = np.argmax(np.array(pi_hat), axis=1)
    greedy_oracle = np.argmax(np.array(pi_oracle), axis=1)
    return float(np.mean(greedy_hat == greedy_oracle))


def run_seed(seed, oracle_policy, env, utility, true_params):
    from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.contrib.gail import GAILEstimator, GAILConfig
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    panel = env.generate_panel(
        n_individuals=N_EXPERT_TRAJS, n_periods=N_PERIODS, seed=seed
    )
    problem = env.problem_spec
    transitions = env.transition_matrices

    seed_results = {}

    # --- IQ-Learn ---
    print(f"  [seed={seed}] IQ-Learn...", flush=True)
    t0 = time.time()
    iq_result = IQLearnEstimator(
        config=IQLearnConfig(q_type="tabular", alpha=1.0, max_iter=500)
    ).estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
    iq_time = time.time() - t0
    iq_params = np.array(iq_result.parameters[:2])
    seed_results["iq_learn"] = {
        "params": iq_params.tolist(),
        "policy_acc": policy_accuracy(iq_result.policy, oracle_policy),
        "cos_sim": cosine_similarity(iq_params, true_params),
        "time": iq_time,
        "converged": bool(iq_result.converged),
    }
    print(
        f"    acc={seed_results['iq_learn']['policy_acc']:.3f}  "
        f"cos={seed_results['iq_learn']['cos_sim']:.3f}  "
        f"t={iq_time:.1f}s",
        flush=True,
    )

    # --- MCE-IRL ---
    print(f"  [seed={seed}] MCE-IRL...", flush=True)
    t0 = time.time()
    mce_result = MCEIRLEstimator(
        config=MCEIRLConfig(outer_max_iter=200, compute_se=False, verbose=False)
    ).estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
    mce_time = time.time() - t0
    mce_params = np.array(mce_result.parameters[:2])
    seed_results["mce_irl"] = {
        "params": mce_params.tolist(),
        "policy_acc": policy_accuracy(mce_result.policy, oracle_policy),
        "cos_sim": cosine_similarity(mce_params, true_params),
        "time": mce_time,
        "converged": bool(mce_result.converged),
    }
    print(
        f"    acc={seed_results['mce_irl']['policy_acc']:.3f}  "
        f"cos={seed_results['mce_irl']['cos_sim']:.3f}  "
        f"t={mce_time:.1f}s",
        flush=True,
    )

    # --- GAIL ---
    # GAIL requires ActionDependentReward (same feature matrix, different wrapper)
    from econirl.preferences.action_reward import ActionDependentReward
    gail_utility = ActionDependentReward(
        feature_matrix=jnp.array(utility.feature_matrix),
        parameter_names=list(utility.parameter_names),
    )
    print(f"  [seed={seed}] GAIL...", flush=True)
    t0 = time.time()
    gail_result = GAILEstimator(
        config=GAILConfig(
            discriminator_type="linear",
            max_rounds=200,
            discriminator_lr=0.05,
            discriminator_steps=5,
            compute_se=False,
            verbose=False,
        )
    ).estimate(panel=panel, utility=gail_utility, problem=problem, transitions=transitions)
    gail_time = time.time() - t0
    seed_results["gail"] = {
        "params": None,
        "policy_acc": policy_accuracy(gail_result.policy, oracle_policy),
        "cos_sim": None,
        "time": gail_time,
        "converged": bool(gail_result.converged),
    }
    print(
        f"    acc={seed_results['gail']['policy_acc']:.3f}  "
        f"t={gail_time:.1f}s",
        flush=True,
    )

    # --- BC ---
    print(f"  [seed={seed}] BC...", flush=True)
    t0 = time.time()
    bc_result = BehavioralCloningEstimator().estimate(
        panel=panel, utility=utility, problem=problem, transitions=transitions
    )
    bc_time = time.time() - t0
    seed_results["bc"] = {
        "params": None,
        "policy_acc": policy_accuracy(bc_result.policy, oracle_policy),
        "cos_sim": None,
        "time": bc_time,
        "converged": True,
    }
    print(f"    acc={seed_results['bc']['policy_acc']:.3f}  t={bc_time:.2f}s", flush=True)

    return seed_results


def aggregate(all_seed_results, methods):
    summary = {}
    for m in methods:
        accs = [r[m]["policy_acc"] for r in all_seed_results]
        times = [r[m]["time"] for r in all_seed_results]
        cos_vals = [
            r[m]["cos_sim"]
            for r in all_seed_results
            if r[m]["cos_sim"] is not None
        ]
        summary[m] = {
            "policy_acc_mean": float(np.mean(accs)),
            "policy_acc_std": float(np.std(accs)),
            "cos_sim_mean": float(np.mean(cos_vals)) if cos_vals else None,
            "cos_sim_std": float(np.std(cos_vals)) if cos_vals else None,
            "time_mean": float(np.mean(times)),
        }
    return summary


def write_tex(summary):
    def fmt_acc(m):
        mu = summary[m]["policy_acc_mean"] * 100
        sig = summary[m]["policy_acc_std"] * 100
        return f"${mu:.1f} \\pm {sig:.1f}$"

    def fmt_cos(m):
        if summary[m]["cos_sim_mean"] is None:
            return r"\multicolumn{1}{c}{---}"
        mu = summary[m]["cos_sim_mean"]
        sig = summary[m]["cos_sim_std"]
        return f"${mu:.3f} \\pm {sig:.3f}$"

    def fmt_time(m):
        t = summary[m]["time_mean"]
        if t < 1.0:
            return r"${<}1$"
        return f"${t:.0f}$"

    tex = rf"""% Auto-generated by iq_learn_run.py

\begin{{table}}[H]
\centering\footnotesize
\caption{{Low-data stability benchmark: {N_EXPERT_TRAJS} expert trajectories ({N_EXPERT_TRAJS * N_PERIODS}~obs), {len(SEEDS)}~seeds.
  Policy accuracy is the fraction of states where the greedy action matches
  oracle NFXP trained on {N_ORACLE_TRAJS:,}~trajectories ($\beta=0.95$). Reward cos-similarity compares recovered
  parameters to true $(\theta_1,\theta_2)=(0.001,3.0)$; not applicable for non-parametric methods.
  Mean~$\pm$~std across seeds.}}
\label{{tab:iq_results}}
\vspace{{-0.5em}}
\begin{{tabular*}}{{\textwidth}}{{@{{\extracolsep{{\fill}}}} l c c r}}
\toprule
Method & Policy acc.~(\%) & Reward cos-sim & Time (s) \\
\midrule
IQ-Learn & {fmt_acc("iq_learn")} & {fmt_cos("iq_learn")} & {fmt_time("iq_learn")} \\
MCE-IRL  & {fmt_acc("mce_irl")}  & {fmt_cos("mce_irl")}  & {fmt_time("mce_irl")} \\
GAIL     & {fmt_acc("gail")}     & {fmt_cos("gail")}     & {fmt_time("gail")} \\
BC       & {fmt_acc("bc")}       & {fmt_cos("bc")}       & {fmt_time("bc")} \\
\bottomrule
\end{{tabular*}}
\end{{table}}
"""
    OUT_TEX.write_text(tex.strip())
    print(f"\nResults written to {OUT_TEX}", flush=True)


def main():
    from econirl.environments.rust_bus import RustBusEnvironment
    from econirl.estimation.nfxp import NFXPEstimator
    from econirl.preferences.linear import LinearUtility

    print("=" * 60)
    print("IQ-Learn Primer: Low-data stability benchmark")
    print(f"  {N_EXPERT_TRAJS} trajectories x {N_PERIODS} periods = {N_EXPERT_TRAJS * N_PERIODS} obs")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60, flush=True)

    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=90,
        discount_factor=0.95,
    )
    utility = LinearUtility.from_environment(env)
    true_params = np.array([0.001, 3.0])

    # Oracle: NFXP on large panel
    print(f"\nComputing oracle (NFXP on {N_ORACLE_TRAJS} trajectories)...", flush=True)
    oracle_panel = env.generate_panel(
        n_individuals=N_ORACLE_TRAJS, n_periods=N_PERIODS, seed=0
    )
    t0 = time.time()
    oracle_result = NFXPEstimator().estimate(
        panel=oracle_panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    oracle_time = time.time() - t0
    oracle_policy = np.array(oracle_result.policy)
    oracle_params = np.array(oracle_result.parameters[:2])
    print(
        f"  Oracle: params={oracle_params.tolist()}, "
        f"LL={oracle_result.log_likelihood:.1f}, t={oracle_time:.1f}s",
        flush=True,
    )

    # 3-seed experiment
    all_seed_results = []
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---", flush=True)
        seed_results = run_seed(seed, oracle_policy, env, utility, true_params)
        all_seed_results.append(seed_results)

    methods = ["iq_learn", "mce_irl", "gail", "bc"]
    summary = aggregate(all_seed_results, methods)

    print("\n\nSummary:")
    print(f"{'Method':>10} {'PolicyAcc (%)':>16} {'CosSim':>14} {'Time(s)':>10}")
    print("-" * 54)
    for m in methods:
        acc_str = (
            f"{summary[m]['policy_acc_mean']*100:.1f}"
            f" ± {summary[m]['policy_acc_std']*100:.1f}"
        )
        cos_str = (
            f"{summary[m]['cos_sim_mean']:.3f} ± {summary[m]['cos_sim_std']:.3f}"
            if summary[m]["cos_sim_mean"] is not None
            else "N/A"
        )
        print(
            f"{m:>10} {acc_str:>16} {cos_str:>14} "
            f"{summary[m]['time_mean']:>10.1f}"
        )

    # Save JSON
    output = {
        "oracle": {
            "params": oracle_params.tolist(),
            "ll": float(oracle_result.log_likelihood),
            "time": oracle_time,
        },
        "seeds": {str(s): r for s, r in zip(SEEDS, all_seed_results)},
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(output, indent=2))
    print(f"\nJSON written to {OUT_JSON}", flush=True)

    write_tex(summary)


if __name__ == "__main__":
    main()
