"""Dixit Entry/Exit: Structural estimation with post-estimation diagnostics.

Demonstrates NFXP, CCP, and MCE-IRL on a firm entry/exit DDC problem
based on the Dixit (1989) model. Sunk entry and exit costs create
hysteresis in firm behavior. Known ground truth enables parameter
recovery evaluation.

Usage:
    python examples/entry-exit/run_estimation.py
"""

import time

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.environments.entry_exit import EntryExitEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.inference import etable
from econirl.inference.fit_metrics import brier_score, kl_divergence
from econirl.inference.hypothesis_tests import vuong_test
from econirl.preferences.linear import LinearUtility
from econirl.simulation.counterfactual import counterfactual_policy, elasticity_analysis


def main():
    print("=" * 65)
    print("Dixit Entry/Exit (20 states, 2 actions)")
    print("=" * 65)

    env = EntryExitEnvironment(discount_factor=0.95)
    panel = env.generate_panel(n_individuals=2000, n_periods=100, seed=42)

    cutoff = int(panel.num_individuals * 0.8)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    print(f"  Train: {train.num_individuals} firms, {train.num_observations} obs")
    print(f"  Test:  {test.num_individuals} firms, {test.num_observations} obs")
    print(f"  True: {env.true_parameters}")

    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    # Estimation
    results = {}

    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust")
    results["NFXP"] = nfxp.estimate(train, utility, problem, transitions)
    print(f"\nNFXP: {time.time() - t0:.1f}s")
    print(results["NFXP"].summary())

    t0 = time.time()
    ccp = CCPEstimator(num_policy_iterations=20, se_method="robust")
    results["CCP"] = ccp.estimate(train, utility, problem, transitions)
    print(f"\nCCP: {time.time() - t0:.1f}s")
    print(results["CCP"].summary())

    t0 = time.time()
    mce_config = MCEIRLConfig(learning_rate=0.1, outer_max_iter=300)
    mce = MCEIRLEstimator(config=mce_config)
    results["MCE-IRL"] = mce.estimate(train, utility, problem, transitions)
    print(f"\nMCE-IRL: {time.time() - t0:.1f}s")
    print(results["MCE-IRL"].summary())

    # Parameter recovery table
    print("\n" + "=" * 65)
    print("Parameter Recovery Table")
    print("=" * 65)
    print(f"{'Parameter':<18} {'True':>8} {'NFXP':>8} {'CCP':>8} {'MCE-IRL':>8}")
    print("-" * 58)
    for i, name in enumerate(env.parameter_names):
        true_val = env.true_parameters[name]
        print(f"{name:<18} {true_val:>8.4f} "
              f"{float(results['NFXP'].parameters[i]):>8.4f} "
              f"{float(results['CCP'].parameters[i]):>8.4f} "
              f"{float(results['MCE-IRL'].parameters[i]):>8.4f}")

    # Post-estimation diagnostics
    print("\n" + "=" * 65)
    print("Post-Estimation Diagnostics")
    print("=" * 65)

    print("\n--- etable() ---")
    print(etable(results["NFXP"], results["CCP"], results["MCE-IRL"]))

    obs_states = jnp.array(train.get_all_states())
    obs_actions = jnp.array(train.get_all_actions())

    print("\n--- Brier Scores ---")
    for name, r in results.items():
        bs = brier_score(r.policy, obs_states, obs_actions)
        print(f"  {name}: {bs['brier_score']:.4f}")

    print("\n--- Vuong Test (NFXP vs MCE-IRL) ---")
    vt = vuong_test(results["NFXP"].policy, results["MCE-IRL"].policy, obs_states, obs_actions)
    print(f"  Z-statistic: {vt['statistic']:.3f}")
    print(f"  P-value: {vt['p_value']:.4f}")
    print(f"  Direction: {vt['direction']}")

    sufficient = train.sufficient_stats(env.num_states, env.num_actions)
    data_ccps = jnp.array(sufficient.empirical_ccps)
    print("\n--- KL Divergence ---")
    for name, r in results.items():
        kl = kl_divergence(data_ccps, r.policy)
        print(f"  {name}: {kl['kl_divergence']:.6f}")

    # Counterfactual: eliminate entry cost
    print("\n" + "=" * 65)
    print("Counterfactual: Free Entry (entry_cost = 0)")
    print("=" * 65)
    best = results["NFXP"]
    entry_cost_idx = env.parameter_names.index("entry_cost")
    new_params = best.parameters.at[entry_cost_idx].set(0.0)
    cf = counterfactual_policy(best, new_params, utility, problem, transitions)
    print(f"Welfare change: {float(cf.welfare_change):+.3f}")

    # Elasticity on entry cost
    print("\n--- Entry Cost Elasticity ---")
    ea = elasticity_analysis(
        best, utility, problem, transitions,
        parameter_name="entry_cost",
        pct_changes=[-1.0, -0.50, -0.25, 0.25, 0.50, 1.0],
    )
    print(f"{'% Change':>10} {'Welfare':>12} {'Avg Policy':>12}")
    print("-" * 36)
    for i, pct in enumerate(ea["pct_changes"]):
        wc = ea["welfare_changes"][i]
        pc = ea["policy_changes"][i]
        print(f"{pct:>+10.0%} {float(wc):>+12.3f} {float(pc):>12.3f}")


if __name__ == "__main__":
    main()
