"""ICU-Sepsis: Clinical Treatment IRL on Real Patient Data

Demonstrates inverse reinforcement learning on the ICU-Sepsis benchmark
MDP derived from MIMIC-III records. The goal is to recover the implicit
reward function driving ICU clinicians' decisions about IV fluid and
vasopressor dosing for sepsis patients.

This example generates expert demonstrations from the MIMIC-III clinician
policy, then estimates the reward function using NFXP and CCP. The
recovered reward reveals how clinicians trade off patient severity
against treatment intensity. Counterfactual analysis shows what happens
when vasopressor costs are doubled.

Run:
    python examples/icu-sepsis/run_estimation.py
"""

import time

import econirl._jax_config  # enable float64 before any JAX ops
import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel
from econirl.datasets.icu_sepsis import load_icu_sepsis
from econirl.environments.icu_sepsis import ICUSepsisEnvironment
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
    print("ICU-Sepsis: Clinical Treatment IRL (716 states, 25 actions)")
    print("=" * 65)

    env = ICUSepsisEnvironment(discount_factor=0.99)

    print("Generating expert demonstrations from clinician policy...")
    panel = load_icu_sepsis(n_individuals=2000, max_steps=20, as_panel=True, seed=42)
    cutoff = int(panel.num_individuals * 0.8)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    print(f"  Train: {train.num_individuals} patients, {train.num_observations} obs")
    print(f"  Test:  {test.num_individuals} patients, {test.num_observations} obs")

    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    # ── Estimation ─────────────────────────────────────────────────────

    results = {}
    for name, EstCls in [("NFXP", NFXPEstimator), ("CCP", CCPEstimator)]:
        t0 = time.time()
        est = EstCls()
        r = est.estimate(train, utility, problem, transitions)
        dt = time.time() - t0
        results[name] = r
        print(f"\n{name}: {dt:.1f}s, converged={r.converged}")
        for pname, val in zip(env.parameter_names, r.parameters):
            print(f"  {pname}: {float(val):.4f}")

    # ── Standard errors ────────────────────────────────────────────────

    print("\nStandard errors (NFXP):")
    se = results["NFXP"].standard_errors
    if se is not None:
        for pname, val in zip(env.parameter_names, se):
            print(f"  {pname}: {float(val):.4f}")

    # ── Counterfactual: Double vasopressor cost ────────────────────────

    print("\n" + "=" * 65)
    print("Counterfactual: Double vasopressor cost")
    print("=" * 65)

    best = results["NFXP"]
    new_params = best.parameters.at[2].set(best.parameters[2] * 2)
    cf = counterfactual_policy(best, new_params, utility, problem, transitions)
    print(f"Welfare change: {float(cf.welfare_change):+.4f}")

    baseline_mean = np.array(cf.baseline_policy[:713]).mean(axis=0)
    counter_mean = np.array(cf.counterfactual_policy[:713]).mean(axis=0)

    print(f"\n{'Vaso Level':>12} {'Baseline':>10} {'Counter':>10} {'Change':>10}")
    print("-" * 44)
    for vl in range(5):
        bp = sum(baseline_mean[fl * 5 + vl] for fl in range(5))
        cp = sum(counter_mean[fl * 5 + vl] for fl in range(5))
        print(f"{'Level ' + str(vl):>12} {bp:>10.3f} {cp:>10.3f} {cp - bp:>+10.3f}")

    # ── Elasticity ─────────────────────────────────────────────────────

    print("\n" + "=" * 65)
    print("Elasticity: SOFA weight sensitivity")
    print("=" * 65)
    ea = elasticity_analysis(
        best, utility, problem, transitions,
        parameter_name="sofa_weight",
        pct_changes=[-0.50, -0.25, 0.25, 0.50],
    )
    print(f"Baseline sofa_weight: {float(ea['baseline_value']):.4f}")
    print(f"{'% Change':>10} {'Welfare Δ':>12} {'Avg Policy Δ':>14}")
    print("-" * 38)
    for i, pct in enumerate(ea["pct_changes"]):
        wc = ea["welfare_changes"][i]
        pc = ea["policy_changes"][i]
        print(f"{pct:>+10.0%} {float(wc):>12.4f} {float(pc):>14.4f}")


if __name__ == "__main__":
    main()
