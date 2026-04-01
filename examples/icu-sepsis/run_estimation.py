"""ICU-Sepsis: Clinical Treatment IRL on Real Patient Data

Demonstrates inverse reinforcement learning on the ICU-Sepsis benchmark
MDP derived from MIMIC-III records. The goal is to recover the implicit
reward function driving ICU clinicians' decisions about IV fluid and
vasopressor dosing for sepsis patients.

This example generates expert demonstrations from the MIMIC-III clinician
policy, then estimates the reward function using NFXP, CCP, and MCE-IRL.
The recovered reward reveals how clinicians trade off patient severity
against treatment intensity.

Run:
    python examples/icu-sepsis/run_estimation.py
"""

import time

import econirl._jax_config  # enable float64 before any JAX ops
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.datasets.icu_sepsis import load_icu_sepsis, load_icu_sepsis_mdp
from econirl.environments.icu_sepsis import ICUSepsisEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility


def split_panel(panel: Panel, train_frac: float = 0.8) -> tuple[Panel, Panel]:
    """Split a panel into train and test sets by individual."""
    cutoff = int(panel.num_individuals * train_frac)
    train = Panel(trajectories=panel.trajectories[:cutoff])
    test = Panel(trajectories=panel.trajectories[cutoff:])
    return train, test


def evaluate_policy(result, panel, utility, problem, transitions):
    """Score estimated policy against observed choices."""
    operator = SoftBellmanOperator(problem, transitions)
    flow_u = utility.compute(result.parameters)
    sol = value_iteration(operator, flow_u, tol=1e-12, max_iter=100_000)
    log_probs = operator.compute_log_choice_probabilities(flow_u, sol.V)
    states = panel.get_all_states()
    actions = panel.get_all_actions()
    ll = float(log_probs[states, actions].sum())
    predicted = sol.policy[states].argmax(axis=1)
    acc = float((predicted == actions).astype(jnp.float32).mean())
    return ll, acc


def main():
    print("=" * 60)
    print("ICU-Sepsis: Clinical Treatment IRL")
    print("=" * 60)

    # ── 1. Load environment and generate data ──────────────────────────

    env = ICUSepsisEnvironment(discount_factor=0.99)
    print(env.describe())

    print("Generating expert demonstrations from clinician policy...")
    panel = load_icu_sepsis(n_individuals=2000, max_steps=20, as_panel=True, seed=42)
    train, test = split_panel(panel, train_frac=0.8)
    print(f"  Train: {train.num_individuals} patients, {train.num_observations} observations")
    print(f"  Test:  {test.num_individuals} patients, {test.num_observations} observations")

    # ── 2. Set up estimation ───────────────────────────────────────────

    transitions = env.transition_matrices
    features = env.feature_matrix
    problem = env.problem_spec

    utility = LinearUtility(
        features=features,
        parameter_names=env.parameter_names,
    )

    init_params = jnp.zeros(len(env.parameter_names))

    # ── 3. Estimate with three methods ─────────────────────────────────

    results = {}

    # NFXP
    print("\n── NFXP ──")
    t0 = time.time()
    nfxp = NFXPEstimator(problem, transitions)
    nfxp_result = nfxp.fit(train, utility, init_params=init_params)
    dt = time.time() - t0
    results["NFXP"] = nfxp_result
    print(f"  Time: {dt:.1f}s")
    for name, val in zip(env.parameter_names, nfxp_result.parameters):
        print(f"  {name}: {float(val):.6f}")

    # CCP
    print("\n── CCP ──")
    t0 = time.time()
    ccp = CCPEstimator(problem, transitions)
    ccp_result = ccp.fit(train, utility, init_params=init_params)
    dt = time.time() - t0
    results["CCP"] = ccp_result
    print(f"  Time: {dt:.1f}s")
    for name, val in zip(env.parameter_names, ccp_result.parameters):
        print(f"  {name}: {float(val):.6f}")

    # MCE-IRL
    print("\n── MCE-IRL ──")
    t0 = time.time()
    mce = MCEIRLEstimator(problem, transitions)
    mce_result = mce.fit(train, utility, init_params=init_params)
    dt = time.time() - t0
    results["MCE-IRL"] = mce_result
    print(f"  Time: {dt:.1f}s")
    for name, val in zip(env.parameter_names, mce_result.parameters):
        print(f"  {name}: {float(val):.6f}")

    # ── 4. Out-of-sample evaluation ────────────────────────────────────

    print("\n── Out-of-Sample Evaluation ──")
    print(f"{'Estimator':<12} {'Train LL':>10} {'Test LL':>10} {'Test Acc':>10}")
    print("-" * 44)
    for name, result in results.items():
        train_ll, train_acc = evaluate_policy(
            result, train, utility, problem, transitions
        )
        test_ll, test_acc = evaluate_policy(
            result, test, utility, problem, transitions
        )
        print(f"{name:<12} {train_ll:>10.1f} {test_ll:>10.1f} {test_acc:>10.3f}")

    # ── 5. Clinical interpretation ─────────────────────────────────────

    print("\n── Clinical Interpretation ──")
    print("Parameter signs reveal the implicit reward structure:")
    best = results["NFXP"]
    for name, val in zip(env.parameter_names, best.parameters):
        val_f = float(val)
        if "sofa" in name:
            direction = "sicker patients have lower utility" if val_f < 0 else "unexpected positive"
        elif "fluid" in name:
            direction = "higher fluid doses are costly" if val_f < 0 else "higher fluid doses are preferred"
        elif "vaso" in name:
            direction = "higher vaso doses are costly" if val_f < 0 else "higher vaso doses are preferred"
        else:
            direction = ""
        print(f"  {name} = {val_f:.4f}  ({direction})")


if __name__ == "__main__":
    main()
