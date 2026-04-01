#!/usr/bin/env python3
"""
Full Estimator Benchmark on Rust Bus Engine (Simulated Data)
=============================================================

Runs all compatible estimators on simulated Rust (1987) bus data with
known true parameters, measuring parameter recovery, log-likelihood,
cosine similarity, and policy accuracy.

True parameters: operating_cost=0.001, replacement_cost=3.0
Environment: 90 mileage bins, 2 actions, gamma=0.9999

Usage:
    python examples/rust-bus-engine/benchmark_all_estimators.py
"""

import csv
import os
import time
import traceback

import numpy as np
import jax.numpy as jnp

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel

# ── Configuration ──
N_INDIVIDUALS = 200
N_PERIODS = 100
SEED = 42
TIMEOUT = 300  # 5 min per estimator
CSV_PATH = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")

CSV_COLUMNS = [
    "estimator", "time_seconds", "converged",
    "operating_cost", "replacement_cost",
    "log_likelihood", "cosine_similarity", "max_policy_diff",
    "status",
]


def cosine_sim(a, b):
    a = jnp.asarray(a, dtype=jnp.float64).flatten()
    b = jnp.asarray(b, dtype=jnp.float64).flatten()
    dot = jnp.dot(a, b)
    norm = jnp.linalg.norm(a) * jnp.linalg.norm(b)
    if norm < 1e-15:
        return float("nan")
    return float(dot / norm)


def setup():
    """Create environment, simulate data, return all needed objects."""
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0,
        num_mileage_bins=90, discount_factor=0.9999,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.array([0.001, 3.0])

    panel = simulate_panel(env, n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)
    n_obs = sum(len(t.states) for t in panel.trajectories)

    # Compute true policy for comparison
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import policy_iteration
    operator = SoftBellmanOperator(problem, jnp.asarray(transitions, dtype=jnp.float64))
    true_reward = jnp.asarray(utility.compute(jnp.asarray(true_params, dtype=jnp.float32)), dtype=jnp.float64)
    true_result = policy_iteration(operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix")
    true_policy = true_result.policy

    print(f"Environment: 90 states, 2 actions, gamma=0.9999")
    print(f"True params: operating_cost=0.001, replacement_cost=3.0")
    print(f"Simulated: {N_INDIVIDUALS} individuals x {N_PERIODS} periods = {n_obs:,} observations")

    return env, utility, problem, transitions, panel, true_params, true_policy


def build_estimators():
    """Build all estimators with appropriate configs for gamma=0.9999."""
    estimators = {}

    # ── 1. NFXP-NK (gold standard) ──
    from econirl.estimation.nfxp import NFXPEstimator
    estimators["NFXP-NK"] = NFXPEstimator(
        optimizer="BHHH", inner_solver="policy",
        inner_tol=1e-12, inner_max_iter=200,
        compute_hessian=True,
        outer_tol=1e-3, verbose=True,
    )

    # ── 2. CCP-NPL ──
    from econirl.estimation.ccp import CCPEstimator
    estimators["CCP-NPL"] = CCPEstimator(
        num_policy_iterations=20, compute_hessian=True, verbose=True,
    )

    # ── 3. MCE-IRL ──
    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    estimators["MCE-IRL"] = MCEIRLEstimator(config=MCEIRLConfig(
        optimizer="L-BFGS-B", inner_solver="policy",
        inner_max_iter=200, inner_tol=1e-6,
        outer_max_iter=500, outer_tol=1e-6,
        compute_se=False, verbose=True,
    ))

    # ── 4. MaxEnt IRL ──
    from econirl.contrib.maxent_irl import MaxEntIRLEstimator
    estimators["MaxEnt IRL"] = MaxEntIRLEstimator(
        optimizer="L-BFGS-B", inner_solver="hybrid",
        inner_tol=1e-8, inner_max_iter=100000,
        outer_tol=1e-6, outer_max_iter=500, verbose=True,
    )

    # ── 5. AIRL ──
    from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
    estimators["AIRL"] = AIRLEstimator(config=AIRLConfig(
        reward_type="linear", reward_lr=0.01,
        max_rounds=200, generator_solver="hybrid",
        generator_max_iter=100000, verbose=True,
    ))

    # ── 6. GAIL ──
    from econirl.contrib.gail import GAILEstimator, GAILConfig
    estimators["GAIL"] = GAILEstimator(config=GAILConfig(
        discriminator_type="linear", discriminator_lr=0.01,
        max_rounds=50, generator_solver="hybrid",
        generator_max_iter=100000, verbose=True,
    ))

    # ── 7. IQ-Learn ──
    from econirl.contrib.iq_learn import IQLearnEstimator, IQLearnConfig
    estimators["IQ-Learn"] = IQLearnEstimator(config=IQLearnConfig(
        q_type="linear", divergence="chi2",
        optimizer="L-BFGS-B", max_iter=500, verbose=True,
    ))

    # ── 8. GLADIUS ──
    from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
    estimators["GLADIUS"] = GLADIUSEstimator(config=GLADIUSConfig(
        q_hidden_dim=64, v_hidden_dim=64,
        max_epochs=200, verbose=True,
    ))

    # ── 9. TD-CCP ──
    from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
    estimators["TD-CCP"] = TDCCPEstimator(config=TDCCPConfig(
        hidden_dim=64, avi_iterations=20,
        epochs_per_avi=100, n_policy_iterations=3,
        verbose=True,
    ))

    # ── 10. NNES ──
    from econirl.estimation.nnes import NNESEstimator
    estimators["NNES"] = NNESEstimator(
        hidden_dim=64, num_layers=2,
        v_epochs=200, outer_max_iter=100,
        n_outer_iterations=3, verbose=True,
    )

    # ── 11. SEES ──
    from econirl.estimation.sees import SEESEstimator
    estimators["SEES"] = SEESEstimator(
        basis_type="fourier", basis_dim=20,
        max_iter=200, verbose=True,
    )

    # ── 12. Max Margin Planning ──
    from econirl.contrib.max_margin_planning import MaxMarginPlanningEstimator, MMPConfig
    estimators["Max Margin"] = MaxMarginPlanningEstimator(config=MMPConfig(
        max_iterations=500, verbose=True,
    ))

    # ── 13. Bayesian IRL ──
    from econirl.contrib.bayesian_irl import BayesianIRLEstimator
    estimators["Bayesian IRL"] = BayesianIRLEstimator(
        n_samples=200, burnin=50,
        inner_max_iter=100000, verbose=True,
    )

    # ── 14. Behavioral Cloning (baseline) ──
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    estimators["BC (baseline)"] = BehavioralCloningEstimator(verbose=True)

    return estimators


def run_one(name, estimator, panel, utility, problem, transitions,
            true_params, true_policy):
    """Run one estimator and return a results dict."""
    print(f"\n{'=' * 60}")
    print(f"  Running: {name}")
    print(f"{'=' * 60}")

    row = {c: "" for c in CSV_COLUMNS}
    row["estimator"] = name

    # Some estimators need ActionDependentReward
    est_utility = utility
    needs_adr = name in ("AIRL", "GAIL", "MaxEnt IRL", "IQ-Learn", "Max Margin")
    if needs_adr:
        est_utility = ActionDependentReward(
            feature_matrix=utility.feature_matrix,
            parameter_names=utility.parameter_names,
        )

    t0 = time.time()
    try:
        result = estimator.estimate(
            panel=panel, utility=est_utility,
            problem=problem, transitions=transitions,
        )
        elapsed = time.time() - t0
        row["time_seconds"] = f"{elapsed:.1f}"
        row["converged"] = str(result.converged)
        row["status"] = "OK"

        if result.log_likelihood is not None:
            row["log_likelihood"] = f"{result.log_likelihood:.2f}"

        if result.parameters is not None and len(result.parameters) == 2:
            row["operating_cost"] = f"{result.parameters[0].item():.6f}"
            row["replacement_cost"] = f"{result.parameters[1].item():.6f}"
            row["cosine_similarity"] = f"{cosine_sim(result.parameters, true_params):.6f}"

        if result.policy is not None and true_policy is not None:
            try:
                pol = jnp.asarray(result.policy, dtype=jnp.float32)
                tp = jnp.asarray(true_policy, dtype=jnp.float32)
                if pol.shape == tp.shape:
                    row["max_policy_diff"] = f"{float(jnp.abs(pol - tp).max()):.6f}"
            except Exception:
                pass

        # Print summary
        print(f"\n  [OK] {name}: {elapsed:.1f}s, converged={result.converged}")
        if result.log_likelihood is not None:
            print(f"  LL: {result.log_likelihood:.2f}")
        if result.parameters is not None and len(result.parameters) == 2:
            print(f"  operating_cost: {result.parameters[0].item():.6f}")
            print(f"  replacement_cost: {result.parameters[1].item():.6f}")
            print(f"  cosine(θ, θ_true): {row.get('cosine_similarity', 'N/A')}")

    except Exception as e:
        elapsed = time.time() - t0
        row["time_seconds"] = f"{elapsed:.1f}"
        row["status"] = f"FAIL: {e}"
        print(f"\n  [FAIL] {name}: {elapsed:.1f}s — {e}")
        traceback.print_exc()

    return row


def print_summary(results):
    """Print final summary table."""
    print(f"\n{'=' * 90}")
    print(f"  BENCHMARK SUMMARY — Rust Bus Engine (simulated, N={N_INDIVIDUALS}x{N_PERIODS})")
    print(f"{'=' * 90}")
    print(f"  True: operating_cost=0.001000, replacement_cost=3.000000")
    print(f"\n  {'Estimator':<18} {'Time':>6} {'Conv':>5} {'LL':>10} {'op_cost':>10} {'RC':>10} {'cos(θ)':>8} {'Status'}")
    print("  " + "-" * 85)

    for r in results:
        name = r["estimator"]
        t = r.get("time_seconds", "")
        conv = r.get("converged", "")[:5]
        ll = r.get("log_likelihood", "")
        oc = r.get("operating_cost", "")
        rc = r.get("replacement_cost", "")
        cs = r.get("cosine_similarity", "")
        st = r.get("status", "")
        if st.startswith("FAIL"):
            st = "FAIL"
        print(f"  {name:<18} {t:>6} {conv:>5} {ll:>10} {oc:>10} {rc:>10} {cs:>8} {st}")


def main():
    print("=" * 60)
    print("  Rust Bus Engine — Full Estimator Benchmark")
    print("=" * 60)

    env, utility, problem, transitions, panel, true_params, true_policy = setup()
    estimators = build_estimators()

    # Check which estimators already completed
    completed = set()
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            for row in csv.DictReader(f):
                if row.get("status") in ("OK", "FAIL") or row.get("status", "").startswith("FAIL"):
                    completed.add(row["estimator"])
        print(f"Already completed: {completed or 'none'}")
    else:
        with open(CSV_PATH, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    results = []
    for name, estimator in estimators.items():
        if name in completed:
            print(f"\n  SKIP: {name} (already done)")
            continue

        row = run_one(name, estimator, panel, utility, problem, transitions,
                      true_params, true_policy)
        results.append(row)

        # Append to CSV after each estimator
        with open(CSV_PATH, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

    # Load all results for summary
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH) as f:
            results = list(csv.DictReader(f))

    print_summary(results)
    print(f"\n  Results saved to: {CSV_PATH}")


if __name__ == "__main__":
    main()
