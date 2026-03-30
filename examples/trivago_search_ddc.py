#!/usr/bin/env python3
"""Trivago Hotel Search DDC: Estimating search costs from booking sessions.

Models hotel search on Trivago as a sequential DDC. At each step, users
decide to browse (view hotel details), refine (change filters), clickout
(book), or abandon. Structural parameters reveal search costs and
the value of booking relative to the outside option.

The key question: "What is the cost of one more search step, and how
does it differ between mobile and desktop?"

Run: python examples/trivago_search_ddc.py
"""

import time
import traceback

import numpy as np
import torch

from econirl.core.types import DDCProblem, Panel
from econirl.preferences.linear import LinearUtility

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SESSIONS = 10000      # Number of sessions to load (full dataset is ~910K)
DISCOUNT = 0.95         # Search is short-horizon (typically 5-20 steps)
SIGMA = 1.0             # Logit scale parameter
N_STATES = 37           # 36 search states + 1 absorbing terminal
N_ACTIONS = 4           # browse, refine, clickout, abandon
SEED = 42

FEATURE_NAMES = ["step_cost", "browse_cost", "refine_cost", "clickout_value"]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load Trivago sessions and build MDP components."""
    from econirl.datasets.trivago_search import (
        load_trivago_sessions,
        build_trivago_mdp,
        build_trivago_panel,
        build_trivago_features,
        build_trivago_transitions,
    )

    print("Loading Trivago hotel search sessions...")
    t0 = time.time()

    sessions = load_trivago_sessions(n_sessions=N_SESSIONS)
    mdp = build_trivago_mdp(sessions)
    panel = build_trivago_panel(mdp)
    features = build_trivago_features(n_states=N_STATES, n_actions=N_ACTIONS)
    transitions = build_trivago_transitions(mdp, n_states=N_STATES, n_actions=N_ACTIONS)

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  MDP: {N_STATES} states, {N_ACTIONS} actions, {len(FEATURE_NAMES)} features")
    print(f"  Sessions: {panel.num_individuals:,} | Steps: {panel.num_observations:,}")

    return panel, features, transitions, mdp


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def split_panel(panel, train_frac=0.8):
    """Split panel into train and test by session."""
    n_train = int(panel.num_individuals * train_frac)
    train_trajs = panel.trajectories[:n_train]
    test_trajs = panel.trajectories[n_train:]
    train_panel = Panel(trajectories=train_trajs)
    test_panel = Panel(trajectories=test_trajs)
    return train_panel, test_panel


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(policy, panel, name):
    """Compute test log-likelihood per step and step accuracy.

    Parameters
    ----------
    policy : torch.Tensor of shape (n_states, n_actions)
        Choice probabilities P(a|s).
    panel : Panel
        Test data panel.
    name : str
        Estimator name (for display).

    Returns
    -------
    dict with keys: test_ll, step_acc
    """
    all_states = panel.get_all_states()
    all_actions = panel.get_all_actions()

    # Ensure float
    if policy.dtype != torch.float32:
        policy = policy.float()

    # Log-likelihood per step
    log_probs = torch.log(policy.clamp(min=1e-10))
    test_ll = log_probs[all_states, all_actions].mean().item()

    # Step accuracy: argmax prediction matches observed action
    predicted = policy.argmax(dim=1)
    correct = (predicted[all_states] == all_actions).float().mean().item()

    return {"test_ll": test_ll, "step_acc": correct}


# ---------------------------------------------------------------------------
# Estimator runners
# ---------------------------------------------------------------------------


def run_bc(train_panel, utility, problem, transitions):
    """Behavioral Cloning: empirical choice frequencies."""
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    print("\n--- Behavioral Cloning ---")
    t0 = time.time()

    estimator = BehavioralCloningEstimator(smoothing=1.0, verbose=True)
    result = estimator.estimate(train_panel, utility, problem, transitions)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Train LL: {result.log_likelihood:,.2f}")

    return result.policy, elapsed, result


def run_ccp(train_panel, utility, problem, transitions):
    """CCP / Hotz-Miller estimator."""
    from econirl.estimation.ccp import CCPEstimator

    print("\n--- CCP (Hotz-Miller) ---")
    t0 = time.time()

    estimator = CCPEstimator(
        num_policy_iterations=1,   # Hotz-Miller (one step)
        ccp_smoothing=1e-4,
        outer_max_iter=500,
        se_method="asymptotic",
        compute_hessian=False,
        verbose=True,
    )
    result = estimator.estimate(train_panel, utility, problem, transitions)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Converged: {result.converged}")
    print(f"  Train LL: {result.log_likelihood:,.2f}")
    print(f"  Parameters:")
    for i, name in enumerate(FEATURE_NAMES):
        val = result.parameters[i].item()
        print(f"    {name:<20} {val:>10.4f}")

    return result.policy, elapsed, result


def run_nfxp(train_panel, utility, problem, transitions):
    """NFXP (Rust 1987) -- full nested fixed point MLE."""
    from econirl.estimation.nfxp import NFXPEstimator

    print("\n--- NFXP (Nested Fixed Point) ---")
    t0 = time.time()

    estimator = NFXPEstimator(
        optimizer="BHHH",
        inner_solver="hybrid",
        inner_tol=1e-10,
        outer_tol=1e-6,
        outer_max_iter=500,
        se_method="asymptotic",
        compute_hessian=False,
        verbose=True,
    )
    result = estimator.estimate(train_panel, utility, problem, transitions)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Converged: {result.converged}")
    print(f"  Train LL: {result.log_likelihood:,.2f}")
    print(f"  Parameters:")
    for i, name in enumerate(FEATURE_NAMES):
        val = result.parameters[i].item()
        print(f"    {name:<20} {val:>10.4f}")

    return result.policy, elapsed, result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    panel, features, transitions, mdp = load_data()

    # Verify absorbing state transitions
    absorbing = N_STATES - 1
    for a in range(N_ACTIONS):
        assert transitions[a, absorbing, absorbing].item() > 0.99, (
            f"Absorbing state {absorbing} should self-loop for action {a}"
        )

    # Train/test split
    train_panel, test_panel = split_panel(panel, train_frac=0.8)
    print(f"\n  Train: {train_panel.num_individuals:,} sessions "
          f"({train_panel.num_observations:,} steps)")
    print(f"  Test:  {test_panel.num_individuals:,} sessions "
          f"({test_panel.num_observations:,} steps)")

    # Build utility and problem specs
    utility = LinearUtility(feature_matrix=features, parameter_names=FEATURE_NAMES)
    problem = DDCProblem(
        num_states=N_STATES,
        num_actions=N_ACTIONS,
        discount_factor=DISCOUNT,
        scale_parameter=SIGMA,
    )

    # Dictionary to collect results: name -> (policy, time, summary)
    results = {}

    def _run_and_eval(name, runner):
        """Run a single estimator and evaluate on test data."""
        try:
            policy, elapsed, summary = runner()
            eval_result = evaluate_policy(policy, test_panel, name)
            results[name] = {
                "policy": policy,
                "time": elapsed,
                "summary": summary,
                **eval_result,
            }
        except Exception as e:
            print(f"\n  *** {name} FAILED: {e}")
            traceback.print_exc()
            results[name] = {
                "policy": None,
                "time": float("nan"),
                "summary": None,
                "test_ll": float("nan"),
                "step_acc": float("nan"),
            }

    # 1. Behavioral Cloning (baseline)
    _run_and_eval("BC", lambda: run_bc(
        train_panel, utility, problem, transitions))

    # 2. CCP (fast structural)
    _run_and_eval("CCP", lambda: run_ccp(
        train_panel, utility, problem, transitions))

    # 3. NFXP (gold standard structural)
    _run_and_eval("NFXP", lambda: run_nfxp(
        train_panel, utility, problem, transitions))

    # -----------------------------------------------------------------------
    # Results table
    # -----------------------------------------------------------------------
    print()
    print("=" * 62)
    print("Trivago Hotel Search DDC Benchmark")
    print("=" * 62)
    print(f"MDP: {N_STATES} states, {N_ACTIONS} actions, "
          f"{len(FEATURE_NAMES)} features")
    print(f"Discount: {DISCOUNT}, Scale: {SIGMA}")
    print(f"Train: {train_panel.num_individuals:,} sessions | "
          f"Test: {test_panel.num_individuals:,} sessions")
    print()
    print(f"{'Estimator':<12} {'Test LL':>12} {'Step Acc':>10} "
          f"{'Time (s)':>10} {'Converged':>10}")
    print("-" * 56)

    for name in ["BC", "CCP", "NFXP"]:
        r = results.get(name, {})
        ll = r.get("test_ll", float("nan"))
        acc = r.get("step_acc", float("nan"))
        t = r.get("time", float("nan"))
        summary = r.get("summary")
        converged = (
            "yes" if (summary and summary.converged)
            else ("FAIL" if summary is None else "no")
        )
        print(f"{name:<12} {ll:>12.4f} {acc * 100:>9.1f}% {t:>10.1f} {converged:>10}")

    print("=" * 62)

    # -----------------------------------------------------------------------
    # Structural parameters side by side
    # -----------------------------------------------------------------------
    structural = ["CCP", "NFXP"]
    active = [n for n in structural
              if results.get(n, {}).get("summary") is not None]

    if active:
        print(f"\nStructural Parameters (point estimates):")
        header = f"{'Feature':<20}" + "".join(f"{n:>12}" for n in active)
        print(header)
        print("-" * (20 + 12 * len(active)))

        for i, fname in enumerate(FEATURE_NAMES):
            row = f"{fname:<20}"
            for est_name in active:
                val = results[est_name]["summary"].parameters[i].item()
                row += f"{val:>12.4f}"
            print(row)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------
    print("""
Interpretation:
- step_cost < 0: searching is costly (grows with depth)
- browse_cost < 0: each browse action has a baseline cost
- refine_cost < 0: refining search has a cost
- clickout_value > 0: booking has positive value
""")


if __name__ == "__main__":
    main()
