#!/usr/bin/env python3
"""Shanghai Route-Choice Benchmark: 4 estimators on real taxi route data.

Benchmarks BC, CCP, NNES, and TD-CCP on the Shanghai taxi route-choice
dataset (Zhao & Liang 2022). This demonstrates econirl on a real-world
IRL/DDC problem with 714 states and 8 actions.

Network: 714 road-segment states, 8 compass-direction actions, 7 features
    (normalized length + 6 road-type one-hots).
Features are action-dependent: phi(s, a) = edge_features[next_state(s, a)].

Reference:
    Zhao, Z., & Liang, Y. (2022). Deep Inverse Reinforcement Learning
    for Route Choice Modeling. arXiv:2206.10598.

Run:
    python examples/shanghai_route_choice.py
"""

import time
import traceback

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_SIZE = 1000       # Number of training routes (100, 1000, or 10000)
CV_FOLD = 0             # Cross-validation fold (0-4)
DISCOUNT = 0.95         # Route choice is finite-horizon; lower discount is fine
SIGMA = 1.0             # Logit scale parameter
SEED = 42

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data():
    """Load Shanghai network and trajectory data, build MDP components."""
    from econirl.datasets.shanghai_route import (
        load_shanghai_network,
        load_shanghai_trajectories,
        parse_trajectories_to_panel,
        build_transition_matrix,
        build_edge_features,
        build_state_action_features,
    )

    print("Loading Shanghai network and trajectories...")
    t0 = time.time()

    network = load_shanghai_network()
    n_states = network["n_states"]
    n_actions = network["n_actions"]
    transit = network["transit"]

    train_df = load_shanghai_trajectories(
        split="train", cv=CV_FOLD, size=TRAIN_SIZE,
    )
    test_df = load_shanghai_trajectories(split="test", cv=CV_FOLD)

    # Build MDP components
    transitions = build_transition_matrix(transit, n_states, n_actions)
    edge_features = build_edge_features(network["edges"], n_states)
    sa_features = build_state_action_features(
        edge_features, transit, n_states, n_actions,
    )

    # Parse trajectories into panels
    train_panel = parse_trajectories_to_panel(train_df, transit, n_states, n_actions)
    test_panel = parse_trajectories_to_panel(test_df, transit, n_states, n_actions)

    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")
    print(f"  Network: {n_states} states, {n_actions} actions, 7 features")
    print(f"  Transitions: {len(transit)} edges, {transitions.shape}")
    print(f"  Train: {train_panel.num_individuals:,} routes, "
          f"{train_panel.num_observations:,} steps")
    print(f"  Test:  {test_panel.num_individuals:,} routes, "
          f"{test_panel.num_observations:,} steps")

    return (
        n_states, n_actions, transitions, sa_features,
        train_panel, test_panel, network,
    )


# ---------------------------------------------------------------------------
# Utility specification
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "length", "residential", "primary", "secondary",
    "tertiary", "living_street", "unclassified",
]


def build_utility(sa_features):
    """Build LinearUtility from state-action features (714, 8, 7)."""
    from econirl.preferences.linear import LinearUtility

    utility = LinearUtility(
        feature_matrix=sa_features,
        parameter_names=FEATURE_NAMES,
    )
    return utility


# ---------------------------------------------------------------------------
# Problem specification
# ---------------------------------------------------------------------------


def build_problem(n_states, n_actions, state_dim=None, state_encoder=None):
    """Build DDCProblem for Shanghai route choice."""
    from econirl.core.types import DDCProblem

    return DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=DISCOUNT,
        scale_parameter=SIGMA,
        state_dim=state_dim,
        state_encoder=state_encoder,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_policy(policy, panel, n_states, n_actions, name):
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


def run_bc(train_panel, utility, problem, transitions, n_states, n_actions):
    """Behavioral Cloning: empirical choice frequencies."""
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator

    print("\n--- Behavioral Cloning ---")
    t0 = time.time()

    estimator = BehavioralCloningEstimator(smoothing=1.0, verbose=True)
    result = estimator.estimate(train_panel, utility, problem, transitions)

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Train LL: {result.log_likelihood:,.2f}")
    print(f"  Train accuracy: {result.goodness_of_fit.prediction_accuracy:.4f}")

    return result.policy, elapsed, result


def run_ccp(train_panel, utility, problem, transitions, n_states, n_actions):
    """CCP / Hotz-Miller estimator."""
    from econirl.estimation.ccp import CCPEstimator

    print("\n--- CCP (Hotz-Miller) ---")
    t0 = time.time()

    estimator = CCPEstimator(
        num_policy_iterations=1,   # Hotz-Miller (one step)
        ccp_smoothing=1e-4,
        outer_max_iter=500,
        se_method="asymptotic",
        compute_hessian=False,     # Skip -- sparse data gives singular Hessian
        verbose=True,
    )
    result = estimator.estimate(
        train_panel, utility, problem, transitions,
    )

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Converged: {result.converged}")
    print(f"  Train LL: {result.log_likelihood:,.2f}")
    print(f"  Parameters:")
    for i, name in enumerate(FEATURE_NAMES):
        val = result.parameters[i].item()
        print(f"    {name:<20} {val:>10.4f}")

    return result.policy, elapsed, result


def run_nnes(train_panel, utility, problem_neural, transitions,
             n_states, n_actions, initial_params=None):
    """NNES: Neural Network Estimation of Structural models."""
    from econirl.estimation.nnes import NNESEstimator

    print("\n--- NNES (Nguyen 2025) ---")
    if initial_params is not None:
        print(f"  Warm-starting from CCP parameters")
    t0 = time.time()

    estimator = NNESEstimator(
        hidden_dim=64,
        num_layers=2,
        v_lr=3e-4,             # Lower LR for stability on large state space
        v_epochs=500,
        v_batch_size=4096,
        outer_max_iter=200,
        outer_tol=1e-6,
        n_outer_iterations=2,  # 2 iterations: bootstrap V then refine theta
        compute_se=False,      # Skip SE -- sparse data gives singular Hessian
        verbose=True,
    )
    result = estimator.estimate(
        train_panel, utility, problem_neural, transitions,
        initial_params=initial_params,
    )

    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")
    print(f"  Converged: {result.converged}")
    print(f"  Train LL: {result.log_likelihood:,.2f}")
    print(f"  Parameters:")
    for i, name in enumerate(FEATURE_NAMES):
        val = result.parameters[i].item()
        print(f"    {name:<20} {val:>10.4f}")

    return result.policy, elapsed, result


def run_tdccp(train_panel, utility, problem_neural, transitions,
              n_states, n_actions, initial_params=None):
    """TD-CCP: Temporal-Difference CCP with neural EV approximation."""
    from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig

    print("\n--- TD-CCP Neural ---")
    if initial_params is not None:
        print(f"  Warm-starting from CCP parameters")
    t0 = time.time()

    config = TDCCPConfig(
        hidden_dim=64,
        num_hidden_layers=2,
        avi_iterations=15,
        epochs_per_avi=20,
        learning_rate=1e-3,
        batch_size=4096,
        ccp_smoothing=0.01,
        outer_max_iter=200,
        outer_tol=1e-6,
        n_policy_iterations=2,
        compute_se=False,      # Skip SE -- sparse data gives singular Hessian
        verbose=True,
    )
    estimator = TDCCPEstimator(config=config)
    result = estimator.estimate(
        train_panel, utility, problem_neural, transitions,
        initial_params=initial_params,
    )

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
    (n_states, n_actions, transitions, sa_features,
     train_panel, test_panel, network) = load_data()

    # Build utility and problem specs
    utility = build_utility(sa_features)
    problem = build_problem(n_states, n_actions)

    # Neural estimators need a state_encoder
    # Use normalized state index as a simple 1-D feature
    def state_encoder(s):
        return (s.float() / (n_states - 1)).unsqueeze(-1)

    problem_neural = build_problem(
        n_states, n_actions,
        state_dim=1,
        state_encoder=state_encoder,
    )

    # Dictionary to collect results: name -> (policy, time, summary)
    results = {}

    def _run_and_eval(name, runner):
        """Run a single estimator and evaluate on test data."""
        try:
            policy, elapsed, summary = runner()
            eval_result = evaluate_policy(
                policy, test_panel, n_states, n_actions, name,
            )
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
        train_panel, utility, problem, transitions, n_states, n_actions))

    # 2. CCP (structural baseline)
    _run_and_eval("CCP", lambda: run_ccp(
        train_panel, utility, problem, transitions, n_states, n_actions))

    # Use CCP parameters as warm-start for TD-CCP.
    # NNES uses its own CCP-bootstrap for initial params because its
    # V-network training is more stable starting from near-zero rewards.
    ccp_init = None
    if results.get("CCP", {}).get("summary") is not None:
        ccp_init = results["CCP"]["summary"].parameters.clone()

    # 3. NNES (uses internal CCP bootstrap for init)
    _run_and_eval("NNES", lambda: run_nnes(
        train_panel, utility, problem_neural, transitions,
        n_states, n_actions, initial_params=None))

    # 4. TD-CCP (warm-started from CCP)
    _run_and_eval("TD-CCP", lambda: run_tdccp(
        train_panel, utility, problem_neural, transitions,
        n_states, n_actions, initial_params=ccp_init))

    # Print summary table
    print()
    print("=" * 74)
    print("Shanghai Route-Choice Benchmark (Zhao & Liang 2022)")
    print("=" * 74)
    print(f"Network: {n_states} states, {n_actions} actions, "
          f"{len(FEATURE_NAMES)} features")
    print(f"Discount: {DISCOUNT}, Scale: {SIGMA}")
    print(f"Train: {train_panel.num_individuals:,} routes "
          f"({train_panel.num_observations:,} steps) | "
          f"Test: {test_panel.num_individuals:,} routes "
          f"({test_panel.num_observations:,} steps)")
    print()
    print(f"{'Estimator':<12} {'Test LL/step':>12} {'Step Acc':>10} "
          f"{'Time (s)':>10} {'Converged':>10}")
    print("-" * 56)

    for name in ["BC", "CCP", "NNES", "TD-CCP"]:
        r = results.get(name, {})
        ll = r.get("test_ll", float("nan"))
        acc = r.get("step_acc", float("nan"))
        t = r.get("time", float("nan"))
        summary = r.get("summary")
        converged = "yes" if (summary and summary.converged) else ("FAIL" if summary is None else "no")
        print(f"{name:<12} {ll:>12.4f} {acc:>10.4f} {t:>10.1f} {converged:>10}")

    print("=" * 74)

    # Print structural parameters side by side for structural estimators
    structural = ["CCP", "NNES", "TD-CCP"]
    active = [n for n in structural if results.get(n, {}).get("summary") is not None]

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

    print()


if __name__ == "__main__":
    main()
