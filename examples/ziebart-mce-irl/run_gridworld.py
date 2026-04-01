"""
Gridworld Benchmark: 3 Reward Cases × In/Out/Transfer
======================================================

Tests MCE IRL, MaxEnt IRL, and NFXP on three reward specifications
with deterministic dynamics, high N, and full evaluation.

Case 1 — State-action rewards:  R(s,a) = θ^T φ(s,a), features vary by action.
Case 2 — Rust-style rewards:    R(s,keep)=-θ_c·dist, R(s,move)=-RC. State
         features mapped differently per action (like Rust bus).
Case 3 — Pure state-only:       R(s) = θ^T φ(s), same for all actions.
         Policy driven by transition-mediated value differences only.

Usage:
    python run_gridworld.py
    python run_gridworld.py --grid-size 5 --n-traj 2000
"""

import argparse
import time

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.contrib.maxent_irl import MaxEntIRLEstimator
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel_from_policy


# =====================================================================
# Environment builders for each reward case
# =====================================================================

def _build_transitions(grid_size):
    """Deterministic 5-action gridworld transitions."""
    n_states = grid_size * grid_size
    transitions = np.zeros((5, n_states, n_states))
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]  # N,S,E,W,Stay
    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        for a, (dr, dc) in enumerate(deltas):
            nr, nc = r + dr, c + dc
            ns = (nr * grid_size + nc) if (0 <= nr < grid_size and 0 <= nc < grid_size) else s
            transitions[a, s, ns] = 1.0
    return jnp.array(transitions)


def build_case1_state_action(grid_size, discount):
    """Case 1: Full state-action features. Best identified."""
    n_states = grid_size * grid_size
    transitions = _build_transitions(grid_size)
    goal_r, goal_c = grid_size - 1, grid_size - 1
    deltas = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]

    names = ["move_cost", "goal_approach", "northward", "eastward"]
    F = np.zeros((n_states, 5, 4))
    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        d = abs(r - goal_r) + abs(c - goal_c)
        for a, (dr, dc) in enumerate(deltas):
            nr, nc = r + dr, c + dc
            ns = (nr * grid_size + nc) if (0 <= nr < grid_size and 0 <= nc < grid_size) else s
            nd = abs(ns // grid_size - goal_r) + abs(ns % grid_size - goal_c)
            F[s, a, 0] = -1.0 if ns != s else 0.0
            F[s, a, 1] = (1.0 if nd < d else -1.0) if ns != s else 0.0
            F[s, a, 2] = 1.0 if a == 0 else (-1.0 if a == 1 else 0.0)
            F[s, a, 3] = 1.0 if a == 2 else (-1.0 if a == 3 else 0.0)

    F = jnp.array(F)
    theta = jnp.array([-0.5, 2.0, 0.1, 0.1])
    prob = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)
    rfn = ActionDependentReward(feature_matrix=F, parameter_names=names)
    return prob, transitions, rfn, names, theta


def build_case2_rust_style(grid_size, discount):
    """Case 2: State features mapped differently per action (Rust-style).
    keep: pay operating cost proportional to distance from goal.
    move (N/S/E/W): pay fixed movement cost.
    """
    n_states = grid_size * grid_size
    transitions = _build_transitions(grid_size)
    goal_r, goal_c = grid_size - 1, grid_size - 1

    names = ["operating_cost", "move_cost"]
    F = np.zeros((n_states, 5, 2))
    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        dist = (abs(r - goal_r) + abs(c - goal_c)) / (2.0 * grid_size)
        for a in range(5):
            if a == 4:  # Stay: pay operating cost
                F[s, a, 0] = -dist
                F[s, a, 1] = 0.0
            else:  # Move: pay fixed movement cost
                F[s, a, 0] = 0.0
                F[s, a, 1] = -1.0

    F = jnp.array(F)
    theta = jnp.array([2.0, 0.3])
    prob = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)
    rfn = ActionDependentReward(feature_matrix=F, parameter_names=names)
    return prob, transitions, rfn, names, theta


def build_case3_state_only(grid_size, discount):
    """Case 3: Pure state-only features — same for all actions.
    Policy driven entirely by transition-mediated value differences.
    """
    n_states = grid_size * grid_size
    transitions = _build_transitions(grid_size)
    goal_r, goal_c = grid_size - 1, grid_size - 1

    names = ["goal_dist", "center_dist"]
    F = np.zeros((n_states, 5, 2))
    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        goal_d = -(abs(r - goal_r) + abs(c - goal_c)) / (2.0 * grid_size)
        center_d = -np.sqrt((r - grid_size/2)**2 + (c - grid_size/2)**2) / grid_size
        for a in range(5):  # Same features for all actions
            F[s, a, 0] = goal_d
            F[s, a, 1] = center_d

    F = jnp.array(F)
    theta = jnp.array([3.0, 0.5])
    prob = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)
    rfn = ActionDependentReward(feature_matrix=F, parameter_names=names)
    return prob, transitions, rfn, names, theta


# =====================================================================
# Helpers
# =====================================================================

def generate_data(prob, trans, rfn, theta, n_traj, n_periods, seed):
    op = SoftBellmanOperator(prob, trans)
    sol = hybrid_iteration(op, rfn.compute(theta), tol=1e-10)
    init = jnp.zeros(prob.num_states).at[0].set(1.0)
    panel = simulate_panel_from_policy(prob, trans, sol.policy, init,
                                       n_individuals=n_traj, n_periods=n_periods, seed=seed)
    return panel, sol.policy


def eval_on_panel(params, rfn, prob, trans, panel):
    op = SoftBellmanOperator(prob, trans)
    sol = hybrid_iteration(op, rfn.compute(params), tol=1e-10)
    lp = op.compute_log_choice_probabilities(rfn.compute(params), sol.V)
    ll = lp[panel.get_all_states(), panel.get_all_actions()].sum().item()
    n = panel.num_observations
    pred = sol.policy.argmax(1)[panel.get_all_states()]
    acc = (pred == panel.get_all_actions()).astype(jnp.float32).mean().item() * 100
    return {"ll_per_obs": ll / n, "accuracy": acc}


def make_estimators(verbose):
    return {
        "MCE IRL": MCEIRLEstimator(config=MCEIRLConfig(
            learning_rate=0.05, outer_max_iter=1000, outer_tol=1e-8,
            inner_solver="hybrid", inner_tol=1e-10, inner_max_iter=10000,
            use_adam=True, compute_se=False, verbose=verbose,
        )),
        "MaxEnt IRL": MaxEntIRLEstimator(
            inner_solver="policy", inner_tol=1e-10, outer_tol=1e-6,
            outer_max_iter=500, compute_hessian=False, verbose=False,
        ),
        "NFXP": NFXPEstimator(
            inner_solver="hybrid", inner_tol=1e-10, outer_tol=1e-8,
            outer_max_iter=500, compute_hessian=False, verbose=False,
        ),
    }


def run_case(case_name, prob, trans, rfn, names, theta, n_traj, n_periods, seed, verbose):
    """Run one reward case: estimate + evaluate."""
    print(f"\n{'#' * 70}")
    print(f"  {case_name}")
    print(f"  True: {dict(zip(names, theta.tolist()))}")
    print(f"{'#' * 70}")

    # Data
    full, true_pol = generate_data(prob, trans, rfn, theta, n_traj, n_periods, seed)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(full.trajectories))
    n_tr = int(len(idx) * 0.7)
    train = Panel(trajectories=[full.trajectories[i] for i in idx[:n_tr]])
    test = Panel(trajectories=[full.trajectories[i] for i in idx[n_tr:]])
    print(f"  Train: {train.num_observations} obs, Test: {test.num_observations} obs")

    # Transfer: stochastic transitions (10% noise)
    rng_key = jnp.array(np.random.RandomState(seed).rand(*trans.shape))
    noise = rng_key / rng_key.sum(axis=2, keepdims=True)
    noisy = 0.9 * trans + 0.1 * noise
    noisy_prob = DDCProblem(num_states=prob.num_states, num_actions=prob.num_actions,
                            discount_factor=prob.discount_factor)
    transfer_panel, _ = generate_data(noisy_prob, noisy, rfn, theta, 1000, n_periods, seed + 10)

    # Estimate
    ESTIMATORS = make_estimators(verbose=False)  # suppress MCE IRL progress for multi-case
    est = {}
    for ename, estimator in ESTIMATORS.items():
        t0 = time.time()
        kw = dict(panel=train, utility=rfn, problem=prob, transitions=trans)
        if ename == "MCE IRL":
            kw["true_params"] = theta
        result = estimator.estimate(**kw)
        p = result.parameters
        cos = (jnp.dot(p, theta) / (jnp.linalg.norm(p) * jnp.linalg.norm(theta))).item()
        rmse = jnp.sqrt(jnp.mean((p - theta) ** 2)).item()
        est[ename] = {"params": p, "cos": cos, "rmse": rmse, "time": time.time() - t0,
                       "converged": result.converged}

    # --- Print reward recovery ---
    hdr = f"  {'Param':<18} {'True':>8}"
    for e in ESTIMATORS: hdr += f" {e:>12}"
    print(f"\n  Reward Recovery:")
    print(hdr)
    print(f"  {'-' * len(hdr)}")
    for i, nm in enumerate(names):
        line = f"  {nm:<18} {theta[i].item():>8.4f}"
        for e in ESTIMATORS: line += f" {est[e]['params'][i].item():>12.4f}"
        print(line)
    line_c = f"  {'Cosine sim':<18} {'':>8}"
    line_r = f"  {'RMSE':<18} {'':>8}"
    for e in ESTIMATORS:
        line_c += f" {est[e]['cos']:>12.4f}"
        line_r += f" {est[e]['rmse']:>12.4f}"
    print(line_c)
    print(line_r)

    # --- Print policy performance ---
    scenarios = [
        ("In-sample", train, prob, trans),
        ("Out-of-sample", test, prob, trans),
        ("Transfer: stochastic", transfer_panel, noisy_prob, noisy),
    ]
    print(f"\n  Policy Performance:")
    for sc_name, sc_panel, sc_prob, sc_trans in scenarios:
        h = f"  {sc_name + ' (n=' + str(sc_panel.num_observations) + ')':<40}"
        for e in ESTIMATORS: h += f" {e:>12}"
        print(h)
        ll_l = f"  {'  LL/obs':<40}"
        ac_l = f"  {'  Accuracy %':<40}"
        for e in ESTIMATORS:
            ev = eval_on_panel(est[e]["params"], rfn, sc_prob, sc_trans, sc_panel)
            ll_l += f" {ev['ll_per_obs']:>12.4f}"
            ac_l += f" {ev['accuracy']:>12.1f}"
        print(ll_l)
        print(ac_l)

    return est


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-traj", type=int, default=2000)
    parser.add_argument("--n-periods", type=int, default=50)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    gs, nt, np_, d, s = args.grid_size, args.n_traj, args.n_periods, args.discount, args.seed

    print("=" * 70)
    print(f"  Gridworld Benchmark: 3 Reward Cases ({gs}x{gs}, N={nt}, T={np_})")
    print("=" * 70)

    cases = [
        ("Case 1: State-action rewards (full identification)",
         *build_case1_state_action(gs, d)),
        ("Case 2: Rust-style (state features, action-dependent mapping)",
         *build_case2_rust_style(gs, d)),
        ("Case 3: Pure state-only (weak identification)",
         *build_case3_state_only(gs, d)),
    ]

    for case_name, prob, trans, rfn, names, theta in cases:
        run_case(case_name, prob, trans, rfn, names, theta, nt, np_, s, args.verbose)

    print(f"\n{'=' * 70}")
    print("  Done.")
    print("=" * 70)


if __name__ == "__main__":
    main()
