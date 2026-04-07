#!/usr/bin/env python3
"""AAIRL Primer -- auto-generate results for aairl.tex.

Two-segment serialized content MDP (51 states, 3 actions, 6 features).
Segment A = quality-sensitive ("Heavy Reader"), Segment B = price-sensitive.
Shows AAIRL K=2 recovers both segment reward functions and identifies users.
Baselines: BC (logistic regression), AIRL K=1 (homogeneous, wrong model).

Usage:
    cd papers/econirl_package/primers/aairl
    python aairl_run.py
    pdflatex aairl.tex
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig
from econirl.preferences.action_reward import ActionDependentReward

OUT = Path(__file__).resolve().parent / "aairl_results.tex"
JSON_OUT = Path(__file__).resolve().parent / "aairl_results.json"

# --- Environment constants ---
N_EPISODES = 50
N_STATES = N_EPISODES + 1
N_ACTIONS = 3
ABSORBING = N_EPISODES
EXIT_ACTION = 2
DISCOUNT = 0.90

PARAM_NAMES = ["buy_cost", "quality_buy", "cliffhanger_buy",
               "wait_cost", "quality_wait", "progress_buy"]

# Two-segment true parameters (linear, 6-dimensional).
# Contrasting but realistic: type-A values quality, type-B is price-sensitive.
TRUE_A = np.array([-0.2,  2.5,  1.2, -0.4,  0.8,  0.6])   # quality-loving Heavy Reader
TRUE_B = np.array([-1.8,  0.4,  0.3, -0.1,  0.1,  0.2])   # price-sensitive
PRIOR_A = 0.55
N_USERS = 400
N_BOOKS = 3


def build_environment(seed=42):
    """Build shared feature matrix and transition tensor."""
    rng = np.random.default_rng(seed)
    quality = np.zeros(N_EPISODES)
    for book in range(5):
        for ep in range(10):
            s = book * 10 + ep
            base = 0.5 + 0.3 * np.sin(np.pi * ep / 9)
            quality[s] = np.clip(base + 0.1 * rng.standard_normal(), 0.1, 1.0)
    cliffhanger = (quality > 0.7).astype(float)
    book_progress = np.array([(s % 10) / 9.0 for s in range(N_EPISODES)])

    T = np.zeros((N_ACTIONS, N_STATES, N_STATES))
    for s in range(N_EPISODES):
        next_ep = s + 1
        if next_ep >= N_EPISODES or (s + 1) % 10 == 0:
            T[0, s, ABSORBING] = 1.0
        else:
            T[0, s, next_ep] = 1.0
        T[1, s, s] = 1.0
        T[2, s, ABSORBING] = 1.0
    for a in range(N_ACTIONS):
        T[a, ABSORBING, ABSORBING] = 1.0

    features = np.zeros((N_STATES, N_ACTIONS, 6))
    for s in range(N_EPISODES):
        features[s, 0, 0] = 1.0               # buy_cost intercept
        features[s, 0, 1] = quality[s]        # quality_buy
        features[s, 0, 2] = cliffhanger[s]    # cliffhanger_buy
        features[s, 1, 3] = 1.0               # wait_cost intercept
        features[s, 1, 4] = quality[s]        # quality_wait
        features[s, 0, 5] = book_progress[s]  # progress_buy

    transitions = jnp.array(T, dtype=jnp.float32)
    features_j = jnp.array(features, dtype=jnp.float32)
    utility = ActionDependentReward(features_j, PARAM_NAMES)
    problem = DDCProblem(N_STATES, N_ACTIONS, DISCOUNT)
    return transitions, features_j, features, utility, problem


def make_policy(transitions, features_j, params, problem):
    """Compute softmax policy under given linear parameters."""
    reward = jnp.einsum("sak,k->sa", features_j,
                        jnp.array(params, dtype=jnp.float32))
    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, reward, tol=1e-10, max_iter=5000)
    return np.array(result.policy)


def generate_panel(transitions, features_j, problem, seed=42):
    """Generate N_USERS x N_BOOKS trajectories with two latent segments."""
    rng = np.random.default_rng(seed)
    policy_A = make_policy(transitions, features_j, TRUE_A, problem)
    policy_B = make_policy(transitions, features_j, TRUE_B, problem)

    n_A = int(round(PRIOR_A * N_USERS))
    types = np.array([0] * n_A + [1] * (N_USERS - n_A))
    rng.shuffle(types)

    trajectories = []
    true_assignments = []
    for i, seg in enumerate(types):
        pol = policy_A if seg == 0 else policy_B
        books = rng.choice(5, size=N_BOOKS, replace=False)
        for book in books:
            state = int(book * 10)
            states, actions, next_states = [], [], []
            for _ in range(20):
                if state == ABSORBING:
                    break
                probs = np.maximum(pol[state], 0)
                probs /= probs.sum()
                action = int(rng.choice(N_ACTIONS, p=probs))
                ns = int(jnp.argmax(transitions[action, state]))
                states.append(state)
                actions.append(action)
                next_states.append(ns)
                state = ns
            if len(states) >= 2:
                trajectories.append(Trajectory(
                    states=jnp.array(states, dtype=jnp.int32),
                    actions=jnp.array(actions, dtype=jnp.int32),
                    next_states=jnp.array(next_states, dtype=jnp.int32),
                    individual_id=i,
                ))
                true_assignments.append(seg)
    return Panel(trajectories=trajectories), np.array(true_assignments)


def cosine_sim(a, b):
    a, b = np.asarray(a, float).flatten(), np.asarray(b, float).flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def bc_policy_accuracy(panel):
    """Logistic-regression BC: predict action from state (lower bound baseline)."""
    from sklearn.linear_model import LogisticRegression
    X, y = [], []
    for traj in panel.trajectories:
        for s, a in zip(np.array(traj.states), np.array(traj.actions)):
            X.append([s])
            y.append(a)
    X, y = np.array(X), np.array(y)
    clf = LogisticRegression(max_iter=500, C=1.0).fit(X, y)
    return float(clf.score(X, y))


def policy_accuracy(panel, policy):
    """Fraction of (s,a) pairs where argmax policy agrees with observed action."""
    correct, total = 0, 0
    pol = np.array(policy)
    for traj in panel.trajectories:
        for s, a in zip(np.array(traj.states), np.array(traj.actions)):
            if s < pol.shape[0] and int(pol[s].argmax()) == a:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def main():
    print("Building environment...")
    transitions, features_j, features_np, utility, problem = build_environment()

    # Precompute true reward matrices for cosine similarity (tabular)
    true_rm_A = np.einsum("sak,k->sa", features_np, TRUE_A)  # (51, 3)
    true_rm_B = np.einsum("sak,k->sa", features_np, TRUE_B)  # (51, 3)

    print("Generating 2-segment panel...")
    panel, true_types = generate_panel(transitions, features_j, problem)
    n_obs = sum(len(t.states) for t in panel.trajectories)
    print(f"  {len(panel.trajectories)} trajectories, {n_obs:,} observations")
    print(f"  True split: {(true_types == 0).sum()} type-A, {(true_types == 1).sum()} type-B")

    # --- BC baseline ---
    print("\nRunning BC baseline (logistic regression)...")
    try:
        bc_acc = bc_policy_accuracy(panel)
        print(f"  BC accuracy: {bc_acc:.3f}")
    except ImportError:
        print("  sklearn not available, skipping BC")
        bc_acc = float("nan")

    # --- AIRL K=1 (homogeneous, misspecified model) ---
    print("\nRunning AIRL K=1 (homogeneous — misspecified)...")
    t0 = time.time()
    airl1 = AIRLEstimator(AIRLConfig(
        reward_type="linear",
        max_rounds=200,
        reward_lr=0.01,
        discriminator_steps=3,
        policy_step_size=0.15,
        use_shaping=True,
        convergence_tol=1e-4,
        verbose=True,
    ))
    r1 = airl1.estimate(panel, utility, problem, transitions)
    t_airl1 = time.time() - t0
    # K=1 recovers an "average" linear reward — compare to prior-weighted average
    true_avg_params = PRIOR_A * TRUE_A + (1 - PRIOR_A) * TRUE_B
    k1_params = np.array(r1.parameters)
    cos_k1 = cosine_sim(k1_params, true_avg_params)
    pacc_k1 = policy_accuracy(panel, r1.policy)
    print(f"  K=1 cos(avg): {cos_k1:.3f}, policy_acc: {pacc_k1:.3f}, time: {t_airl1:.1f}s")

    # --- AAIRL K=2 (linear reward, antisymmetric init, anchor constraints) ---
    print("\nRunning AAIRL K=2 (heterogeneous — correct model)...")
    t0 = time.time()
    aairl = AIRLHetEstimator(AIRLHetConfig(
        num_segments=2,
        exit_action=EXIT_ACTION,
        absorbing_state=ABSORBING,
        reward_type="linear",
        reward_lr=0.02,
        antisymmetric_init=True,
        prior_min=0.15,
        prior_damping=0.3,
        unit_normalize_reward=True,
        discriminator_steps=3,
        max_airl_rounds=20,
        max_em_iterations=80,
        airl_convergence_tol=1e-4,
        em_convergence_tol=1e-3,
        consistency_weight=0.2,
        verbose=True,
        seed=42,
    ))
    r2 = aairl.estimate(panel, utility, problem, transitions)
    t_aairl = time.time() - t0

    meta = r2.metadata
    n_feat = len(PARAM_NAMES)

    # Extract linear parameters per segment from the concatenated params
    params_flat = np.array(r2.parameters)
    seg_params = [params_flat[:n_feat], params_flat[n_feat:2 * n_feat]]

    # Hungarian matching: align estimated segments to true segments by cosine sim
    cos_mat = np.array([
        [cosine_sim(seg_params[0], TRUE_A), cosine_sim(seg_params[0], TRUE_B)],
        [cosine_sim(seg_params[1], TRUE_A), cosine_sim(seg_params[1], TRUE_B)],
    ])
    row_ind, col_ind = linear_sum_assignment(-cos_mat)
    cos_vals = [cos_mat[row_ind[k], col_ind[k]] for k in range(2)]

    # Segment assignment accuracy
    assignments = np.array(meta["segment_assignments"])
    remapped = np.array([col_ind[a] for a in assignments])
    seg_acc = float((remapped == true_types).mean())

    priors = meta["segment_priors"]
    print(f"\n  AAIRL K=2 results:")
    print(f"    Segment accuracy: {seg_acc:.3f}")
    print(f"    Cosine seg-A: {cos_vals[0]:.3f}, seg-B: {cos_vals[1]:.3f}")
    print(f"    Recovered priors: {[f'{p:.2f}' for p in priors]}")
    print(f"    Time: {t_aairl:.1f}s")

    # Policy accuracy per segment (using segment-specific policies)
    seg_policies = meta["segment_policies"]
    pol_A = np.array(seg_policies[row_ind[0]])
    pol_B = np.array(seg_policies[row_ind[1]])
    trajs_A = Panel(trajectories=[
        t for t, seg in zip(panel.trajectories, true_types) if seg == 0])
    trajs_B = Panel(trajectories=[
        t for t, seg in zip(panel.trajectories, true_types) if seg == 1])
    pacc_A = policy_accuracy(trajs_A, pol_A) if trajs_A.trajectories else 0.0
    pacc_B = policy_accuracy(trajs_B, pol_B) if trajs_B.trajectories else 0.0
    print(f"    Policy acc seg-A: {pacc_A:.3f}, seg-B: {pacc_B:.3f}")

    # --- Write JSON ---
    results = {
        "bc": {"policy_acc": bc_acc},
        "airl_k1": {"cos_avg": cos_k1, "policy_acc": pacc_k1, "time": t_airl1},
        "aairl_k2": {
            "seg_acc": seg_acc,
            "cos_A": cos_vals[0],
            "cos_B": cos_vals[1],
            "prior_A": float(priors[row_ind[0]]),
            "prior_B": float(priors[row_ind[1]]),
            "policy_acc_A": pacc_A,
            "policy_acc_B": pacc_B,
            "time": t_aairl,
        },
        "experiment": {
            "n_users": N_USERS,
            "n_books": N_BOOKS,
            "n_obs": n_obs,
            "prior_A": PRIOR_A,
            "true_A": TRUE_A.tolist(),
            "true_B": TRUE_B.tolist(),
        },
    }
    JSON_OUT.write_text(json.dumps(results, indent=2))

    # --- Write LaTeX macros + table ---
    def f(v, d=3):
        return f"{v:.{d}f}"

    tex = [
        "% Auto-generated by aairl_run.py -- do not edit by hand",
        f"% N={N_USERS} users x {N_BOOKS} books, K=2 segments, {N_STATES} states",
        "",
        f"\\newcommand{{\\aairlNusers}}{{{N_USERS}}}",
        f"\\newcommand{{\\aairlNobs}}{{{n_obs:,}}}",
        f"\\newcommand{{\\aairlSegAcc}}{{{f(seg_acc)}}}",
        f"\\newcommand{{\\aairlCosA}}{{{f(cos_vals[0])}}}",
        f"\\newcommand{{\\aairlCosB}}{{{f(cos_vals[1])}}}",
        f"\\newcommand{{\\aairlPriorA}}{{{f(priors[row_ind[0]])}}}",
        f"\\newcommand{{\\aairlPriorB}}{{{f(priors[row_ind[1]])}}}",
        f"\\newcommand{{\\aairlPaccA}}{{{f(pacc_A)}}}",
        f"\\newcommand{{\\aairlPaccB}}{{{f(pacc_B)}}}",
        f"\\newcommand{{\\aairlTime}}{{{f(t_aairl, 1)}}}",
        f"\\newcommand{{\\bcAcc}}{{{f(bc_acc)}}}",
        f"\\newcommand{{\\airlKoneCos}}{{{f(cos_k1)}}}",
        f"\\newcommand{{\\airlKonePacc}}{{{f(pacc_k1)}}}",
        "",
        "\\begin{table}[H]",
        "\\centering\\small",
        "\\caption{Two-segment recovery on 51-state content MDP."
        " AAIRL~$K{=}2$ identifies both latent types;"
        " AIRL~$K{=}1$ recovers only a useless average;"
        " BC matches surface behavior only.}",
        "\\label{tab:aairl_results}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r}",
        "\\toprule",
        "Metric & BC & AIRL~$K{=}1$ & AAIRL~$K{=}2$ \\\\",
        "\\midrule",
        "Segment assignment accuracy & --- & --- & \\aairlSegAcc \\\\",
        "Cosine sim, Segment~A & --- & \\airlKoneCos & \\aairlCosA \\\\",
        "Cosine sim, Segment~B & --- & \\airlKoneCos & \\aairlCosB \\\\",
        "Policy accuracy, Segment~A & \\bcAcc & \\airlKonePacc & \\aairlPaccA \\\\",
        "Policy accuracy, Segment~B & \\bcAcc & \\airlKonePacc & \\aairlPaccB \\\\",
        "\\bottomrule",
        "\\end{tabular*}",
        "\\end{table}",
    ]
    OUT.write_text("\n".join(tex) + "\n")
    print(f"\nWrote {OUT}")
    print(f"Wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
