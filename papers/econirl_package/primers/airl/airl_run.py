#!/usr/bin/env python3
"""
AIRL Primer -- Companion simulation (Fu et al. 2018).

Tests AIRL on a 5x5 gridworld with state-only reward (goal at corner),
4 actions (N/S/E/W), stochastic transitions, gamma=0.95.
Writes results to airl_results.tex for automatic inclusion in the primer.

Usage:
    cd papers/econirl_package/primers/airl
    python airl_run.py
    pdflatex airl.tex && bibtex airl && pdflatex airl.tex && pdflatex airl.tex
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.airl import AIRLEstimator, AIRLConfig
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.reward import LinearReward

# ---------------------------------------------------------------------------
# 5x5 Gridworld with state-only reward
# ---------------------------------------------------------------------------

GRID = 5
N_STATES = GRID * GRID  # 25
N_ACTIONS = 4           # 0=N, 1=S, 2=E, 3=W
GAMMA = 0.95
GOAL = (GRID - 1) * GRID + (GRID - 1)  # bottom-right corner = state 24
SLIP_PROB = 0.2  # 20% chance of random action


def state_to_rc(s):
    return s // GRID, s % GRID


def rc_to_state(r, c):
    return r * GRID + c


def build_gridworld():
    """Build 5x5 gridworld with stochastic transitions and state-only reward."""
    T = np.zeros((N_ACTIONS, N_STATES, N_STATES))
    # Action deltas: N=(-1,0), S=(+1,0), E=(0,+1), W=(0,-1)
    deltas = [(-1, 0), (1, 0), (0, 1), (0, -1)]

    for s in range(N_STATES):
        r, c = state_to_rc(s)
        for a in range(N_ACTIONS):
            for actual_a in range(N_ACTIONS):
                if actual_a == a:
                    prob = 1.0 - SLIP_PROB
                else:
                    prob = SLIP_PROB / (N_ACTIONS - 1)

                dr, dc = deltas[actual_a]
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID and 0 <= nc < GRID:
                    ns = rc_to_state(nr, nc)
                else:
                    ns = s  # bounce off wall
                T[a, s, ns] += prob

    # State-only reward: 1 at goal, 0 elsewhere
    true_reward_vec = np.zeros(N_STATES)
    true_reward_vec[GOAL] = 1.0

    # State features for LinearReward: identity (one feature per state)
    # But for AIRL state-only, we use tabular reward
    # For MCE-IRL comparison, use position features
    # Features: (row/4, col/4, goal_indicator)
    n_features = 3
    features = np.zeros((N_STATES, n_features))
    for s in range(N_STATES):
        r, c = state_to_rc(s)
        features[s, 0] = r / (GRID - 1)
        features[s, 1] = c / (GRID - 1)
        features[s, 2] = 1.0 if s == GOAL else 0.0

    transitions = jnp.array(T, dtype=jnp.float32)
    problem = DDCProblem(N_STATES, N_ACTIONS, GAMMA)

    # True reward as (N_STATES, N_ACTIONS) -- same across actions (state-only)
    true_reward_sa = np.tile(true_reward_vec[:, None], (1, N_ACTIONS))

    return {
        "transitions": transitions,
        "problem": problem,
        "true_reward_vec": true_reward_vec,
        "true_reward_sa": jnp.array(true_reward_sa, dtype=jnp.float32),
        "features": features,
        "n_features": n_features,
    }


def generate_expert_panel(env, n_trajs=500, max_steps=50, seed=42):
    """Generate expert demonstrations from the optimal policy."""
    rng = np.random.default_rng(seed)
    operator = SoftBellmanOperator(env["problem"], env["transitions"])
    result = value_iteration(operator, env["true_reward_sa"], tol=1e-10, max_iter=5000)
    policy = np.array(result.policy)

    trajectories = []
    for i in range(n_trajs):
        state = rng.integers(0, N_STATES)
        states, actions, next_states = [], [], []
        for _ in range(max_steps):
            probs = np.maximum(policy[state], 0)
            probs /= probs.sum()
            action = rng.choice(N_ACTIONS, p=probs)
            trans_probs = np.array(env["transitions"][action, state])
            ns = rng.choice(N_STATES, p=trans_probs)
            states.append(state)
            actions.append(action)
            next_states.append(ns)
            state = ns
        trajectories.append(Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=i,
        ))
    return Panel(trajectories=trajectories), policy


def evaluate_reward(recovered, true_vec):
    """Compute cosine similarity and policy match between recovered and true reward."""
    # Average recovered reward across actions (state-only comparison)
    if recovered.ndim == 2:
        rec_vec = np.array(recovered).mean(axis=1)
    else:
        rec_vec = np.array(recovered)

    # Cosine similarity
    dot = np.dot(rec_vec, true_vec)
    norm_rec = np.linalg.norm(rec_vec)
    norm_true = np.linalg.norm(true_vec)
    cosine = dot / max(norm_rec * norm_true, 1e-10)

    # Correlation
    corr = np.corrcoef(rec_vec, true_vec)[0, 1]

    return cosine, corr, rec_vec


def write_results_tex(metrics, path="airl_results.tex"):
    lines = ["% Auto-generated by airl_run.py -- do not edit by hand\n"]
    for name, true_val, airl_val in metrics:
        tex_name = name.replace("_", r"\_")
        lines.append(f"{tex_name} & ${true_val}$ & ${airl_val}$ \\\\\n")
    Path(path).write_text("".join(lines))
    print(f"Results written to {path}")


def main():
    print("=" * 60)
    print("AIRL Primer: 5x5 Gridworld (state-only reward)")
    print("=" * 60)

    env = build_gridworld()
    print(f"States: {N_STATES}, Actions: {N_ACTIONS}, Gamma: {GAMMA}")
    print(f"Goal state: {GOAL} (bottom-right corner)")

    print("\nGenerating expert demonstrations...")
    panel, expert_policy = generate_expert_panel(env, n_trajs=500, max_steps=50)
    n_obs = sum(len(t.states) for t in panel.trajectories)
    print(f"  {len(panel.trajectories)} trajectories, {n_obs} observations")

    # AIRL (tabular, state-only reward)
    print("\nRunning AIRL (tabular, state-only, conservative VI)...")
    t0 = time.time()
    airl = AIRLEstimator(AIRLConfig(
        reward_type="tabular",
        max_rounds=1000,
        reward_lr=0.02,
        reward_weight_decay=0.0,
        discriminator_steps=3,
        policy_step_size=0.3,
        use_shaping=True,
        convergence_tol=1e-6,
        verbose=True,
    ))
    airl_result = airl.estimate(
        panel, LinearReward(jnp.array(env["features"], dtype=jnp.float32),
                            ["row", "col", "goal"]),
        env["problem"], env["transitions"]
    )
    airl_time = time.time() - t0
    print(f"  Time: {airl_time:.1f}s")

    # Extract reward matrix
    airl_reward = None
    if airl_result.metadata and "reward_matrix" in airl_result.metadata:
        airl_reward = np.array(airl_result.metadata["reward_matrix"])

    # Evaluate
    if airl_reward is not None:
        cosine, corr, airl_vec = evaluate_reward(airl_reward, env["true_reward_vec"])
    else:
        cosine, corr, airl_vec = 0.0, 0.0, np.zeros(N_STATES)

    # Policy match
    if airl_result.policy is not None:
        airl_greedy = np.array(airl_result.policy).argmax(axis=1)
        expert_greedy = expert_policy.argmax(axis=1)
        policy_match = (airl_greedy == expert_greedy).mean()
    else:
        policy_match = 0.0

    # Print
    print(f"\n{'Metric':>25} {'Value':>10}")
    print("-" * 38)
    print(f"{'Cosine similarity':>25} {cosine:>10.4f}")
    print(f"{'Correlation':>25} {corr:>10.4f}")
    print(f"{'Policy match rate':>25} {policy_match:>10.4f}")
    print(f"{'AIRL time (s)':>25} {airl_time:>10.1f}")
    print(f"{'Goal reward (true)':>25} {'1.00':>10}")
    print(f"{'Goal reward (AIRL)':>25} {airl_vec[GOAL]:>10.4f}")
    print(f"{'Mean non-goal (AIRL)':>25} {np.mean(airl_vec[np.arange(N_STATES) != GOAL]):>10.4f}")

    # Write tex
    metrics = [
        ("Cosine similarity", "---", f"{cosine:.3f}"),
        ("Correlation", "---", f"{corr:.3f}"),
        ("Policy match rate", "---", f"{policy_match:.3f}"),
        ("Goal reward", "1.00", f"{airl_vec[GOAL]:.3f}"),
        ("Mean non-goal reward", "0.00", f"{np.mean(airl_vec[np.arange(N_STATES) != GOAL]):.3f}"),
    ]
    write_results_tex(metrics)

    # JSON
    out = {
        "cosine_similarity": float(cosine),
        "correlation": float(corr),
        "policy_match": float(policy_match),
        "goal_reward_true": 1.0,
        "goal_reward_airl": float(airl_vec[GOAL]),
        "mean_nongol_airl": float(np.mean(airl_vec[np.arange(N_STATES) != GOAL])),
        "airl_time": airl_time,
    }
    Path("airl_results.json").write_text(json.dumps(out, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
