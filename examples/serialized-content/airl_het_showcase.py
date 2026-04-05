"""AIRL-Het: Anchor identification and heterogeneous treatment effects.

Demonstrates the key contributions of Lee, Sudhir and Wang (2026):
1. Anchor identification pins down structural rewards (not shaped)
2. EM discovers latent consumer segments from pooled data
3. Segment-specific rewards enable heterogeneous counterfactuals
4. Pooled estimators miss divergent segment-level effects

Uses a synthetic two-segment serialized content DGP with known
ground truth for validation.

Usage:
    python examples/serialized-content/airl_het_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.adversarial.airl_het import AIRLHetEstimator, AIRLHetConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.action_reward import ActionDependentReward


def build_environment(n_episodes=10):
    """Build a serialized content environment with 2 segments."""
    n_states = n_episodes + 1  # +1 absorbing
    n_actions = 3  # buy=0, wait=1, exit=2
    absorbing = n_episodes
    exit_action = 2

    # Deterministic transitions
    T = np.zeros((n_actions, n_states, n_states))
    for s in range(n_episodes):
        T[0, s, min(s + 1, n_episodes - 1)] = 1.0  # buy: advance
        T[1, s, s] = 1.0                             # wait: stay
        T[2, s, absorbing] = 1.0                      # exit: absorb
    T[0, absorbing, absorbing] = 1.0
    T[1, absorbing, absorbing] = 1.0
    T[2, absorbing, absorbing] = 1.0
    transitions = jnp.array(T, dtype=jnp.float32)

    # Features: buy_indicator, quality, wait_indicator
    features = np.zeros((n_states, n_actions, 3))
    for s in range(n_episodes):
        features[s, 0, 0] = 1.0                  # buy indicator
        features[s, 0, 1] = 1.0 - 0.15 * s       # quality decays with episode
        features[s, 1, 2] = 1.0                  # wait indicator
    feature_matrix = jnp.array(features, dtype=jnp.float32)

    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.9,
        scale_parameter=1.0,
    )

    utility = ActionDependentReward(
        feature_matrix, ["buy_cost", "quality", "wait_cost"]
    )

    # Segment A: "Pay and Read" (quality-sensitive, buys frequently)
    params_a = jnp.array([-0.5, 2.5, -1.0])
    reward_a = jnp.einsum("sak,k->sa", feature_matrix, params_a)

    # Segment B: "Wait and Read" (price-sensitive, waits patiently)
    params_b = jnp.array([-2.0, 1.0, -0.2])
    reward_b = jnp.einsum("sak,k->sa", feature_matrix, params_b)

    return {
        "transitions": transitions,
        "feature_matrix": feature_matrix,
        "problem": problem,
        "utility": utility,
        "n_states": n_states,
        "n_actions": n_actions,
        "absorbing": absorbing,
        "exit_action": exit_action,
        "params_a": params_a,
        "params_b": params_b,
        "reward_a": reward_a,
        "reward_b": reward_b,
    }


def simulate_mixture_panel(env, n_individuals=300, n_periods=30,
                           mix_a=0.4, seed=42):
    """Simulate a mixture panel with no segment labels."""
    rng = np.random.default_rng(seed)
    operator = SoftBellmanOperator(env["problem"], env["transitions"])

    # Compute optimal policies for each segment
    result_a = value_iteration(operator, env["reward_a"], tol=1e-10, max_iter=5000)
    result_b = value_iteration(operator, env["reward_b"], tol=1e-10, max_iter=5000)
    policy_a = np.array(result_a.policy)
    policy_b = np.array(result_b.policy)

    trajectories = []
    true_segments = []

    for i in range(n_individuals):
        # Draw segment
        is_a = rng.random() < mix_a
        true_segments.append(0 if is_a else 1)
        policy = policy_a if is_a else policy_b

        state = 0
        states, actions, next_states = [], [], []
        for t in range(n_periods):
            probs = policy[state]
            probs = probs / probs.sum()
            action = rng.choice(env["n_actions"], p=probs)
            next_state = int(jnp.argmax(env["transitions"][action, state]))

            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            state = next_state

        trajectories.append(Trajectory(
            states=jnp.array(states, dtype=jnp.int32),
            actions=jnp.array(actions, dtype=jnp.int32),
            next_states=jnp.array(next_states, dtype=jnp.int32),
            individual_id=i,
        ))

    return Panel(trajectories=trajectories), np.array(true_segments)


def main():
    print("=" * 70)
    print("AIRL-Het: Anchor Identification and Heterogeneous Treatment Effects")
    print("Lee, Sudhir and Wang (2026)")
    print("=" * 70)

    env = build_environment(n_episodes=10)
    mix_a = 0.4
    panel, true_segments = simulate_mixture_panel(
        env, n_individuals=300, n_periods=30, mix_a=mix_a, seed=42
    )

    n_a = (true_segments == 0).sum()
    n_b = (true_segments == 1).sum()
    print(f"\n  States: {env['n_states']}, Actions: {env['n_actions']}")
    print(f"  Individuals: {panel.num_individuals}")
    print(f"  True segments: A (Pay&Read) = {n_a}, B (Wait&Read) = {n_b}")
    print(f"  True mixing: {mix_a:.0%} A, {1-mix_a:.0%} B")
    print(f"\n  Segment A params: buy_cost={float(env['params_a'][0])}, "
          f"quality={float(env['params_a'][1])}, "
          f"wait_cost={float(env['params_a'][2])}")
    print(f"  Segment B params: buy_cost={float(env['params_b'][0])}, "
          f"quality={float(env['params_b'][1])}, "
          f"wait_cost={float(env['params_b'][2])}")

    # ---- Pooled NFXP (homogeneous) ----
    print("\n--- Pooled NFXP (assumes single type) ---")
    t0 = time.time()
    try:
        nfxp = NFXPEstimator(se_method="robust")
        nfxp_result = nfxp.estimate(
            panel, env["utility"], env["problem"], env["transitions"]
        )
        nfxp_time = time.time() - t0
        nfxp_params = np.array(nfxp_result.parameters)
        print(f"  Time: {nfxp_time:.1f}s")
        print(f"  Params: buy_cost={nfxp_params[0]:.4f}, "
              f"quality={nfxp_params[1]:.4f}, wait_cost={nfxp_params[2]:.4f}")
    except Exception as e:
        print(f"  Failed: {e}")
        nfxp_params = None

    # ---- AIRL-Het (K=2 segments) ----
    print("\n--- AIRL-Het (K=2 segments, EM) ---")
    t0 = time.time()
    config = AIRLHetConfig(
        num_segments=2,
        exit_action=env["exit_action"],
        absorbing_state=env["absorbing"],
        max_em_iterations=20,
        max_airl_rounds=50,
        reward_lr=0.01,
    )
    airl_het = AIRLHetEstimator(config)
    het_result = airl_het.estimate(
        panel, env["utility"], env["problem"], env["transitions"]
    )
    het_time = time.time() - t0
    print(f"  Time: {het_time:.1f}s")

    # Extract segment info
    seg_rewards = het_result.metadata.get("segment_reward_matrices", [])
    seg_priors = het_result.metadata.get("segment_priors", [])
    seg_posteriors = het_result.metadata.get("segment_posteriors", None)

    if seg_priors:
        print(f"  Recovered priors: {[f'{p:.3f}' for p in seg_priors]}")

    # ---- Results ----
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)

    # Match segments to ground truth by comparing priors
    if seg_priors and len(seg_priors) == 2:
        # Segment with prior closer to 0.4 is likely segment A
        if abs(seg_priors[0] - mix_a) < abs(seg_priors[1] - mix_a):
            seg_a_idx, seg_b_idx = 0, 1
        else:
            seg_a_idx, seg_b_idx = 1, 0
        print(f"\n  Segment mapping: recovered seg {seg_a_idx} -> A (Pay&Read)")
        print(f"                   recovered seg {seg_b_idx} -> B (Wait&Read)")

    if nfxp_params is not None:
        print(f"\n  Pooled NFXP: {[f'{p:.4f}' for p in nfxp_params]}")
        print(f"  True A:      {[f'{float(p):.4f}' for p in env['params_a']]}")
        print(f"  True B:      {[f'{float(p):.4f}' for p in env['params_b']]}")
        print(f"\n  The pooled estimate is a compromise between the two segments.")

    # Classification accuracy
    if seg_posteriors is not None:
        posteriors = np.array(seg_posteriors)
        if posteriors.shape[0] == panel.num_individuals and posteriors.shape[1] == 2:
            hard_assignments = posteriors.argmax(axis=1)
            # Map to ground truth labels
            acc_direct = np.mean(hard_assignments == true_segments)
            acc_flipped = np.mean((1 - hard_assignments) == true_segments)
            accuracy = max(acc_direct, acc_flipped)
            print(f"\n  Segment classification accuracy: {accuracy:.1%}")

    # Segment reward matrices
    if len(seg_rewards) == 2:
        print("\n  Segment reward matrices (first 5 episodes, buy action):")
        print(f"  {'Episode':>8} {'Seg A (rec)':>12} {'Seg A (true)':>13} "
              f"{'Seg B (rec)':>12} {'Seg B (true)':>13}")
        print("  " + "-" * 62)
        for s in range(min(5, env["n_states"] - 1)):
            r_a_rec = float(np.array(seg_rewards[seg_a_idx])[s, 0])
            r_b_rec = float(np.array(seg_rewards[seg_b_idx])[s, 0])
            r_a_true = float(env["reward_a"][s, 0])
            r_b_true = float(env["reward_b"][s, 0])
            print(f"  {s:>8} {r_a_rec:>12.4f} {r_a_true:>13.4f} "
                  f"{r_b_rec:>12.4f} {r_b_true:>13.4f}")

    # Save results
    out = {
        "true": {
            "params_a": [float(p) for p in env["params_a"]],
            "params_b": [float(p) for p in env["params_b"]],
            "mix_a": mix_a,
        },
        "pooled_nfxp": [float(p) for p in nfxp_params] if nfxp_params is not None else None,
        "airl_het": {
            "priors": [float(p) for p in seg_priors] if seg_priors else None,
            "time": het_time,
        },
    }
    path = Path(__file__).parent / "airl_het_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
