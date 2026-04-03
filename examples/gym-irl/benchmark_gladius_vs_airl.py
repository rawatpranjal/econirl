#!/usr/bin/env python3
"""
GLADIUS vs AIRL Benchmark on OpenAI Gym Environments
=====================================================

Head-to-head comparison of NeuralGLADIUS and NeuralAIRL on three classic
control environments following the experimental protocol from Kang et al.
(2025, arXiv:2502.14131).

Environments: CartPole-v1, Acrobot-v1, LunarLander-v2
Data conditions: 2 trajectories (low), 15 trajectories (high)
Seeds: 3 random initializations per condition

Evaluation metrics:
  1. Action prediction accuracy on training data
  2. Normalized reward: greedy policy reward / expert reward
  3. Forward RL: train PPO on recovered rewards, test on true rewards

Prerequisites:
    python scripts/generate_gym_expert_data.py

Usage:
    python examples/gym-irl/benchmark_gladius_vs_airl.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from gym_irl_utils import (
    load_expert_data,
    build_dataframe_and_encoder,
    evaluate_neural_policy,
)
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.estimators.neural_airl import NeuralAIRL

# Optional: forward RL evaluation
try:
    from econirl.rl.ppo import train_ppo, evaluate_policy, PPOConfig, ActorCritic
    FORWARD_RL_AVAILABLE = True
except ImportError:
    FORWARD_RL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ENV_CONFIGS = {
    "CartPole-v1": {"obs_dim": 4, "n_actions": 2, "n_bins": 6},
    "Acrobot-v1": {"obs_dim": 6, "n_actions": 3, "n_bins": 5},
    "LunarLander-v3": {"obs_dim": 8, "n_actions": 4, "n_bins": 3},
}

DATA_CONDITIONS = [2, 15]
SEEDS = [42, 123, 456]
FORWARD_RL = True  # Set False to skip PPO re-training evaluation


# ---------------------------------------------------------------------------
# Reward wrapper for forward RL evaluation
# ---------------------------------------------------------------------------

class IRLRewardWrapper(gym.Wrapper):
    """Replace environment rewards with IRL-recovered rewards.

    Wraps a Gymnasium environment so that step() returns the reward
    predicted by a fitted NeuralGLADIUS or NeuralAIRL model instead
    of the true environment reward.
    """

    def __init__(self, env, model, norm_stats, n_actions, model_type="gladius"):
        super().__init__(env)
        self.model = model
        self.obs_min = norm_stats["obs_min"]
        self.obs_range = norm_stats["obs_range"]
        self.n_actions = n_actions
        self.model_type = model_type
        self._last_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        obs, true_reward, terminated, truncated, info = self.env.step(action)

        # Compute IRL reward from the previous observation and action
        obs_normed = (self._last_obs - self.obs_min) / self.obs_range
        obs_normed = np.clip(obs_normed, 0.0, 1.0)

        with torch.no_grad():
            s_feat = torch.tensor(obs_normed, dtype=torch.float32).unsqueeze(0)
            ctx_feat = self.model._context_encoder(
                torch.zeros(1, dtype=torch.long)
            )

            if self.model_type == "gladius":
                # Implied reward: r = Q(s,a) - beta * EV(s,a)
                a_onehot = torch.zeros(1, self.n_actions)
                a_onehot[0, action] = 1.0
                q_val = self.model._q_net(s_feat, ctx_feat, a_onehot)
                ev_val = self.model._ev_net(s_feat, ctx_feat, a_onehot)
                irl_reward = float(
                    (q_val - self.model.discount * ev_val).item()
                )
            else:
                # AIRL reward: g(s,a)
                a_onehot = torch.zeros(1, self.n_actions)
                a_onehot[0, action] = 1.0
                irl_reward = float(
                    self.model._reward_net(s_feat, ctx_feat, a_onehot).item()
                )

        self._last_obs = obs
        return obs, irl_reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Estimator fitting functions
# ---------------------------------------------------------------------------

def fit_gladius(df, state_encoder, obs_dim, n_actions, max_epochs, patience):
    """Fit NeuralGLADIUS and return the fitted model with timing."""
    model = NeuralGLADIUS(
        n_actions=n_actions,
        discount=0.99,
        scale=1.0,
        q_hidden_dim=64,
        q_num_layers=2,
        ev_hidden_dim=64,
        ev_num_layers=2,
        batch_size=512,
        max_epochs=max_epochs,
        lr=5e-4,
        bellman_weight=5.0,
        patience=patience,
        alternating_updates=True,
        lr_decay_rate=0.0005,
        state_encoder=state_encoder,
        state_dim=obs_dim,
        verbose=False,
    )
    t0 = time.time()
    model.fit(data=df, state="state", action="action", id="agent_id")
    elapsed = time.time() - t0
    return model, elapsed


def fit_airl(df, state_encoder, obs_dim, n_actions, max_epochs, patience):
    """Fit NeuralAIRL and return the fitted model with timing."""
    model = NeuralAIRL(
        n_actions=n_actions,
        discount=0.99,
        reward_hidden_dim=64,
        reward_num_layers=2,
        shaping_hidden_dim=64,
        shaping_num_layers=2,
        policy_hidden_dim=64,
        policy_num_layers=2,
        batch_size=512,
        max_epochs=max_epochs,
        disc_lr=1e-3,
        policy_lr=1e-3,
        disc_steps=5,
        patience=patience,
        state_encoder=state_encoder,
        state_dim=obs_dim,
        verbose=False,
    )
    t0 = time.time()
    model.fit(data=df, state="state", action="action", id="agent_id")
    elapsed = time.time() - t0
    return model, elapsed


# ---------------------------------------------------------------------------
# Forward RL evaluation
# ---------------------------------------------------------------------------

def run_forward_rl(model, env_name, norm_stats, n_actions, model_type, seed):
    """Train PPO on IRL-recovered rewards, evaluate on true rewards.

    Returns (mean_reward, std_reward) on the true environment, or
    (None, None) if forward RL is not available.
    """
    if not FORWARD_RL_AVAILABLE:
        return None, None

    def env_factory():
        base_env = gym.make(env_name)
        return IRLRewardWrapper(
            base_env, model, norm_stats, n_actions, model_type
        )

    config = PPOConfig(
        max_timesteps=100_000,
        n_steps=2048,
        hidden_dim=64,
        num_layers=2,
        verbose=False,
    )
    ppo_model = train_ppo(env_name, config=config, seed=seed,
                          env_factory=env_factory)
    mean_r, std_r = evaluate_policy(env_name, ppo_model, n_episodes=50,
                                    seed=seed + 10000)
    return mean_r, std_r


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark():
    results = []

    for env_name, env_cfg in ENV_CONFIGS.items():
        obs_dim = env_cfg["obs_dim"]
        n_actions = env_cfg["n_actions"]
        n_bins = env_cfg["n_bins"]

        # Check if expert data exists
        try:
            test_data = load_expert_data(env_name)
        except FileNotFoundError as e:
            print(f"SKIP {env_name}: {e}")
            continue

        print(f"\n{'='*70}")
        print(f"Environment: {env_name}")
        print(f"  obs_dim={obs_dim}, n_actions={n_actions}, n_bins={n_bins}")
        print(f"  Expert avg reward: {test_data['expert_avg_reward']:.1f}")
        print(f"{'='*70}")

        expert_avg_reward = test_data["expert_avg_reward"]

        for n_traj in DATA_CONDITIONS:
            # Determine epoch/patience budget
            max_epochs = 500 if n_traj >= 15 else 300
            patience = 100 if n_traj >= 15 else 50

            print(f"\n  --- {n_traj} trajectories (epochs={max_epochs}, "
                  f"patience={patience}) ---")

            for seed in SEEDS:
                # Set seeds for reproducibility
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Load and subset data
                data = load_expert_data(env_name, n_trajectories=n_traj)
                df, encoder, n_states, norm_stats = build_dataframe_and_encoder(
                    data["observations"],
                    data["next_observations"],
                    data["actions"],
                    data["episode_ids"],
                    n_bins,
                )

                n_transitions = len(data["actions"])
                print(f"\n  seed={seed}, transitions={n_transitions}, "
                      f"states={n_states}")

                # --- GLADIUS ---
                print(f"    Fitting GLADIUS...", end="", flush=True)
                gladius_model, gladius_time = fit_gladius(
                    df, encoder, obs_dim, n_actions, max_epochs, patience
                )

                # Compute action accuracy on training data
                state_ids = df["state"].values
                gladius_proba = gladius_model.predict_proba(state_ids)
                gladius_pred = gladius_proba.argmax(axis=1)
                gladius_acc = float(
                    (gladius_pred == data["actions"]).mean()
                )

                gladius_mean_r, gladius_std_r = evaluate_neural_policy(
                    gladius_model, env_name, n_actions, norm_stats,
                    n_episodes=50, model_type="gladius",
                )
                gladius_norm_r = gladius_mean_r / abs(expert_avg_reward)
                print(f" done ({gladius_time:.1f}s) | "
                      f"acc={gladius_acc:.1%} | "
                      f"reward={gladius_mean_r:.1f} | "
                      f"norm={gladius_norm_r:.2f}")

                # --- AIRL ---
                print(f"    Fitting AIRL...", end="", flush=True)
                airl_model, airl_time = fit_airl(
                    df, encoder, obs_dim, n_actions, max_epochs, patience
                )

                airl_proba = airl_model.predict_proba(state_ids)
                airl_pred = airl_proba.argmax(axis=1)
                airl_acc = float((airl_pred == data["actions"]).mean())

                airl_mean_r, airl_std_r = evaluate_neural_policy(
                    airl_model, env_name, n_actions, norm_stats,
                    n_episodes=50, model_type="airl",
                )
                airl_norm_r = airl_mean_r / abs(expert_avg_reward)
                print(f" done ({airl_time:.1f}s) | "
                      f"acc={airl_acc:.1%} | "
                      f"reward={airl_mean_r:.1f} | "
                      f"norm={airl_norm_r:.2f}")

                # --- Forward RL ---
                gladius_frl_mean, gladius_frl_std = None, None
                airl_frl_mean, airl_frl_std = None, None

                if FORWARD_RL and FORWARD_RL_AVAILABLE:
                    print(f"    Forward RL (GLADIUS)...", end="", flush=True)
                    gladius_frl_mean, gladius_frl_std = run_forward_rl(
                        gladius_model, env_name, norm_stats,
                        n_actions, "gladius", seed,
                    )
                    if gladius_frl_mean is not None:
                        print(f" {gladius_frl_mean:.1f} +/- "
                              f"{gladius_frl_std:.1f}")
                    else:
                        print(" skipped")

                    print(f"    Forward RL (AIRL)...", end="", flush=True)
                    airl_frl_mean, airl_frl_std = run_forward_rl(
                        airl_model, env_name, norm_stats,
                        n_actions, "airl", seed,
                    )
                    if airl_frl_mean is not None:
                        print(f" {airl_frl_mean:.1f} +/- "
                              f"{airl_frl_std:.1f}")
                    else:
                        print(" skipped")

                # Store results
                result = {
                    "env": env_name,
                    "n_trajectories": n_traj,
                    "seed": seed,
                    "n_transitions": n_transitions,
                    "expert_avg_reward": expert_avg_reward,
                    "gladius": {
                        "action_accuracy": gladius_acc,
                        "mean_reward": gladius_mean_r,
                        "std_reward": gladius_std_r,
                        "normalized_reward": gladius_norm_r,
                        "time_s": gladius_time,
                        "epochs": gladius_model.n_epochs_,
                        "converged": gladius_model.converged_,
                        "forward_rl_mean": gladius_frl_mean,
                        "forward_rl_std": gladius_frl_std,
                    },
                    "airl": {
                        "action_accuracy": airl_acc,
                        "mean_reward": airl_mean_r,
                        "std_reward": airl_std_r,
                        "normalized_reward": airl_norm_r,
                        "time_s": airl_time,
                        "epochs": airl_model.n_epochs_,
                        "converged": airl_model.converged_,
                        "forward_rl_mean": airl_frl_mean,
                        "forward_rl_std": airl_frl_std,
                    },
                }
                results.append(result)

    # --- Save results ---
    out_dir = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "gym-irl-benchmark"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # --- Print summary table ---
    print_summary(results)

    return results


def print_summary(results):
    """Print a formatted summary table of all results."""
    print("\n" + "=" * 90)
    print("SUMMARY: GLADIUS vs AIRL on Gym Environments")
    print("=" * 90)

    # Group by (env, n_traj) and average over seeds
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        key = (r["env"], r["n_trajectories"])
        grouped[key].append(r)

    header = (
        f"{'Env':<18} {'N':>3} | "
        f"{'GLADIUS Acc':>10} {'Reward':>8} {'Norm':>6} | "
        f"{'AIRL Acc':>10} {'Reward':>8} {'Norm':>6}"
    )
    if any(r["gladius"]["forward_rl_mean"] is not None for r in results):
        header += f" | {'G-FRL':>7} {'A-FRL':>7}"
    print(header)
    print("-" * len(header))

    for (env, n_traj), runs in sorted(grouped.items()):
        g_acc = np.mean([r["gladius"]["action_accuracy"] for r in runs])
        g_rew = np.mean([r["gladius"]["mean_reward"] for r in runs])
        g_nrm = np.mean([r["gladius"]["normalized_reward"] for r in runs])
        a_acc = np.mean([r["airl"]["action_accuracy"] for r in runs])
        a_rew = np.mean([r["airl"]["mean_reward"] for r in runs])
        a_nrm = np.mean([r["airl"]["normalized_reward"] for r in runs])

        line = (
            f"{env:<18} {n_traj:>3} | "
            f"{g_acc:>10.1%} {g_rew:>8.1f} {g_nrm:>6.2f} | "
            f"{a_acc:>10.1%} {a_rew:>8.1f} {a_nrm:>6.2f}"
        )

        g_frl_vals = [r["gladius"]["forward_rl_mean"] for r in runs
                      if r["gladius"]["forward_rl_mean"] is not None]
        a_frl_vals = [r["airl"]["forward_rl_mean"] for r in runs
                      if r["airl"]["forward_rl_mean"] is not None]
        if g_frl_vals or a_frl_vals:
            g_frl = np.mean(g_frl_vals) if g_frl_vals else float("nan")
            a_frl = np.mean(a_frl_vals) if a_frl_vals else float("nan")
            line += f" | {g_frl:>7.1f} {a_frl:>7.1f}"

        print(line)

    print()


if __name__ == "__main__":
    run_benchmark()
