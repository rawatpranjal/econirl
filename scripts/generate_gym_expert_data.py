#!/usr/bin/env python3
"""
Generate expert trajectory data for Gym IRL examples.

Creates .npz files with expert demonstrations from gymnasium environments.
Uses a hand-coded heuristic for CartPole (already optimal at 500/500) and
JAX PPO for Acrobot and LunarLander.

Usage:
    python scripts/generate_gym_expert_data.py

Output:
    data/gym-experts/cartpole_expert.npz
    data/gym-experts/acrobot_expert.npz
    data/gym-experts/lunarlander_expert.npz  (requires gymnasium[box2d])
"""

import os
import warnings

import gymnasium as gym
import numpy as np

warnings.filterwarnings("ignore", message="A JAX array is being set as static")

from econirl.rl.ppo import (
    ActorCritic,
    PPOConfig,
    collect_expert_data,
    evaluate_policy,
    train_ppo,
)


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "gym-experts")


def cartpole_expert_policy(obs):
    """Simple PD controller for CartPole that achieves near-optimal reward.

    Uses pole angle and angular velocity to decide push direction.
    Achieves 500/500 reward consistently.
    """
    # obs: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    angle = obs[2]
    angular_vel = obs[3]
    # Push right if pole is falling right, left if falling left.
    # The threshold is tuned for CartPole-v1.
    if angle + 0.25 * angular_vel > 0:
        return 1  # push right
    else:
        return 0  # push left


def collect_heuristic_trajectories(env_name, policy_fn, n_trajectories=100,
                                   max_steps=500, seed=42):
    """Collect expert trajectories using a heuristic policy."""
    env = gym.make(env_name)

    all_obs = []
    all_actions = []
    all_next_obs = []
    all_rewards = []
    all_episode_ids = []
    all_dones = []

    total_reward = 0.0
    for ep in range(n_trajectories):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0.0

        for step in range(max_steps):
            action = policy_fn(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            all_obs.append(obs.copy())
            all_actions.append(action)
            all_next_obs.append(next_obs.copy())
            all_rewards.append(reward)
            all_episode_ids.append(ep)
            all_dones.append(done)

            episode_reward += reward
            obs = next_obs

            if done:
                break

        total_reward += episode_reward

    env.close()

    avg_reward = total_reward / n_trajectories
    print(f"  {env_name}: {n_trajectories} episodes, "
          f"avg reward = {avg_reward:.1f}")

    return {
        "observations": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int32),
        "next_observations": np.array(all_next_obs, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "dones": np.array(all_dones, dtype=bool),
    }


def train_and_collect_ppo(env_name, max_timesteps, n_trajectories=100,
                          max_steps=1000, seed=42):
    """Train a PPO expert and collect demonstrations."""
    print(f"  Training PPO on {env_name} ({max_timesteps} timesteps)...")
    config = PPOConfig(
        max_timesteps=max_timesteps,
        n_steps=2048,
        verbose=True,
    )
    model = train_ppo(env_name, config=config, seed=seed)

    # Evaluate the trained expert
    mean_r, std_r = evaluate_policy(env_name, model, n_episodes=20, seed=0)
    print(f"  Expert quality: {mean_r:.1f} +/- {std_r:.1f}")

    # Collect demonstrations
    print(f"  Collecting {n_trajectories} trajectories...")
    data = collect_expert_data(
        env_name, model, n_trajectories=n_trajectories,
        max_steps=max_steps, seed=seed,
    )
    return data


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating expert trajectory data...")
    print()

    # CartPole: heuristic (already optimal)
    print("CartPole-v1 (heuristic):")
    cartpole_data = collect_heuristic_trajectories(
        "CartPole-v1", cartpole_expert_policy,
        n_trajectories=100, max_steps=500, seed=42,
    )
    path = os.path.join(DATA_DIR, "cartpole_expert.npz")
    np.savez_compressed(path, **cartpole_data)
    print(f"  Saved to {path} "
          f"({cartpole_data['observations'].shape[0]} transitions)")
    print()

    # Acrobot: PPO (200K steps reaches ~-80 avg reward)
    print("Acrobot-v1 (PPO):")
    acrobot_data = train_and_collect_ppo(
        "Acrobot-v1", max_timesteps=200_000,
        n_trajectories=100, max_steps=500, seed=42,
    )
    path = os.path.join(DATA_DIR, "acrobot_expert.npz")
    np.savez_compressed(path, **acrobot_data)
    print(f"  Saved to {path} "
          f"({acrobot_data['observations'].shape[0]} transitions)")
    print()

    # LunarLander: PPO (500K steps reaches ~200+ avg reward)
    print("LunarLander-v3 (PPO):")
    try:
        gym.make("LunarLander-v3")
    except Exception:
        print("  Skipping LunarLander (requires gymnasium[box2d]). "
              "Install with:")
        print('    pip install "gymnasium[box2d]"')
        print()
        print("Done.")
        return

    ll_data = train_and_collect_ppo(
        "LunarLander-v3", max_timesteps=500_000,
        n_trajectories=100, max_steps=1000, seed=42,
    )
    path = os.path.join(DATA_DIR, "lunarlander_expert.npz")
    np.savez_compressed(path, **ll_data)
    print(f"  Saved to {path} "
          f"({ll_data['observations'].shape[0]} transitions)")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
