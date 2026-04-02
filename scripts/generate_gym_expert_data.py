#!/usr/bin/env python3
"""
Generate expert trajectory data for Gym IRL examples.

Creates .npz files with expert demonstrations from gymnasium environments.
Uses hand-coded heuristic policies for CartPole and Acrobot (no external
dependencies beyond gymnasium). For LunarLander, optionally uses
Stable-Baselines3 pre-trained PPO models.

Usage:
    python scripts/generate_gym_expert_data.py

Output:
    data/gym-experts/cartpole_expert.npz
    data/gym-experts/acrobot_expert.npz
    data/gym-experts/lunarlander_expert.npz  (requires SB3 + Box2D)
"""

import os

import gymnasium as gym
import numpy as np


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


def acrobot_expert_policy(obs):
    """Energy-pumping heuristic for Acrobot.

    Not truly optimal, but generates reasonable demonstrations.
    Acrobot-v1 expert policy based on angular velocity direction.
    """
    # obs: [cos(theta1), sin(theta1), cos(theta2), sin(theta2),
    #       theta1_dot, theta2_dot]
    theta1_dot = obs[4]
    theta2_dot = obs[5]
    sin_theta1 = obs[1]
    cos_theta1 = obs[0]

    # Swing in the direction that increases energy
    # Torque direction based on angular velocity of link 2
    # relative to link 1 and current position
    if theta2_dot > 0:
        return 2  # positive torque
    elif theta2_dot < 0:
        return 0  # negative torque
    else:
        # When stationary, push based on position
        return 2 if sin_theta1 > 0 else 0


def collect_trajectories(env_name, policy_fn, n_trajectories=100,
                         max_steps=500, seed=42):
    """Collect expert trajectories from a gymnasium environment.

    Returns arrays of (observations, actions, next_observations, rewards,
    episode_ids, dones).
    """
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


def try_lunarlander():
    """Try to generate LunarLander data using SB3 pre-trained PPO."""
    try:
        from stable_baselines3 import PPO
        from huggingface_sb3 import load_from_hub
    except ImportError:
        print("  Skipping LunarLander (requires stable-baselines3 and "
              "huggingface-sb3). Install with:")
        print("    pip install stable-baselines3 huggingface-sb3 "
              '"gymnasium[box2d]"')
        return None

    try:
        checkpoint = load_from_hub(
            repo_id="sb3/ppo-LunarLander-v2",
            filename="ppo-LunarLander-v2.zip",
        )
        model = PPO.load(checkpoint)
    except Exception as e:
        print(f"  Skipping LunarLander (failed to load model: {e})")
        return None

    env = gym.make("LunarLander-v2")
    all_obs = []
    all_actions = []
    all_next_obs = []
    all_rewards = []
    all_episode_ids = []
    all_dones = []
    total_reward = 0.0

    for ep in range(100):
        obs, info = env.reset(seed=42 + ep)
        episode_reward = 0.0

        for step in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
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
    avg_reward = total_reward / 100
    print(f"  LunarLander-v2: 100 episodes, avg reward = {avg_reward:.1f}")

    return {
        "observations": np.array(all_obs, dtype=np.float32),
        "actions": np.array(all_actions, dtype=np.int32),
        "next_observations": np.array(all_next_obs, dtype=np.float32),
        "rewards": np.array(all_rewards, dtype=np.float32),
        "episode_ids": np.array(all_episode_ids, dtype=np.int32),
        "dones": np.array(all_dones, dtype=bool),
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating expert trajectory data...")
    print()

    # CartPole
    print("CartPole-v1:")
    cartpole_data = collect_trajectories(
        "CartPole-v1", cartpole_expert_policy,
        n_trajectories=100, max_steps=500, seed=42,
    )
    path = os.path.join(DATA_DIR, "cartpole_expert.npz")
    np.savez_compressed(path, **cartpole_data)
    print(f"  Saved to {path} ({cartpole_data['observations'].shape[0]} transitions)")
    print()

    # Acrobot
    print("Acrobot-v1:")
    acrobot_data = collect_trajectories(
        "Acrobot-v1", acrobot_expert_policy,
        n_trajectories=100, max_steps=500, seed=42,
    )
    path = os.path.join(DATA_DIR, "acrobot_expert.npz")
    np.savez_compressed(path, **acrobot_data)
    print(f"  Saved to {path} ({acrobot_data['observations'].shape[0]} transitions)")
    print()

    # LunarLander (optional)
    print("LunarLander-v2:")
    ll_data = try_lunarlander()
    if ll_data is not None:
        path = os.path.join(DATA_DIR, "lunarlander_expert.npz")
        np.savez_compressed(path, **ll_data)
        print(f"  Saved to {path} ({ll_data['observations'].shape[0]} transitions)")
    print()

    print("Done.")


if __name__ == "__main__":
    main()
