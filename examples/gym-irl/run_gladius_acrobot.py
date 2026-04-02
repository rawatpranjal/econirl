#!/usr/bin/env python3
"""
GLADIUS IRL on Acrobot Expert Demonstrations
=============================================

Recovers rewards from an Acrobot heuristic expert policy using
NeuralGLADIUS, then evaluates the recovered policy in the simulator.

Acrobot has 6-dimensional continuous state (cos/sin of two joint angles
plus angular velocities) and 3 discrete actions (negative torque, zero,
positive torque). This is a harder IRL problem than CartPole because
the expert policy is suboptimal (heuristic, not RL-trained).

Prerequisites:
    python scripts/generate_gym_expert_data.py

Usage:
    python examples/gym-irl/run_gladius_acrobot.py
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

# ── Load expert data ──
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "gym-experts",
    "acrobot_expert.npz",
)
if not os.path.exists(DATA_PATH):
    print("Expert data not found. Generate it first:")
    print("  python scripts/generate_gym_expert_data.py")
    sys.exit(1)

data = np.load(DATA_PATH)
observations = data["observations"]
actions = data["actions"]
next_observations = data["next_observations"]
rewards = data["rewards"]
episode_ids = data["episode_ids"]

N_OBS = observations.shape[0]
OBS_DIM = observations.shape[1]
N_ACTIONS = 3
N_EPISODES = episode_ids.max() + 1

print(f"Acrobot-v1 expert data: {N_OBS} transitions from "
      f"{N_EPISODES} episodes")
print(f"Observation dim: {OBS_DIM}, Actions: {N_ACTIONS}")
print(f"Expert avg reward per episode: "
      f"{rewards.sum() / N_EPISODES:.1f}")
print()

# ── Build DataFrame ──
N_BINS_PER_DIM = 10
obs_min = observations.min(axis=0)
obs_max = observations.max(axis=0)
obs_range = np.maximum(obs_max - obs_min, 1e-8)

bins = ((observations - obs_min) / obs_range * (N_BINS_PER_DIM - 1)).astype(int)
bins = np.clip(bins, 0, N_BINS_PER_DIM - 1)
state_ids = np.ravel_multi_index(bins.T, [N_BINS_PER_DIM] * OBS_DIM)

next_bins = ((next_observations - obs_min) / obs_range * (N_BINS_PER_DIM - 1)).astype(int)
next_bins = np.clip(next_bins, 0, N_BINS_PER_DIM - 1)
next_state_ids = np.ravel_multi_index(
    next_bins.T, [N_BINS_PER_DIM] * OBS_DIM
)

n_states = max(state_ids.max(), next_state_ids.max()) + 1
state_to_obs = np.zeros((n_states, OBS_DIM), dtype=np.float32)
for i, sid in enumerate(state_ids):
    state_to_obs[sid] = observations[i]
for i, sid in enumerate(next_state_ids):
    state_to_obs[sid] = next_observations[i]

obs_global_min = state_to_obs.min(axis=0)
obs_global_range = np.maximum(state_to_obs.max(axis=0) - obs_global_min, 1e-8)
state_to_obs_normed = (state_to_obs - obs_global_min) / obs_global_range
obs_lookup = torch.tensor(state_to_obs_normed, dtype=torch.float32)


def state_encoder(state_indices: torch.Tensor) -> torch.Tensor:
    """Map state indices to normalized observation vectors."""
    return obs_lookup[state_indices.long()]


df = pd.DataFrame({
    "agent_id": episode_ids,
    "state": state_ids,
    "action": actions,
    "reward": rewards,
})

print(f"Binned into {n_states} unique states")
print()

# ── Fit NeuralGLADIUS ──
from econirl.estimators.neural_gladius import NeuralGLADIUS

print("Fitting NeuralGLADIUS...")
model = NeuralGLADIUS(
    n_actions=N_ACTIONS,
    discount=0.99,
    scale=1.0,
    q_hidden_dim=64,
    q_num_layers=2,
    ev_hidden_dim=64,
    ev_num_layers=2,
    batch_size=256,
    max_epochs=300,
    lr=1e-3,
    bellman_weight=1.0,
    patience=30,
    alternating_updates=True,
    lr_decay_rate=0.001,
    state_encoder=state_encoder,
    state_dim=OBS_DIM,
    verbose=True,
)

model.fit(data=df, state="state", action="action", id="agent_id")
print()

# ── Evaluate ──
print("=" * 60)
print("Results")
print("=" * 60)

print(f"\nEpochs trained: {model.n_epochs_}")
print(f"Converged: {model.converged_}")

predicted_proba = model.predict_proba(state_ids)
predicted_actions = predicted_proba.argmax(axis=1)
accuracy = (predicted_actions == actions).mean()
print(f"Action prediction accuracy: {accuracy:.1%}")

predicted_rewards = model.predict_reward(
    torch.tensor(state_ids, dtype=torch.long),
    torch.tensor(actions, dtype=torch.long),
).numpy()
corr = np.corrcoef(predicted_rewards, rewards)[0, 1]
print(f"Reward correlation (recovered vs true): {corr:.4f}")

# Evaluate in simulator using Q-network directly for generalization
obs_global_min = state_to_obs.min(axis=0)
obs_global_range = np.maximum(state_to_obs.max(axis=0) - obs_global_min, 1e-8)

print("\nEvaluating recovered policy in Acrobot-v1...")
env = gym.make("Acrobot-v1")
n_eval_episodes = 50
eval_rewards = []

model._q_net.eval()
for ep in range(n_eval_episodes):
    obs, _ = env.reset(seed=1000 + ep)
    episode_reward = 0.0

    for step in range(500):
        obs_normed = (obs - obs_global_min) / obs_global_range
        obs_normed = np.clip(obs_normed, 0.0, 1.0)

        with torch.no_grad():
            s_feat = torch.tensor(
                obs_normed, dtype=torch.float32
            ).unsqueeze(0)
            ctx_feat = model._context_encoder(
                torch.zeros(1, dtype=torch.long)
            )
            q_vals = model._q_net.all_actions(
                s_feat, ctx_feat, N_ACTIONS
            )
            action = int(q_vals.argmax(dim=1).item())

        obs, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        if terminated or truncated:
            break

    eval_rewards.append(episode_reward)

env.close()

avg_eval = np.mean(eval_rewards)
std_eval = np.std(eval_rewards)
expert_avg = rewards.sum() / N_EPISODES
print(f"Recovered policy: {avg_eval:.1f} +/- {std_eval:.1f} "
      f"(expert heuristic: {expert_avg:.1f})")
print()
