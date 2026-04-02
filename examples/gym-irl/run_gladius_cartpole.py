#!/usr/bin/env python3
"""
GLADIUS IRL on CartPole Expert Demonstrations
==============================================

Recovers rewards from a CartPole expert policy using NeuralGLADIUS,
then evaluates the recovered rewards against the true simulator
rewards and trains a greedy policy from the recovered Q-values.

This demonstrates GLADIUS on a continuous-state environment where
traditional tabular methods cannot be applied directly.

Prerequisites:
    python scripts/generate_gym_expert_data.py

Usage:
    python examples/gym-irl/run_gladius_cartpole.py
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
    "cartpole_expert.npz",
)
if not os.path.exists(DATA_PATH):
    print("Expert data not found. Generate it first:")
    print("  python scripts/generate_gym_expert_data.py")
    sys.exit(1)

data = np.load(DATA_PATH)
observations = data["observations"]     # (N, 4)
actions = data["actions"]               # (N,)
next_observations = data["next_observations"]  # (N, 4)
rewards = data["rewards"]               # (N,)
episode_ids = data["episode_ids"]       # (N,)

N_OBS = observations.shape[0]
OBS_DIM = observations.shape[1]
N_ACTIONS = 2
N_EPISODES = episode_ids.max() + 1

print(f"CartPole-v1 expert data: {N_OBS} transitions from "
      f"{N_EPISODES} episodes")
print(f"Observation dim: {OBS_DIM}, Actions: {N_ACTIONS}")
print(f"Expert avg reward per episode: "
      f"{rewards.sum() / N_EPISODES:.1f}")
print()

# ── Build DataFrame for NeuralGLADIUS ──
# Assign each unique observation a state index via approximate binning.
# NeuralGLADIUS uses a custom state_encoder to map indices back to
# raw observations, so the binning resolution doesn't limit accuracy.
# Use coarse bins to keep the state space manageable (6^4 = 1296 bins).
N_BINS_PER_DIM = 6
obs_min = observations.min(axis=0)
obs_max = observations.max(axis=0)
obs_range = np.maximum(obs_max - obs_min, 1e-8)

# Bin each dimension and create a composite state index
bins = ((observations - obs_min) / obs_range * (N_BINS_PER_DIM - 1)).astype(int)
bins = np.clip(bins, 0, N_BINS_PER_DIM - 1)
state_ids = np.ravel_multi_index(
    bins.T, [N_BINS_PER_DIM] * OBS_DIM
)

next_bins = ((next_observations - obs_min) / obs_range * (N_BINS_PER_DIM - 1)).astype(int)
next_bins = np.clip(next_bins, 0, N_BINS_PER_DIM - 1)
next_state_ids = np.ravel_multi_index(
    next_bins.T, [N_BINS_PER_DIM] * OBS_DIM
)

# Store the observation vectors indexed by state_id for the encoder
n_states = max(state_ids.max(), next_state_ids.max()) + 1
state_to_obs = np.zeros((n_states, OBS_DIM), dtype=np.float32)
for i, sid in enumerate(state_ids):
    state_to_obs[sid] = observations[i]
for i, sid in enumerate(next_state_ids):
    state_to_obs[sid] = next_observations[i]

# Normalize observations to [0, 1] for the encoder
obs_global_min = state_to_obs.min(axis=0)
obs_global_range = np.maximum(state_to_obs.max(axis=0) - obs_global_min, 1e-8)
state_to_obs_normed = (state_to_obs - obs_global_min) / obs_global_range
obs_lookup = torch.tensor(state_to_obs_normed, dtype=torch.float32)


def state_encoder(state_indices: torch.Tensor) -> torch.Tensor:
    """Map state indices to normalized observation vectors."""
    return obs_lookup[state_indices.long()]


# Build DataFrame (NeuralGLADIUS expects state/action/id columns)
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

print("Fitting NeuralGLADIUS (alternating updates, LR decay)...")
model = NeuralGLADIUS(
    n_actions=N_ACTIONS,
    discount=0.99,
    scale=1.0,
    q_hidden_dim=64,
    q_num_layers=2,
    ev_hidden_dim=64,
    ev_num_layers=2,
    batch_size=512,
    max_epochs=500,
    lr=5e-4,
    bellman_weight=5.0,
    patience=100,
    alternating_updates=True,
    lr_decay_rate=0.0005,
    state_encoder=state_encoder,
    state_dim=OBS_DIM,
    verbose=True,
)

model.fit(data=df, state="state", action="action", id="agent_id")
print()

# ── Evaluate recovered rewards ──
print("=" * 60)
print("Results")
print("=" * 60)

# 1. Policy accuracy: compare predicted vs expert actions
print(f"\nEpochs trained: {model.n_epochs_}")
print(f"Converged: {model.converged_}")

# Predict actions for all observed states
predicted_proba = model.predict_proba(state_ids)
predicted_actions = predicted_proba.argmax(axis=1)
accuracy = (predicted_actions == actions).mean()
print(f"Action prediction accuracy: {accuracy:.1%}")

# 2. Reward analysis
predicted_rewards = model.predict_reward(
    torch.tensor(state_ids, dtype=torch.long),
    torch.tensor(actions, dtype=torch.long),
).numpy()

# CartPole has constant reward of 1.0 per step, so correlation is
# not meaningful. Instead report reward statistics.
print(f"Recovered reward: mean={predicted_rewards.mean():.4f}, "
      f"std={predicted_rewards.std():.4f}")
print(f"True reward: constant 1.0 per step (survival reward)")

# 3. Evaluate the greedy policy in the simulator.
# Use the Q-network directly on raw observations rather than the
# tabular policy_ table, because the tabular lookup fails for
# unseen state bins. The neural network generalizes to new states.
print("\nEvaluating recovered policy in CartPole-v1...")
env = gym.make("CartPole-v1")
n_eval_episodes = 50
eval_rewards = []

model._q_net.eval()
for ep in range(n_eval_episodes):
    obs, _ = env.reset(seed=1000 + ep)
    episode_reward = 0.0

    for step in range(500):
        # Normalize observation to [0,1] using training data statistics
        obs_normed = (obs - obs_global_min) / obs_global_range
        obs_normed = np.clip(obs_normed, 0.0, 1.0)

        # Feed through Q-network to get action values
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
print(f"Recovered policy: {avg_eval:.1f} +/- {std_eval:.1f} "
      f"(expert: 500.0)")
print(f"Optimality: {avg_eval / 500.0:.1%}")
print()
