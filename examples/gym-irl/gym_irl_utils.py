"""Shared utilities for Gym IRL benchmarks.

Consolidates the data loading, state binning, encoder construction, and
policy evaluation logic that was previously duplicated across the
individual environment scripts (run_gladius_cartpole.py, etc.).
"""

from __future__ import annotations

import os

import gymnasium as gym
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_expert_data(
    env_name: str,
    data_dir: str | None = None,
    n_trajectories: int | None = None,
) -> dict:
    """Load expert demonstration data for a Gym environment.

    Parameters
    ----------
    env_name : str
        Gymnasium environment ID. Mapped to filenames:
        CartPole-v1 -> cartpole_expert.npz, etc.
    data_dir : str, optional
        Path to directory containing .npz files. Defaults to
        <project_root>/data/gym-experts/.
    n_trajectories : int, optional
        If provided, subset to the first N episodes.

    Returns
    -------
    dict with keys:
        observations, actions, next_observations, rewards, episode_ids,
        dones (all numpy arrays), plus metadata: obs_dim, n_actions,
        n_episodes, expert_avg_reward.
    """
    name_map = {
        "CartPole-v1": "cartpole_expert.npz",
        "Acrobot-v1": "acrobot_expert.npz",
        "LunarLander-v3": "lunarlander_expert.npz",
    }
    if data_dir is None:
        data_dir = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "gym-experts"
        )

    filename = name_map.get(env_name)
    if filename is None:
        raise ValueError(f"No expert data mapping for {env_name}")

    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expert data not found at {path}. "
            f"Generate it first: python scripts/generate_gym_expert_data.py"
        )

    data = np.load(path)
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    episode_ids = data["episode_ids"]
    dones = data["dones"]

    # Subset to first N trajectories if requested
    if n_trajectories is not None:
        mask = episode_ids < n_trajectories
        observations = observations[mask]
        actions = actions[mask]
        next_observations = next_observations[mask]
        rewards = rewards[mask]
        episode_ids = episode_ids[mask]
        dones = dones[mask]

    n_episodes = int(episode_ids.max()) + 1
    obs_dim = observations.shape[1]

    # Infer n_actions from the environment spec
    action_counts = {
        "CartPole-v1": 2,
        "Acrobot-v1": 3,
        "LunarLander-v3": 4,
    }
    n_actions = action_counts.get(env_name, int(actions.max()) + 1)

    expert_avg_reward = float(rewards.sum() / n_episodes)

    return {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "episode_ids": episode_ids,
        "dones": dones,
        "obs_dim": obs_dim,
        "n_actions": n_actions,
        "n_episodes": n_episodes,
        "expert_avg_reward": expert_avg_reward,
    }


# ---------------------------------------------------------------------------
# State binning and encoder construction
# ---------------------------------------------------------------------------


def build_dataframe_and_encoder(
    observations: np.ndarray,
    next_observations: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    n_bins_per_dim: int,
) -> tuple[pd.DataFrame, callable, int, dict]:
    """Bin continuous observations into discrete states and build a
    state encoder for neural estimators.

    Parameters
    ----------
    observations : array of shape (N, obs_dim)
    next_observations : array of shape (N, obs_dim)
    actions : array of shape (N,)
    episode_ids : array of shape (N,)
    n_bins_per_dim : int
        Number of bins per observation dimension.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns: agent_id, state, action, reward (reward=0).
    state_encoder : callable
        Maps state index tensors to normalized observation vectors.
    n_states : int
        Total number of unique state bins.
    norm_stats : dict
        Contains obs_min and obs_range for normalizing raw observations
        at evaluation time.
    """
    obs_dim = observations.shape[1]
    obs_min = observations.min(axis=0)
    obs_max = observations.max(axis=0)
    obs_range = np.maximum(obs_max - obs_min, 1e-8)

    # Bin each dimension
    bins = ((observations - obs_min) / obs_range * (n_bins_per_dim - 1)).astype(int)
    bins = np.clip(bins, 0, n_bins_per_dim - 1)
    state_ids = np.ravel_multi_index(bins.T, [n_bins_per_dim] * obs_dim)

    next_bins = ((next_observations - obs_min) / obs_range * (n_bins_per_dim - 1)).astype(int)
    next_bins = np.clip(next_bins, 0, n_bins_per_dim - 1)
    next_state_ids = np.ravel_multi_index(
        next_bins.T, [n_bins_per_dim] * obs_dim
    )

    # Build observation lookup table indexed by state_id
    n_states = max(int(state_ids.max()), int(next_state_ids.max())) + 1
    state_to_obs = np.zeros((n_states, obs_dim), dtype=np.float32)
    for i, sid in enumerate(state_ids):
        state_to_obs[sid] = observations[i]
    for i, sid in enumerate(next_state_ids):
        state_to_obs[sid] = next_observations[i]

    # Normalize to [0, 1]
    obs_global_min = state_to_obs.min(axis=0)
    obs_global_range = np.maximum(
        state_to_obs.max(axis=0) - obs_global_min, 1e-8
    )
    state_to_obs_normed = (state_to_obs - obs_global_min) / obs_global_range
    obs_lookup = state_to_obs_normed.astype(np.float32, copy=False)

    def state_encoder(state_indices) -> np.ndarray:
        state_idx = np.asarray(state_indices, dtype=np.int32)
        return obs_lookup[state_idx]

    df = pd.DataFrame({
        "agent_id": episode_ids,
        "state": state_ids,
        "action": actions,
        "reward": np.zeros(len(actions), dtype=np.float32),
    })

    norm_stats = {
        "obs_min": obs_global_min,
        "obs_range": obs_global_range,
    }

    return df, state_encoder, n_states, norm_stats


# ---------------------------------------------------------------------------
# Policy evaluation in the simulator
# ---------------------------------------------------------------------------


def evaluate_neural_policy(
    model,
    env_name: str,
    n_actions: int,
    norm_stats: dict,
    n_episodes: int = 100,
    model_type: str = "gladius",
    seed: int = 1000,
) -> tuple[float, float]:
    """Evaluate a fitted NeuralGLADIUS or NeuralAIRL policy in the simulator.

    Uses the neural network directly on raw (normalized) observations
    rather than the tabular policy table, for generalization to unseen
    states.

    Parameters
    ----------
    model : NeuralGLADIUS or NeuralAIRL
        Fitted estimator with internal networks.
    env_name : str
        Gymnasium environment ID.
    n_actions : int
        Number of discrete actions.
    norm_stats : dict
        Contains obs_min and obs_range from training data.
    n_episodes : int
        Number of evaluation episodes.
    model_type : str
        "gladius" or "airl", determines which internal network to query.
    seed : int
        Random seed for environment resets.

    Returns
    -------
    mean_reward : float
    std_reward : float
    """
    obs_min = norm_stats["obs_min"]
    obs_range = norm_stats["obs_range"]

    env = gym.make(env_name)
    eval_rewards = []

    # Put networks in eval mode
    if model_type == "gladius":
        model._q_net.eval()
    else:
        model._policy_net.eval()

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0

        for step in range(1000):
            obs_normed = (obs - obs_min) / obs_range
            obs_normed = np.clip(obs_normed, 0.0, 1.0)

            s_feat = np.asarray(obs_normed, dtype=np.float32)[None, :]
            ctx_feat = model._context_encoder(np.zeros(1, dtype=np.int32))

            if model_type == "gladius":
                q_vals = np.asarray(
                    model._q_net.all_actions(s_feat, ctx_feat, n_actions)
                )
                action = int(np.argmax(q_vals, axis=1)[0])
            else:
                probs = np.asarray(model._policy_net(s_feat, ctx_feat))
                action = int(np.argmax(probs, axis=1)[0])

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        eval_rewards.append(episode_reward)

    env.close()

    return float(np.mean(eval_rewards)), float(np.std(eval_rewards))
