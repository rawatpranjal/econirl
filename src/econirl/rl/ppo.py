"""Proximal Policy Optimization for discrete action spaces.

Minimal JAX/Equinox implementation of PPO (Schulman et al. 2017) for
training expert policies on Gymnasium environments. Used for:
1. Generating expert demonstration data for IRL benchmarks.
2. Forward RL evaluation (train on recovered rewards, test on true rewards).

Only supports discrete action spaces and continuous observation spaces.
Gymnasium environments must return numpy arrays from step/reset.

Reference:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import equinox as eqx
import optax


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class ActorCritic(eqx.Module):
    """MLP with shared trunk, separate policy-logit and value heads."""

    trunk: eqx.nn.MLP
    policy_head: eqx.nn.Linear
    value_head: eqx.nn.Linear

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        *,
        key: jax.Array,
    ):
        trunk_key, policy_key, value_key = jr.split(key, 3)
        self.trunk = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=max(num_layers - 1, 1),
            activation=jax.nn.tanh,
            key=trunk_key,
        )
        self.policy_head = eqx.nn.Linear(
            hidden_dim, n_actions, key=policy_key
        )
        self.value_head = eqx.nn.Linear(
            hidden_dim, 1, key=value_key
        )

    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass returning (logits, value).

        Parameters
        ----------
        obs : jax.Array
            Observation of shape (obs_dim,).

        Returns
        -------
        logits : jax.Array
            Action logits of shape (n_actions,).
        value : jax.Array
            Scalar state value of shape ().
        """
        x = self.trunk(obs)
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        return logits, value


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    n_steps: int = 2048
    n_epochs: int = 10
    mini_batch_size: int = 64
    max_timesteps: int = 500_000
    hidden_dim: int = 64
    num_layers: int = 2
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True
    verbose: bool = True


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation.

    Parameters
    ----------
    rewards : array of shape (n_steps,)
    values : array of shape (n_steps,)
    dones : array of shape (n_steps,)
    last_value : float
        Bootstrap value for the final state.
    gamma : float
        Discount factor.
    gae_lambda : float
        GAE lambda parameter.

    Returns
    -------
    advantages : array of shape (n_steps,)
    returns : array of shape (n_steps,)
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_value = last_value if t == n - 1 else values[t + 1]
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_ppo(
    env_name: str,
    config: PPOConfig | None = None,
    seed: int = 42,
    env_factory: Callable[[], gym.Env] | None = None,
) -> ActorCritic:
    """Train a PPO agent on a Gymnasium environment.

    Parameters
    ----------
    env_name : str
        Gymnasium environment ID (e.g. "CartPole-v1").
    config : PPOConfig, optional
        Training hyperparameters. Defaults to PPOConfig().
    seed : int
        Random seed for reproducibility.
    env_factory : callable, optional
        If provided, called instead of gym.make(env_name) to create the
        environment. Useful for reward wrappers in forward RL evaluation.

    Returns
    -------
    ActorCritic
        Trained actor-critic model.
    """
    if config is None:
        config = PPOConfig()

    # Create environment
    if env_factory is not None:
        env = env_factory()
    else:
        env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize model and optimizer
    key = jr.PRNGKey(seed)
    key, model_key = jr.split(key)
    model = ActorCritic(
        obs_dim, n_actions, config.hidden_dim, config.num_layers,
        key=model_key,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.lr),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # Rollout buffers (numpy for gym interaction)
    obs_buf = np.zeros((config.n_steps, obs_dim), dtype=np.float32)
    act_buf = np.zeros(config.n_steps, dtype=np.int32)
    rew_buf = np.zeros(config.n_steps, dtype=np.float32)
    done_buf = np.zeros(config.n_steps, dtype=np.float32)
    val_buf = np.zeros(config.n_steps, dtype=np.float32)
    logp_buf = np.zeros(config.n_steps, dtype=np.float32)

    # JIT-compiled forward pass
    @eqx.filter_jit
    def get_action_and_value(
        model: ActorCritic, obs: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        logits, value = model(obs)
        log_probs = jax.nn.log_softmax(logits)
        action = jr.categorical(key, logits)
        return action, log_probs[action], value

    @eqx.filter_jit
    def get_value(model: ActorCritic, obs: jax.Array) -> jax.Array:
        _, value = model(obs)
        return value

    # JIT-compiled PPO loss and update step
    @eqx.filter_jit
    def ppo_loss(
        model: ActorCritic,
        obs_batch: jax.Array,
        act_batch: jax.Array,
        old_logp_batch: jax.Array,
        adv_batch: jax.Array,
        ret_batch: jax.Array,
    ) -> jax.Array:
        # Vectorize over the batch
        logits, values = jax.vmap(model)(obs_batch)
        log_probs = jax.nn.log_softmax(logits)
        action_log_probs = log_probs[jnp.arange(len(act_batch)), act_batch]

        # Clipped surrogate objective
        ratio = jnp.exp(action_log_probs - old_logp_batch)
        clipped_ratio = jnp.clip(ratio, 1.0 - config.clip_epsilon,
                                  1.0 + config.clip_epsilon)
        policy_loss = -jnp.minimum(ratio * adv_batch,
                                    clipped_ratio * adv_batch).mean()

        # Value loss
        value_loss = ((values - ret_batch) ** 2).mean()

        # Entropy bonus
        probs = jax.nn.softmax(logits)
        entropy = -(probs * log_probs).sum(axis=-1).mean()

        return (policy_loss
                + config.value_coef * value_loss
                - config.entropy_coef * entropy)

    @eqx.filter_jit
    def update_step(
        model: ActorCritic,
        opt_state: optax.OptState,
        obs_batch: jax.Array,
        act_batch: jax.Array,
        old_logp_batch: jax.Array,
        adv_batch: jax.Array,
        ret_batch: jax.Array,
    ) -> tuple[ActorCritic, optax.OptState, jax.Array]:
        loss, grads = eqx.filter_value_and_grad(ppo_loss)(
            model, obs_batch, act_batch, old_logp_batch,
            adv_batch, ret_batch,
        )
        updates, opt_state = optimizer.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # Training loop
    obs, _ = env.reset(seed=seed)
    total_timesteps = 0
    episode_rewards = []
    current_episode_reward = 0.0

    while total_timesteps < config.max_timesteps:
        # Collect rollout
        for step in range(config.n_steps):
            obs_buf[step] = obs
            key, action_key = jr.split(key)
            action, logp, value = get_action_and_value(
                model, jnp.array(obs, dtype=jnp.float32), action_key
            )
            act_buf[step] = int(action)
            logp_buf[step] = float(logp)
            val_buf[step] = float(value)

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated
            rew_buf[step] = reward
            done_buf[step] = float(done)
            current_episode_reward += reward

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                obs, _ = env.reset()

        total_timesteps += config.n_steps

        # Bootstrap value for the last state
        last_value = float(get_value(
            model, jnp.array(obs, dtype=jnp.float32)
        ))

        # Compute GAE
        advantages, returns = compute_gae(
            rew_buf, val_buf, done_buf, last_value,
            config.gamma, config.gae_lambda,
        )

        if config.normalize_advantages:
            adv_std = advantages.std()
            if adv_std > 1e-8:
                advantages = (advantages - advantages.mean()) / adv_std

        # Convert to JAX arrays for training
        j_obs = jnp.array(obs_buf)
        j_act = jnp.array(act_buf)
        j_logp = jnp.array(logp_buf)
        j_adv = jnp.array(advantages)
        j_ret = jnp.array(returns)

        # Mini-batch PPO updates
        n = config.n_steps
        for _ in range(config.n_epochs):
            key, shuffle_key = jr.split(key)
            indices = jr.permutation(shuffle_key, n)
            for start in range(0, n, config.mini_batch_size):
                end = min(start + config.mini_batch_size, n)
                mb_idx = indices[start:end]
                model, opt_state, loss = update_step(
                    model, opt_state,
                    j_obs[mb_idx], j_act[mb_idx], j_logp[mb_idx],
                    j_adv[mb_idx], j_ret[mb_idx],
                )

        # Logging
        if config.verbose and episode_rewards:
            recent = episode_rewards[-10:]
            avg = sum(recent) / len(recent)
            print(f"  PPO step {total_timesteps:>7d} | "
                  f"episodes: {len(episode_rewards):>4d} | "
                  f"avg reward (last 10): {avg:>8.1f}")

    env.close()

    if config.verbose:
        if episode_rewards:
            final_avg = sum(episode_rewards[-20:]) / min(20, len(episode_rewards))
            print(f"  Training complete. Final avg (last 20): {final_avg:.1f}")
        else:
            print("  Training complete. No episodes finished.")

    return model


# ---------------------------------------------------------------------------
# Data collection and evaluation
# ---------------------------------------------------------------------------


def collect_expert_data(
    env_name: str,
    model: ActorCritic,
    n_trajectories: int = 100,
    max_steps: int = 1000,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Collect expert demonstrations from a trained PPO policy.

    Runs the policy deterministically (argmax over logits) to produce
    clean expert trajectories in the same format as the existing
    generate_gym_expert_data.py script.

    Parameters
    ----------
    env_name : str
        Gymnasium environment ID.
    model : ActorCritic
        Trained PPO model.
    n_trajectories : int
        Number of episodes to collect.
    max_steps : int
        Maximum steps per episode.
    seed : int
        Random seed for environment resets.

    Returns
    -------
    dict with keys: observations, actions, next_observations, rewards,
        episode_ids, dones. All numpy arrays.
    """
    env = gym.make(env_name)

    all_obs = []
    all_actions = []
    all_next_obs = []
    all_rewards = []
    all_episode_ids = []
    all_dones = []
    total_reward = 0.0

    @eqx.filter_jit
    def get_greedy_action(model: ActorCritic, obs: jax.Array) -> jax.Array:
        logits, _ = model(obs)
        return jnp.argmax(logits)

    for ep in range(n_trajectories):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0

        for step in range(max_steps):
            action = int(get_greedy_action(
                model, jnp.array(obs, dtype=jnp.float32)
            ))
            next_obs, reward, terminated, truncated, _ = env.step(action)
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


def evaluate_policy(
    env_name: str,
    model: ActorCritic,
    n_episodes: int = 100,
    max_steps: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """Evaluate a trained PPO policy in the environment.

    Parameters
    ----------
    env_name : str
        Gymnasium environment ID.
    model : ActorCritic
        Trained PPO model.
    n_episodes : int
        Number of evaluation episodes.
    max_steps : int
        Maximum steps per episode.
    seed : int
        Random seed for environment resets.

    Returns
    -------
    mean_reward : float
        Mean episode reward.
    std_reward : float
        Standard deviation of episode rewards.
    """
    env = gym.make(env_name)
    rewards = []

    @eqx.filter_jit
    def get_greedy_action(model: ActorCritic, obs: jax.Array) -> jax.Array:
        logits, _ = model(obs)
        return jnp.argmax(logits)

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_reward = 0.0

        for step in range(max_steps):
            action = int(get_greedy_action(
                model, jnp.array(obs, dtype=jnp.float32)
            ))
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break

        rewards.append(episode_reward)

    env.close()
    return float(np.mean(rewards)), float(np.std(rewards))
