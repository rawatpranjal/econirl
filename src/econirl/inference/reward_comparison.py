"""Reward function comparison tools for inverse reinforcement learning.

Tools for comparing reward functions up to shaping equivalence:
- EPIC distance (Gleave et al., 2020): canonicalize then Pearson distance
- Reward shaping detection: test if two rewards differ only by potential shaping

These diagnostics address the fundamental non-identifiability of IRL:
rewards are only identified up to potential-based shaping (Ng et al., 1999)
and S'-redistribution (Skalse et al., 2023).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def epic_distance(
    reward_1: jnp.ndarray,
    reward_2: jnp.ndarray,
    discount_factor: float,
    state_dist: jnp.ndarray | None = None,
    action_dist: jnp.ndarray | None = None,
) -> dict[str, float]:
    """EPIC distance between two reward functions.

    Canonicalizes both rewards to remove potential-based shaping,
    then computes the Pearson distance. The result is invariant to
    potential shaping and positive scaling, so EPIC(R, R + gamma*Phi(s') - Phi(s)) = 0
    and EPIC(R, c*R) = 0 for any c > 0.

    The canonically shaped reward is
        C(R)(s,a,s') = R(s,a,s') + gamma * E_mean(s') - E_mean(s) - gamma * global_mean
    where E_mean(x) = sum_a D_A(a) sum_{s'} D_S(s') R(x, a, s').

    Accepts rewards of shape (S, A, S) for transition-dependent rewards
    or (S, A) for state-action rewards. State-action rewards gain an
    s' dependence after canonicalization.

    Args:
        reward_1: Reward function, shape (S, A, S) or (S, A).
        reward_2: Reward function, same shape as reward_1.
        discount_factor: Discount factor gamma in [0, 1).
        state_dist: Distribution over states, shape (S,). Uniform if None.
        action_dist: Distribution over actions, shape (A,). Uniform if None.

    Returns:
        Dict with keys: epic_distance, pearson_correlation.
    """
    r1 = jnp.asarray(reward_1, dtype=jnp.float32)
    r2 = jnp.asarray(reward_2, dtype=jnp.float32)

    if r1.shape != r2.shape:
        raise ValueError(f"Reward shapes must match: {r1.shape} vs {r2.shape}")

    is_sa = r1.ndim == 2
    if is_sa:
        num_states, num_actions = r1.shape
    else:
        num_states, num_actions, _ = r1.shape

    if state_dist is None:
        state_dist = jnp.ones(num_states) / num_states
    if action_dist is None:
        action_dist = jnp.ones(num_actions) / num_actions

    c1 = _canonicalize(r1, discount_factor, state_dist, action_dist)
    c2 = _canonicalize(r2, discount_factor, state_dist, action_dist)

    c1_flat = c1.ravel()
    c2_flat = c2.ravel()

    rho = float(jnp.corrcoef(c1_flat, c2_flat)[0, 1])
    # Clamp to [-1, 1] for numerical safety
    rho = max(-1.0, min(1.0, rho))

    distance = np.sqrt((1.0 - rho) / 2.0)

    return {"epic_distance": distance, "pearson_correlation": rho}


def _canonicalize(
    reward: jnp.ndarray,
    gamma: float,
    state_dist: jnp.ndarray,
    action_dist: jnp.ndarray,
) -> jnp.ndarray:
    """Canonicalize a reward to remove potential shaping.

    For R(s,a,s'): C(R)(s,a,s') = R(s,a,s') + gamma*E_mean(s') - E_mean(s) - gamma*global
    For R(s,a):    same formula, where E_mean(x) = sum_a D_A(a) * R(x,a)
                   and the canonical form gains s' dependence through E_mean(s').
    """
    is_sa = reward.ndim == 2

    if is_sa:
        # E_mean(x) = sum_a D_A(a) * R(x, a), shape (S,)
        e_mean = reward @ action_dist
    else:
        # E_mean(x) = sum_a D_A(a) * sum_{s'} D_S(s') * R(x, a, s'), shape (S,)
        e_mean = jnp.einsum("sap,a,p->s", reward, action_dist, state_dist)

    global_mean = jnp.sum(state_dist * e_mean)

    if is_sa:
        num_states, num_actions = reward.shape
        # C(R)(s,a,s') = R(s,a) + gamma*E_mean(s') - E_mean(s) - gamma*global
        # Shape: (S, A, S) -- expand R(s,a) along s' axis
        canonical = (
            reward[:, :, None]
            + gamma * e_mean[None, None, :]
            - e_mean[:, None, None]
            - gamma * global_mean
        )
    else:
        # C(R)(s,a,s') = R(s,a,s') + gamma*E_mean(s') - E_mean(s) - gamma*global
        canonical = (
            reward
            + gamma * e_mean[None, None, :]
            - e_mean[:, None, None]
            - gamma * global_mean
        )

    return canonical


def detect_reward_shaping(
    reward_1: jnp.ndarray,
    reward_2: jnp.ndarray,
    discount_factor: float,
    transitions: jnp.ndarray | None = None,
) -> dict[str, float | bool]:
    """Detect whether two reward functions differ only by potential shaping.

    Solves R2 - R1 = gamma * Phi(s') - Phi(s) for the potential function Phi
    via least squares. A near-zero residual confirms that the two rewards
    are equivalent up to potential-based shaping.

    For R(s,a) rewards without s' dependence, the shaping equation becomes
    R2(s,a) - R1(s,a) = gamma * E[Phi(s')|s,a] - Phi(s), which requires
    the transition matrix.

    Args:
        reward_1: Baseline reward, shape (S, A, S) or (S, A).
        reward_2: Comparison reward, same shape as reward_1.
        discount_factor: Discount factor gamma.
        transitions: Transition probabilities P(s'|s,a), shape (A, S, S).
            Required when rewards have shape (S, A).

    Returns:
        Dict with keys: is_shaping (bool), residual_norm (float),
        relative_residual (float), potential (list of floats),
        max_absolute_residual (float).
    """
    r1 = jnp.asarray(reward_1, dtype=jnp.float32)
    r2 = jnp.asarray(reward_2, dtype=jnp.float32)
    diff = r2 - r1
    gamma = discount_factor

    is_sa = r1.ndim == 2

    if is_sa:
        if transitions is None:
            raise ValueError("transitions required for R(s,a) rewards")
        num_states, num_actions = r1.shape
        transitions = jnp.asarray(transitions, dtype=jnp.float32)

        # For each (s,a): diff(s,a) = gamma * sum_{s'} P(s'|s,a) * Phi(s') - Phi(s)
        # Design matrix X of shape (S*A, S): row for (s,a) has
        # X[s*A+a, :] = gamma * P(s'|s,a) for all s', minus 1 at column s
        n_eq = num_states * num_actions
        X = np.zeros((n_eq, num_states), dtype=np.float32)
        y = np.asarray(diff.ravel(), dtype=np.float32)

        for s in range(num_states):
            for a in range(num_actions):
                row = s * num_actions + a
                X[row, :] = gamma * np.asarray(transitions[a, s, :])
                X[row, s] -= 1.0
    else:
        num_states, num_actions, _ = r1.shape
        # For each (s,a,s'): diff(s,a,s') = gamma * Phi(s') - Phi(s)
        n_eq = num_states * num_actions * num_states
        X = np.zeros((n_eq, num_states), dtype=np.float32)
        y = np.asarray(diff.ravel(), dtype=np.float32)

        idx = 0
        for s in range(num_states):
            for a in range(num_actions):
                for sp in range(num_states):
                    X[idx, sp] += gamma
                    X[idx, s] -= 1.0
                    idx += 1

    # Pin Phi(0) = 0 for identification: drop column 0
    X_reduced = X[:, 1:]
    result = np.linalg.lstsq(X_reduced, y, rcond=None)
    phi_reduced = result[0]

    phi = np.zeros(num_states, dtype=np.float32)
    phi[1:] = phi_reduced

    residuals = y - X @ phi
    residual_norm = float(np.linalg.norm(residuals))
    diff_norm = float(np.linalg.norm(y))
    relative_residual = residual_norm / diff_norm if diff_norm > 1e-12 else 0.0
    max_abs_residual = float(np.max(np.abs(residuals)))

    return {
        "is_shaping": relative_residual < 0.01,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "potential": [float(x) for x in phi],
        "max_absolute_residual": max_abs_residual,
    }
