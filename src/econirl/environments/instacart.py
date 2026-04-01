"""Grocery reorder DDC environment for repeat-purchase estimation.

This module implements a synthetic repeat-purchase environment modeled on
Instacart-style grocery delivery data. A consumer decides each period
whether to reorder a product category. The state captures purchase
frequency and recency of last order.

State space:
    20 states = 5 purchase frequency buckets x 4 recency buckets.

Action space:
    2 actions: Skip (0) and Reorder (1).

Utility specification:
    U(s, a=reorder) = habit_strength * purchase_frequency
                      + recency_effect * (1 / days_since_last)
                      + reorder_cost
    U(s, a=skip)    = 0 (normalized)

Reference:
    Instacart Market Basket Analysis (Kaggle 2017).
    Erdem, T. & Keane, M.P. (1996). "Decision-Making Under Uncertainty."
    Marketing Science.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


N_PURCHASE_BUCKETS = 5
N_RECENCY_BUCKETS = 4
N_STATES = N_PURCHASE_BUCKETS * N_RECENCY_BUCKETS
N_ACTIONS = 2

PURCHASE_LABELS = ["0 orders", "1-2 orders", "3-5 orders", "6-10 orders", "11+ orders"]
RECENCY_LABELS = ["0-3 days", "4-7 days", "8-14 days", "15+ days"]


def state_to_buckets(state: int) -> tuple[int, int]:
    """Convert flat state index to (purchase_bucket, recency_bucket)."""
    return state // N_RECENCY_BUCKETS, state % N_RECENCY_BUCKETS


def buckets_to_state(purchase_bucket: int, recency_bucket: int) -> int:
    """Convert (purchase_bucket, recency_bucket) to flat state index."""
    return purchase_bucket * N_RECENCY_BUCKETS + recency_bucket


class InstacartEnvironment(DDCEnvironment):
    """Grocery reorder environment for repeat-purchase DDC.

    A consumer decides each period whether to reorder a product category
    from a grocery delivery platform. The state captures past purchase
    frequency and recency of last order. Reordering increases the purchase
    frequency bucket and resets recency. Skipping increases recency.

    State space:
        20 states = 5 purchase frequency buckets x 4 recency buckets.

    Action space:
        Skip (0) or Reorder (1).

    Example:
        >>> env = InstacartEnvironment()
        >>> obs, info = env.reset()
        >>> pb, rb = state_to_buckets(obs)
        >>> print(f"Purchase history: {PURCHASE_LABELS[pb]}, Last order: {RECENCY_LABELS[rb]}")
    """

    def __init__(
        self,
        habit_strength: float = 0.3,
        recency_effect: float = 0.5,
        reorder_cost: float = -0.2,
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._habit_strength = habit_strength
        self._recency_effect = recency_effect
        self._reorder_cost = reorder_cost

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @property
    def num_states(self) -> int:
        return N_STATES

    @property
    def num_actions(self) -> int:
        return N_ACTIONS

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "habit_strength": self._habit_strength,
            "recency_effect": self._recency_effect,
            "reorder_cost": self._reorder_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["habit_strength", "recency_effect", "reorder_cost"]

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition matrices for skip and reorder actions.

        Skip (action 0): Recency bucket increases. Purchase bucket stays
            the same or decays with probability 0.1.
        Reorder (action 1): Purchase bucket increases. Recency resets to 0.
        """
        transitions = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=np.float32)

        for s in range(N_STATES):
            pb, rb = state_to_buckets(s)

            # Skip
            new_rb_skip = min(rb + 1, N_RECENCY_BUCKETS - 1)
            if pb > 0:
                ns_decay = buckets_to_state(pb - 1, new_rb_skip)
                ns_stay = buckets_to_state(pb, new_rb_skip)
                transitions[0, s, ns_decay] = 0.1
                transitions[0, s, ns_stay] = 0.9
            else:
                ns_stay = buckets_to_state(0, new_rb_skip)
                transitions[0, s, ns_stay] = 1.0

            # Reorder
            new_pb = min(pb + 1, N_PURCHASE_BUCKETS - 1)
            ns_reorder = buckets_to_state(new_pb, 0)
            transitions[1, s, ns_reorder] = 1.0

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Three features. For the skip action all features are zero (baseline).
        For reorder: habit (normalized purchase frequency), recency (inverse
        time since last order), and a reorder cost indicator.
        """
        features = np.zeros((N_STATES, N_ACTIONS, 3), dtype=np.float32)

        for s in range(N_STATES):
            pb, rb = state_to_buckets(s)
            features[s, 1, 0] = pb / (N_PURCHASE_BUCKETS - 1)
            features[s, 1, 1] = 1.0 - rb / (N_RECENCY_BUCKETS - 1)
            features[s, 1, 2] = 1.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        dist = np.zeros(N_STATES)
        dist[buckets_to_state(0, 2)] = 0.3
        dist[buckets_to_state(0, 3)] = 0.3
        dist[buckets_to_state(1, 1)] = 0.2
        dist[buckets_to_state(1, 2)] = 0.2
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        if action == 0:
            return 0.0
        pb, rb = state_to_buckets(state)
        habit = pb / (N_PURCHASE_BUCKETS - 1)
        recency = 1.0 - rb / (N_RECENCY_BUCKETS - 1)
        return (
            self._habit_strength * habit
            + self._recency_effect * recency
            + self._reorder_cost
        )

    def _sample_next_state(self, state: int, action: int) -> int:
        pb, rb = state_to_buckets(state)
        if action == 0:
            new_rb = min(rb + 1, N_RECENCY_BUCKETS - 1)
            if pb > 0 and self._np_random.random() < 0.1:
                pb -= 1
            return buckets_to_state(pb, new_rb)
        else:
            new_pb = min(pb + 1, N_PURCHASE_BUCKETS - 1)
            return buckets_to_state(new_pb, 0)

    def describe(self) -> str:
        return f"""Instacart Grocery Reorder Environment
{'=' * 40}
States: {N_STATES} ({N_PURCHASE_BUCKETS} purchase buckets x {N_RECENCY_BUCKETS} recency buckets)
Actions: Skip (0), Reorder (1)

True Parameters:
  Habit strength:  {self._habit_strength}
  Recency effect:  {self._recency_effect}
  Reorder cost:    {self._reorder_cost}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
