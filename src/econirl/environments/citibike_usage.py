"""Citibike daily usage frequency DDC environment.

This module implements a daily ride/no-ride decision model for
Citibike members. Each day a member decides whether to take a
bikeshare trip. The state captures the day type (weekday or weekend)
and recent usage intensity (number of rides in the last 7 days,
bucketed). This models habitual transportation behavior where past
usage reinforces future usage.

State space:
    n_day_types x n_usage_buckets = 8 states by default.
    Day types: weekday (0), weekend (1).
    Usage buckets: 0 rides (0), 1-2 rides (1), 3-5 rides (2), 6+ rides (3).

Action space:
    2 actions: No ride (0) and Ride (1).

Utility specification:
    U(s, a=ride) = weekend_effect * weekend_indicator
                   + habit_strength * usage_intensity
                   + ride_cost
    U(s, a=no_ride) = 0 (normalized)

Reference:
    Citibike System Data (NYC): https://citibikenyc.com/system-data
    Buchholz, N. (2022). "Spatial Equilibrium, Search Frictions, and
    Dynamic Efficiency in the Taxi Industry." REStud.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


N_DAY_TYPES = 2
N_USAGE_BUCKETS = 4
N_STATES = N_DAY_TYPES * N_USAGE_BUCKETS
N_ACTIONS = 2
N_FEATURES = 3

DAY_LABELS = ["weekday", "weekend"]
USAGE_LABELS = ["0 rides", "1-2 rides", "3-5 rides", "6+ rides"]


def state_to_components(state: int) -> tuple[int, int]:
    """Convert flat state index to (day_type, usage_bucket)."""
    return state // N_USAGE_BUCKETS, state % N_USAGE_BUCKETS


def components_to_state(day_type: int, usage_bucket: int) -> int:
    """Convert (day_type, usage_bucket) to flat state index."""
    return day_type * N_USAGE_BUCKETS + usage_bucket


class CitibikeUsageEnvironment(DDCEnvironment):
    """Citibike daily usage frequency environment for transportation DDC.

    A Citibike member decides each day whether to take a bikeshare
    trip. The state captures day type (weekday or weekend) and recent
    usage intensity. Riding increases the usage bucket (habit
    formation). Not riding decreases it (habit decay). The model
    captures how commuting habits, weekend leisure patterns, and
    the cost of each trip shape daily transportation decisions.

    State space:
        8 states = 2 day types x 4 usage buckets.

    Action space:
        No ride (0) or Ride (1).

    Example:
        >>> env = CitibikeUsageEnvironment()
        >>> obs, info = env.reset()
        >>> dt, ub = state_to_components(obs)
        >>> print(f"Day: {DAY_LABELS[dt]}, Recent usage: {USAGE_LABELS[ub]}")
    """

    def __init__(
        self,
        weekend_effect: float = -0.3,
        habit_strength: float = 0.8,
        ride_cost: float = -0.5,
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._weekend_effect = weekend_effect
        self._habit_strength = habit_strength
        self._ride_cost = ride_cost

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
            "weekend_effect": self._weekend_effect,
            "habit_strength": self._habit_strength,
            "ride_cost": self._ride_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["weekend_effect", "habit_strength", "ride_cost"]

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition matrices for no-ride and ride actions.

        Day type transitions: weekday stays weekday with prob 0.71
        (5/7 days are weekdays), weekend stays weekend with prob 0.29.

        Usage bucket transitions:
          No ride: usage bucket decays by 1 (min 0).
          Ride: usage bucket increases by 1 (max 3).
        """
        transitions = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=np.float32)

        p_weekday_next = 5.0 / 7.0  # prob next day is weekday

        for s in range(N_STATES):
            dt, ub = state_to_components(s)

            for a in range(N_ACTIONS):
                if a == 0:  # no ride
                    new_ub = max(ub - 1, 0)
                else:  # ride
                    new_ub = min(ub + 1, N_USAGE_BUCKETS - 1)

                # Day type transitions (independent of action)
                ns_weekday = components_to_state(0, new_ub)
                ns_weekend = components_to_state(1, new_ub)
                transitions[a, s, ns_weekday] = p_weekday_next
                transitions[a, s, ns_weekend] = 1.0 - p_weekday_next

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Three features. For no-ride action all features are zero.
        For ride: weekend indicator, usage intensity (normalized),
        and a ride cost indicator.
        """
        features = np.zeros((N_STATES, N_ACTIONS, N_FEATURES), dtype=np.float32)

        for s in range(N_STATES):
            dt, ub = state_to_components(s)

            # Ride action features
            features[s, 1, 0] = float(dt == 1)  # weekend
            features[s, 1, 1] = ub / (N_USAGE_BUCKETS - 1)  # usage intensity
            features[s, 1, 2] = 1.0  # ride cost indicator

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Start on a weekday with low recent usage."""
        dist = np.zeros(N_STATES)
        dist[components_to_state(0, 0)] = 0.4
        dist[components_to_state(0, 1)] = 0.3
        dist[components_to_state(1, 0)] = 0.2
        dist[components_to_state(1, 1)] = 0.1
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        if action == 0:
            return 0.0
        dt, ub = state_to_components(state)
        usage_intensity = ub / (N_USAGE_BUCKETS - 1)
        u = self._weekend_effect * float(dt == 1)
        u += self._habit_strength * usage_intensity
        u += self._ride_cost
        return u

    def _sample_next_state(self, state: int, action: int) -> int:
        dt, ub = state_to_components(state)
        if action == 0:
            new_ub = max(ub - 1, 0)
        else:
            new_ub = min(ub + 1, N_USAGE_BUCKETS - 1)

        next_dt = 0 if self._np_random.random() < 5.0 / 7.0 else 1
        return components_to_state(next_dt, new_ub)

    def _state_to_record(self, state: int, action: int) -> dict:
        dt, ub = state_to_components(state)
        return {
            "day_type": dt,
            "usage_bucket": ub,
            "day_label": DAY_LABELS[dt],
            "usage_label": USAGE_LABELS[ub],
            "rode": action == 1,
        }

    @classmethod
    def info(cls) -> dict:
        return {
            "name": "Citibike Daily Usage Frequency",
            "description": (
                "NYC Citibike member daily ride/no-ride decisions. "
                "8 states (day type x recent usage bucket), "
                "2 actions (ride/no ride)."
            ),
            "source": "Synthetic (calibrated to Citibike System Data)",
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "n_features": N_FEATURES,
            "state_description": "Day type (weekday/weekend) x recent usage bucket",
            "action_description": "No ride (0) / Ride (1)",
            "ground_truth": True,
            "use_case": "Transportation DDC, habitual behavior, usage frequency",
        }

    def describe(self) -> str:
        return f"""Citibike Usage Frequency Environment
{'=' * 40}
States: {N_STATES} ({N_DAY_TYPES} day types x {N_USAGE_BUCKETS} usage buckets)
Actions: No Ride (0), Ride (1)

True Parameters:
  Weekend effect:   {self._weekend_effect}
  Habit strength:   {self._habit_strength}
  Ride cost:        {self._ride_cost}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
