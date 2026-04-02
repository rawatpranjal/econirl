"""Citibike station-to-station route choice DDC environment.

This module implements a destination choice model for Citibike
bikeshare trips. A rider at an origin station cluster during a
time-of-day bucket chooses which destination cluster to ride to.
Station locations are clustered by geographic proximity using
K-Means on latitude and longitude coordinates.

State space:
    n_clusters x n_time_buckets discrete states.
    Default: 20 station clusters x 4 time buckets = 80 states.

Action space:
    n_clusters destination choices.
    Default: 20 destination clusters.

Utility specification:
    U(s,a) = distance_weight * distance(origin, destination)
             + popularity_weight * destination_popularity
             + peak_weight * peak_hour_indicator

Reference:
    Citibike System Data (NYC): https://citibikenyc.com/system-data
    Buchholz, N. (2022). "Spatial Equilibrium, Search Frictions, and
    Dynamic Efficiency in the Taxi Industry." REStud.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


N_CLUSTERS = 20
N_TIME_BUCKETS = 4
N_STATES = N_CLUSTERS * N_TIME_BUCKETS
N_ACTIONS = N_CLUSTERS
N_FEATURES = 6

TIME_LABELS = ["night (0-6)", "morning (6-12)", "afternoon (12-18)", "evening (18-24)"]


def state_to_components(state: int) -> tuple[int, int]:
    """Convert flat state index to (origin_cluster, time_bucket)."""
    return state // N_TIME_BUCKETS, state % N_TIME_BUCKETS


def components_to_state(origin_cluster: int, time_bucket: int) -> int:
    """Convert (origin_cluster, time_bucket) to flat state index."""
    return origin_cluster * N_TIME_BUCKETS + time_bucket


class CitibikeRouteEnvironment(DDCEnvironment):
    """Citibike destination choice environment for route choice IRL.

    A rider at an origin station cluster chooses a destination cluster
    given the time of day. Features capture geographic distance between
    clusters, destination popularity, and whether the trip is during
    peak hours. Transitions capture the probability of being at a
    given origin cluster and time bucket after arriving at the
    destination.

    State space:
        80 states = 20 origin clusters x 4 time buckets.

    Action space:
        20 destination clusters.

    Example:
        >>> env = CitibikeRouteEnvironment()
        >>> obs, info = env.reset()
        >>> oc, tb = state_to_components(obs)
        >>> print(f"Origin cluster: {oc}, Time: {TIME_LABELS[tb]}")
    """

    def __init__(
        self,
        distance_weight: float = -1.0,
        popularity_weight: float = 0.5,
        peak_weight: float = 0.3,
        evening_weight: float = 0.2,
        distance_sq_weight: float = -0.5,
        same_cluster_weight: float = 1.0,
        centroids: np.ndarray | None = None,
        dest_popularity: np.ndarray | None = None,
        data_path: str | Path | None = None,
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._distance_weight = distance_weight
        self._popularity_weight = popularity_weight
        self._peak_weight = peak_weight
        self._evening_weight = evening_weight
        self._distance_sq_weight = distance_sq_weight
        self._same_cluster_weight = same_cluster_weight

        # Load or generate centroids
        if centroids is not None:
            self._centroids = centroids
        elif data_path is not None:
            centroids_path = Path(data_path).parent / "citibike_centroids.npy"
            if centroids_path.exists():
                self._centroids = np.load(centroids_path)
            else:
                self._centroids = self._generate_default_centroids()
        else:
            self._centroids = self._generate_default_centroids()

        # Destination popularity (uniform if not provided)
        if dest_popularity is not None:
            self._dest_popularity = dest_popularity
        else:
            self._dest_popularity = np.ones(N_CLUSTERS) / N_CLUSTERS

        # Precompute distance matrix between clusters
        self._distance_matrix = self._compute_distance_matrix()

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @staticmethod
    def _generate_default_centroids() -> np.ndarray:
        """Generate synthetic centroids spread across NYC area."""
        rng = np.random.RandomState(42)
        lats = rng.uniform(40.68, 40.82, N_CLUSTERS)
        lngs = rng.uniform(-74.02, -73.92, N_CLUSTERS)
        return np.column_stack([lats, lngs])

    def _compute_distance_matrix(self) -> np.ndarray:
        """Compute pairwise Euclidean distance between cluster centroids."""
        n = len(self._centroids)
        dist = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(n):
                dist[i, j] = np.sqrt(
                    (self._centroids[i, 0] - self._centroids[j, 0]) ** 2
                    + (self._centroids[i, 1] - self._centroids[j, 1]) ** 2
                )
        # Normalize to [0, 1]
        if dist.max() > 0:
            dist /= dist.max()
        return dist

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
            "distance_weight": self._distance_weight,
            "popularity_weight": self._popularity_weight,
            "peak_weight": self._peak_weight,
            "evening_weight": self._evening_weight,
            "distance_sq_weight": self._distance_sq_weight,
            "same_cluster_weight": self._same_cluster_weight,
        }

    @property
    def parameter_names(self) -> list[str]:
        return [
            "distance_weight", "popularity_weight", "peak_weight",
            "evening_weight", "distance_sq_weight", "same_cluster_weight",
        ]

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition matrices.

        After choosing destination cluster d at time t, the next state
        is (d, t') where t' advances by one bucket with 0.7 probability
        and stays the same with 0.3 probability (modeling variable
        dwell times).
        """
        transitions = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=np.float32)

        for s in range(N_STATES):
            _oc, tb = state_to_components(s)
            for a in range(N_ACTIONS):
                dest = a
                next_tb_same = tb
                next_tb_advance = (tb + 1) % N_TIME_BUCKETS

                ns_same = components_to_state(dest, next_tb_same)
                ns_advance = components_to_state(dest, next_tb_advance)

                transitions[a, s, ns_same] = 0.3
                transitions[a, s, ns_advance] = 0.7

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Six features:
        - distance: normalized distance from origin to destination cluster
        - popularity: destination cluster popularity (fraction of trips)
        - peak_indicator: 1 during morning (6-12) and afternoon (12-18)
        - evening_indicator: 1 during evening (18-24)
        - distance_squared: squared normalized distance (nonlinear aversion)
        - same_cluster: 1 if origin == destination cluster (self-loop)
        """
        features = np.zeros((N_STATES, N_ACTIONS, N_FEATURES), dtype=np.float32)

        for s in range(N_STATES):
            oc, tb = state_to_components(s)
            for a in range(N_ACTIONS):
                dest = a
                d = self._distance_matrix[oc, dest]
                features[s, a, 0] = d
                features[s, a, 1] = self._dest_popularity[dest]
                features[s, a, 2] = 1.0 if tb in [1, 2] else 0.0
                features[s, a, 3] = 1.0 if tb == 3 else 0.0
                features[s, a, 4] = d ** 2
                features[s, a, 5] = 1.0 if oc == dest else 0.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Uniform over clusters, weighted toward morning."""
        dist = np.zeros(N_STATES)
        for c in range(N_CLUSTERS):
            dist[components_to_state(c, 1)] = 0.4 / N_CLUSTERS  # morning
            dist[components_to_state(c, 2)] = 0.3 / N_CLUSTERS  # afternoon
            dist[components_to_state(c, 3)] = 0.2 / N_CLUSTERS  # evening
            dist[components_to_state(c, 0)] = 0.1 / N_CLUSTERS  # night
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        oc, tb = state_to_components(state)
        dest = action
        d = self._distance_matrix[oc, dest]
        u = self._distance_weight * d
        u += self._popularity_weight * self._dest_popularity[dest]
        u += self._peak_weight * (1.0 if tb in [1, 2] else 0.0)
        u += self._evening_weight * (1.0 if tb == 3 else 0.0)
        u += self._distance_sq_weight * (d ** 2)
        u += self._same_cluster_weight * (1.0 if oc == dest else 0.0)
        return u

    def _sample_next_state(self, state: int, action: int) -> int:
        _oc, tb = state_to_components(state)
        dest = action
        if self._np_random.random() < 0.7:
            next_tb = (tb + 1) % N_TIME_BUCKETS
        else:
            next_tb = tb
        return components_to_state(dest, next_tb)

    def _state_to_record(self, state: int, action: int) -> dict:
        oc, tb = state_to_components(state)
        return {
            "origin_cluster": oc,
            "dest_cluster": action,
            "time_bucket": tb,
        }

    @classmethod
    def info(cls) -> dict:
        return {
            "name": "Citibike Route Choice",
            "description": (
                "NYC Citibike station-to-station destination choice. "
                "80 states (20 station clusters x 4 time buckets), "
                "20 actions (destination clusters)."
            ),
            "source": "Synthetic (calibrated to Citibike System Data)",
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "n_features": N_FEATURES,
            "state_description": "Origin station cluster x time-of-day bucket",
            "action_description": "Destination station cluster (0-19)",
            "ground_truth": True,
            "use_case": "Route choice IRL, urban mobility",
        }

    def describe(self) -> str:
        return f"""Citibike Route Choice Environment
{'=' * 40}
States: {N_STATES} ({N_CLUSTERS} station clusters x {N_TIME_BUCKETS} time buckets)
Actions: {N_ACTIONS} destination clusters

True Parameters:
  Distance weight:      {self._distance_weight}
  Popularity weight:    {self._popularity_weight}
  Peak weight:          {self._peak_weight}
  Evening weight:       {self._evening_weight}
  Distance sq weight:   {self._distance_sq_weight}
  Same cluster weight:  {self._same_cluster_weight}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
