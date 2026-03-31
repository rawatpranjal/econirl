"""Objectworld environment for deep IRL benchmarking.

This module implements the Objectworld environment from Levine et al. (2011)
and used in Wulfmeier et al. (2016) for evaluating deep inverse reinforcement
learning. The environment is an N x N grid with colored objects placed
randomly. The reward depends on the minimum Euclidean distance to objects
of each color.

State space:
    N^2 states indexed as row * N + col. No terminal or absorbing state.

Action space:
    5 actions: Left (0), Right (1), Up (2), Down (3), Stay (4).

Reward:
    +1 if within distance 3 of color 0 AND distance 2 of color 1.
    -1 if within distance 3 of color 0 but NOT within distance 2 of color 1.
     0 otherwise.

Feature types:
    "continuous": C dimensions, each the normalized minimum distance to
        the nearest object of that color (divided by grid_size).
    "discrete": C * M binary dimensions, where each indicator is 1 if the
        state is within distance d of color c, for d in 1..M.

References:
    Levine, S., Popovic, Z., & Koltun, V. (2011). Nonlinear inverse
        reinforcement learning with Gaussian processes.
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016). Maximum entropy deep
        inverse reinforcement learning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
from econirl.core.types import Panel, Trajectory, DDCProblem


# Action constants used by _build_grid_transitions and ObjectworldEnvironment.
LEFT = 0
RIGHT = 1
UP = 2
DOWN = 3
STAY = 4


def _build_grid_transitions(grid_size: int) -> torch.Tensor:
    """Build deterministic 5-action grid transition matrices without a terminal state.

    This helper constructs transition matrices for an N x N grid with actions
    Left (0), Right (1), Up (2), Down (3), Stay (4). Moving into a wall leaves
    the agent in place. Unlike GridworldEnvironment, there is no absorbing
    terminal state.

    Args:
        grid_size: Side length N of the N x N grid.

    Returns:
        Tensor of shape (5, N^2, N^2) where result[a, s, s'] = P(s' | s, a).
    """
    n_states = grid_size * grid_size
    n_actions = 5
    transitions = torch.zeros((n_actions, n_states, n_states), dtype=torch.float32)

    for s in range(n_states):
        row = s // grid_size
        col = s % grid_size

        # Left
        new_col = max(col - 1, 0)
        transitions[LEFT, s, row * grid_size + new_col] = 1.0

        # Right
        new_col = min(col + 1, grid_size - 1)
        transitions[RIGHT, s, row * grid_size + new_col] = 1.0

        # Up
        new_row = max(row - 1, 0)
        transitions[UP, s, new_row * grid_size + col] = 1.0

        # Down
        new_row = min(row + 1, grid_size - 1)
        transitions[DOWN, s, new_row * grid_size + col] = 1.0

        # Stay
        transitions[STAY, s, s] = 1.0

    return transitions


class ObjectworldEnvironment(DDCEnvironment):
    """N x N grid with colored objects and distance-based rewards.

    Objects of C colors are placed randomly on the grid. The reward at each
    state depends on the minimum Euclidean distance to objects of each color.
    Two feature representations are available: continuous normalized distances
    and discrete binary distance indicators.

    This environment is designed for benchmarking inverse reinforcement
    learning algorithms where a linear reward function is insufficient
    to capture the true reward structure (the reward depends nonlinearly
    on distances to multiple colors).

    Example:
        >>> env = ObjectworldEnvironment(grid_size=8, n_colors=2, seed=0)
        >>> print(f"States: {env.num_states}, Features: {env.feature_matrix.shape}")
    """

    def __init__(
        self,
        grid_size: int = 32,
        n_colors: int = 2,
        n_objects_per_color: int = 3,
        discount_factor: float = 0.9,
        scale_parameter: float = 1.0,
        feature_type: str = "continuous",
        max_distance: int | None = None,
        seed: int | None = None,
    ):
        """Initialize the Objectworld environment.

        Args:
            grid_size: Side length N of the N x N grid.
            n_colors: Number of object colors C.
            n_objects_per_color: Number of objects placed per color.
            discount_factor: Time discount factor beta in [0, 1).
            scale_parameter: Logit scale parameter sigma > 0.
            feature_type: "continuous" for normalized distances or "discrete"
                for binary distance indicators.
            max_distance: Maximum distance threshold M for discrete features.
                Defaults to grid_size if not specified. Ignored for continuous
                features.
            seed: Random seed for reproducible object placement.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._grid_size = grid_size
        self._n_colors = n_colors
        self._n_objects_per_color = n_objects_per_color
        self._n_states = grid_size * grid_size
        self._feature_type = feature_type
        self._max_distance = max_distance if max_distance is not None else grid_size

        # Set up Gymnasium spaces
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(5)

        # Place objects randomly using the seeded RNG from DDCEnvironment
        self._object_positions = self._place_objects()

        # Precompute minimum distances from every state to each color
        self._min_distances = self._compute_min_distances()

        # Build structural components
        self._transition_matrices = _build_grid_transitions(grid_size)
        self._true_reward = self._compute_reward()
        self._feature_matrix = self._build_feature_matrix()

    def _place_objects(self) -> dict[int, list[int]]:
        """Place objects of each color randomly on the grid.

        Returns:
            Dictionary mapping color index to a list of state indices
            where objects of that color are located.
        """
        positions: dict[int, list[int]] = {}
        for c in range(self._n_colors):
            states = self._np_random.choice(
                self._n_states,
                size=self._n_objects_per_color,
                replace=False,
            )
            positions[c] = states.tolist()
        return positions

    def _state_to_rowcol(self, state: int) -> tuple[int, int]:
        """Convert flat state index to (row, col) coordinates."""
        return state // self._grid_size, state % self._grid_size

    def _euclidean_distance(self, s1: int, s2: int) -> float:
        """Compute Euclidean distance between two states on the grid."""
        r1, c1 = self._state_to_rowcol(s1)
        r2, c2 = self._state_to_rowcol(s2)
        return float(np.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2))

    def _compute_min_distances(self) -> np.ndarray:
        """Compute minimum Euclidean distance from each state to each color.

        Returns:
            Array of shape (n_states, n_colors) where entry [s, c] is the
            minimum distance from state s to any object of color c.
        """
        distances = np.full((self._n_states, self._n_colors), np.inf)
        for c in range(self._n_colors):
            for s in range(self._n_states):
                for obj_state in self._object_positions[c]:
                    d = self._euclidean_distance(s, obj_state)
                    if d < distances[s, c]:
                        distances[s, c] = d
        return distances

    def _compute_reward(self) -> torch.Tensor:
        """Compute the reward for each state based on object distances.

        The reward rule is:
            +1 if min distance to color 0 is within 3 AND
                min distance to color 1 is within 2.
            -1 if min distance to color 0 is within 3 BUT
                min distance to color 1 is NOT within 2.
             0 otherwise.

        Returns:
            Tensor of shape (n_states,) with reward values.
        """
        reward = torch.zeros(self._n_states, dtype=torch.float32)
        for s in range(self._n_states):
            near_color_0 = self._min_distances[s, 0] <= 3.0
            near_color_1 = self._min_distances[s, 1] <= 2.0
            if near_color_0 and near_color_1:
                reward[s] = 1.0
            elif near_color_0 and not near_color_1:
                reward[s] = -1.0
            # Otherwise reward remains 0.0
        return reward

    def _build_feature_matrix(self) -> torch.Tensor:
        """Build feature matrix for utility computation.

        Features are state-only and broadcast identically to all 5 actions.

        For "continuous" features: C dimensions, each the minimum distance
        to the nearest object of that color, normalized by grid_size.

        For "discrete" features: C * M binary dimensions, where each
        indicator is 1 if the state is within distance d of color c,
        for d in 1..M.

        Returns:
            Tensor of shape (n_states, 5, n_features).
        """
        if self._feature_type == "continuous":
            n_features = self._n_colors
            features_per_state = np.zeros((self._n_states, n_features))
            for s in range(self._n_states):
                for c in range(self._n_colors):
                    features_per_state[s, c] = (
                        self._min_distances[s, c] / self._grid_size
                    )
        elif self._feature_type == "discrete":
            M = self._max_distance
            n_features = self._n_colors * M
            features_per_state = np.zeros((self._n_states, n_features))
            for s in range(self._n_states):
                for c in range(self._n_colors):
                    for d in range(1, M + 1):
                        if self._min_distances[s, c] <= d:
                            features_per_state[s, c * M + (d - 1)] = 1.0
        else:
            raise ValueError(
                f"Unknown feature_type '{self._feature_type}'. "
                "Use 'continuous' or 'discrete'."
            )

        # Broadcast state-only features to all 5 actions: (S, K) -> (S, 5, K)
        state_features = torch.tensor(features_per_state, dtype=torch.float32)
        feature_matrix = state_features.unsqueeze(1).expand(
            self._n_states, 5, n_features
        ).clone()
        return feature_matrix

    # ------------------------------------------------------------------
    # DDCEnvironment abstract property implementations
    # ------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._n_states

    @property
    def num_actions(self) -> int:
        return 5

    @property
    def transition_matrices(self) -> torch.Tensor:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> torch.Tensor:
        return self._feature_matrix

    @property
    def true_reward(self) -> torch.Tensor:
        """Return the ground-truth reward vector of shape (num_states,)."""
        return self._true_reward

    @property
    def true_parameters(self) -> dict[str, float]:
        """Return true parameters.

        The Objectworld reward is not a linear function of the features,
        so there is no single parameter vector that recovers the reward
        exactly. This returns placeholder values used only for the
        DDCEnvironment interface.
        """
        return {f"color_{c}_weight": 1.0 for c in range(self._n_colors)}

    @property
    def parameter_names(self) -> list[str]:
        return [f"color_{c}_weight" for c in range(self._n_colors)]

    @property
    def grid_size(self) -> int:
        """Return the side length of the grid."""
        return self._grid_size

    @property
    def state_dim(self) -> int:
        """Two-dimensional grid position."""
        return 2

    def encode_states(self, states: torch.Tensor) -> torch.Tensor:
        """Encode flat state indices to (row, col) normalized to [0, 1].

        Args:
            states: Tensor of flat state indices.

        Returns:
            Tensor of shape (batch, 2) with normalized row and column.
        """
        rows = (states.float() // self._grid_size) / max(self._grid_size - 1, 1)
        cols = (states.float() % self._grid_size) / max(self._grid_size - 1, 1)
        return torch.stack([rows, cols], dim=-1)

    # ------------------------------------------------------------------
    # DDCEnvironment abstract method implementations
    # ------------------------------------------------------------------

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return uniform initial state distribution."""
        return np.ones(self._n_states) / self._n_states

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Return the reward for the given state (action-independent)."""
        return self._true_reward[state].item()

    def _sample_next_state(self, state: int, action: int) -> int:
        """Return deterministic next state."""
        row = state // self._grid_size
        col = state % self._grid_size

        if action == LEFT:
            col = max(col - 1, 0)
        elif action == RIGHT:
            col = min(col + 1, self._grid_size - 1)
        elif action == UP:
            row = max(row - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self._grid_size - 1)
        # STAY: no change

        return row * self._grid_size + col

    # ------------------------------------------------------------------
    # Demonstration generation
    # ------------------------------------------------------------------

    def simulate_demonstrations(
        self,
        n_demos: int,
        max_steps: int = 50,
        noise_fraction: float = 0.3,
        seed: int = 0,
    ) -> Panel:
        """Generate demonstration trajectories from the optimal policy.

        Solves for the optimal policy under the true reward using policy
        iteration, then samples trajectories. Each action is replaced with
        a uniformly random action with probability noise_fraction.

        Args:
            n_demos: Number of trajectories to generate.
            max_steps: Length of each trajectory.
            noise_fraction: Probability of replacing the optimal action
                with a uniform random action at each step.
            seed: Random seed for trajectory sampling.

        Returns:
            Panel containing the generated trajectories.
        """
        rng = np.random.default_rng(seed)

        # Build the reward matrix (S, A) from the state-only reward
        reward_matrix = self._true_reward.unsqueeze(1).expand(
            self._n_states, 5
        ).clone()

        # Solve for optimal policy
        problem = self.problem_spec
        operator = SoftBellmanOperator(problem, self._transition_matrices)
        result = policy_iteration(operator, reward_matrix)
        policy = result.policy  # (S, A)

        trajectories = []
        for i in range(n_demos):
            # Sample initial state uniformly
            state = int(rng.integers(0, self._n_states))
            states_list = []
            actions_list = []
            next_states_list = []

            for _ in range(max_steps):
                # Choose action: optimal with probability (1 - noise_fraction),
                # uniform random otherwise
                if rng.random() < noise_fraction:
                    action = int(rng.integers(0, 5))
                else:
                    probs = policy[state].numpy()
                    action = int(rng.choice(5, p=probs))

                next_state = self._sample_next_state(state, action)

                states_list.append(state)
                actions_list.append(action)
                next_states_list.append(next_state)

                state = next_state

            trajectories.append(
                Trajectory(
                    states=torch.tensor(states_list, dtype=torch.long),
                    actions=torch.tensor(actions_list, dtype=torch.long),
                    next_states=torch.tensor(next_states_list, dtype=torch.long),
                    individual_id=i,
                )
            )

        return Panel(trajectories=trajectories)

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        N = self._grid_size
        return (
            f"Objectworld Environment ({N}x{N})\n"
            f"{'=' * 40}\n"
            f"States: {self.num_states} ({N}x{N} grid)\n"
            f"Actions: Left (0), Right (1), Up (2), Down (3), Stay (4)\n"
            f"Colors: {self._n_colors}, Objects per color: {self._n_objects_per_color}\n"
            f"Feature type: {self._feature_type}\n"
            f"Discount factor: {self._discount_factor}\n"
        )
