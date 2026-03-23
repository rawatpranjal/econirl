"""Deterministic gridworld DDC environment for IRL benchmarking.

This module implements an N x N gridworld as a dynamic discrete choice
environment. The agent navigates on a grid with 5 actions (Left, Right,
Up, Down, Stay) toward an absorbing terminal state at the bottom-right
corner (N-1, N-1).

The environment is designed for inverse reinforcement learning benchmarks
in the infinite-horizon DDC framework with logit preference shocks.

State space:
    N^2 states indexed as row * N + col.
    Terminal state at (N-1, N-1) is absorbing (all actions self-loop).

Action space:
    5 actions: Left (0), Right (1), Up (2), Down (3), Stay (4).

Utility specification:
    U(s, a) = step_penalty * I(non-terminal)
              + terminal_reward * I(next_state == terminal)
              + distance_weight * (-manhattan_distance / (2*N))
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


class GridworldEnvironment(DDCEnvironment):
    """Deterministic N x N gridworld DDC environment.

    An agent moves on a grid toward an absorbing terminal state at the
    bottom-right corner. Designed for IRL benchmarking in the DDC
    framework with logit shocks (infinite horizon, never terminated).

    State space:
        N^2 states. State index = row * N + col.
        Terminal state at (N-1, N-1) is absorbing.

    Action space:
        0 = Left, 1 = Right, 2 = Up, 3 = Down, 4 = Stay

    Transitions:
        Deterministic. Each action moves the agent one cell in the
        corresponding direction, or stays in place if hitting a wall.
        All actions at the terminal state lead back to the terminal.

    Feature matrix:
        3 features per state-action pair:
        - step_penalty indicator (1.0 at non-terminal, 0.0 at terminal)
        - terminal_reward indicator (1.0 when action leads to terminal)
        - distance feature (-manhattan_dist / (2*N) for non-terminal, 0 at terminal)

    Example:
        >>> env = GridworldEnvironment(grid_size=5, discount_factor=0.99)
        >>> obs, info = env.reset()
        >>> print(f"Initial state: {obs} (row={obs // 5}, col={obs % 5})")
    """

    # Action constants
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3
    STAY = 4

    def __init__(
        self,
        grid_size: int = 5,
        step_penalty: float = -0.1,
        terminal_reward: float = 10.0,
        distance_weight: float = 0.1,
        discount_factor: float = 0.99,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the gridworld environment.

        Args:
            grid_size: Side length N of the N x N grid.
            step_penalty: Per-step cost for non-terminal state-action pairs.
            terminal_reward: Reward for reaching the terminal state.
            distance_weight: Weight on the Manhattan distance feature.
            discount_factor: Time discount factor beta in [0, 1).
            scale_parameter: Logit scale parameter sigma > 0.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        if grid_size < 2:
            raise ValueError(f"grid_size must be >= 2, got {grid_size}")

        self._grid_size = grid_size
        self._step_penalty = step_penalty
        self._terminal_reward = terminal_reward
        self._distance_weight = distance_weight
        self._n_states = grid_size * grid_size
        self._terminal_state = self._n_states - 1  # (N-1, N-1)

        # Set up Gymnasium spaces
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(5)

        # Pre-compute transition matrices and features
        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @property
    def grid_size(self) -> int:
        """Return the side length N of the grid."""
        return self._grid_size

    @property
    def terminal_state(self) -> int:
        """Return the index of the terminal (absorbing) state."""
        return self._terminal_state

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
    def true_parameters(self) -> dict[str, float]:
        return {
            "step_penalty": self._step_penalty,
            "terminal_reward": self._terminal_reward,
            "distance_weight": self._distance_weight,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["step_penalty", "terminal_reward", "distance_weight"]

    def _state_to_rowcol(self, state: int) -> tuple[int, int]:
        """Convert a flat state index to (row, col) coordinates."""
        row = state // self._grid_size
        col = state % self._grid_size
        return row, col

    def _rowcol_to_state(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates to a flat state index."""
        return row * self._grid_size + col

    def _manhattan_distance_to_terminal(self, state: int) -> int:
        """Compute the Manhattan distance from state to the terminal."""
        row, col = self._state_to_rowcol(state)
        N = self._grid_size
        return (N - 1 - row) + (N - 1 - col)

    def _get_next_state(self, state: int, action: int) -> int:
        """Compute the deterministic next state for a given action.

        At the terminal state, all actions self-loop.
        At walls, the agent stays in place.

        Args:
            state: Current state index.
            action: Action index (0=Left, 1=Right, 2=Up, 3=Down, 4=Stay).

        Returns:
            Next state index.
        """
        # Terminal state is absorbing
        if state == self._terminal_state:
            return self._terminal_state

        row, col = self._state_to_rowcol(state)
        N = self._grid_size

        if action == self.LEFT:
            col = max(col - 1, 0)
        elif action == self.RIGHT:
            col = min(col + 1, N - 1)
        elif action == self.UP:
            row = max(row - 1, 0)
        elif action == self.DOWN:
            row = min(row + 1, N - 1)
        # STAY: no change

        return self._rowcol_to_state(row, col)

    def _build_transition_matrices(self) -> torch.Tensor:
        """Build deterministic transition matrices P(s'|s,a).

        Returns:
            Tensor of shape (5, N^2, N^2) where
            transitions[a, s, s'] = 1.0 if action a in state s leads to s',
            0.0 otherwise.
        """
        n_actions = 5
        n_states = self._n_states

        transitions = torch.zeros(
            (n_actions, n_states, n_states), dtype=torch.float32
        )

        for a in range(n_actions):
            for s in range(n_states):
                s_next = self._get_next_state(s, a)
                transitions[a, s, s_next] = 1.0

        return transitions

    def _build_feature_matrix(self) -> torch.Tensor:
        """Build feature matrix for utility computation.

        Features are structured so that:
            U(s, a) = theta . phi(s, a)

        where theta = [step_penalty, terminal_reward, distance_weight]

        Feature 0: step_penalty indicator
            1.0 for all non-terminal state-action pairs, 0.0 at terminal.
        Feature 1: terminal_reward indicator
            1.0 when action leads to the terminal state, 0.0 otherwise.
        Feature 2: distance feature
            -manhattan_distance_to_terminal / (2*N) for non-terminal states,
            0.0 at the terminal state.

        Returns:
            Tensor of shape (N^2, 5, 3).
        """
        n_states = self._n_states
        n_actions = 5
        N = self._grid_size

        features = torch.zeros((n_states, n_actions, 3), dtype=torch.float32)

        for s in range(n_states):
            is_terminal = s == self._terminal_state

            for a in range(n_actions):
                s_next = self._get_next_state(s, a)

                # Feature 0: step_penalty indicator
                if not is_terminal:
                    features[s, a, 0] = 1.0

                # Feature 1: terminal_reward indicator
                if s_next == self._terminal_state:
                    features[s, a, 1] = 1.0

                # Feature 2: distance feature
                if not is_terminal:
                    dist = self._manhattan_distance_to_terminal(s)
                    features[s, a, 2] = -dist / (2.0 * N)

        return features

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return initial state distribution (start at top-left corner)."""
        dist = np.zeros(self._n_states)
        dist[0] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility for a state-action pair using true parameters."""
        is_terminal = state == self._terminal_state
        next_state = self._get_next_state(state, action)

        utility = 0.0

        # Step penalty for non-terminal states
        if not is_terminal:
            utility += self._step_penalty

        # Terminal reward when reaching terminal
        if next_state == self._terminal_state:
            utility += self._terminal_reward

        # Distance feature
        if not is_terminal:
            dist = self._manhattan_distance_to_terminal(state)
            utility += self._distance_weight * (-dist / (2.0 * self._grid_size))

        return utility

    def _sample_next_state(self, state: int, action: int) -> int:
        """Sample next state (deterministic in this environment)."""
        return self._get_next_state(state, action)

    def state_to_grid_position(self, state: int) -> tuple[int, int]:
        """Convert state index to grid position (row, col).

        Args:
            state: Flat state index.

        Returns:
            Tuple of (row, col).
        """
        return self._state_to_rowcol(state)

    def grid_position_to_state(self, row: int, col: int) -> int:
        """Convert grid position to state index.

        Args:
            row: Row index (0-indexed from top).
            col: Column index (0-indexed from left).

        Returns:
            Flat state index.
        """
        return self._rowcol_to_state(row, col)

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        N = self._grid_size
        return f"""Gridworld Environment ({N}x{N})
{'=' * 40}
States: {self.num_states} ({N}x{N} grid)
Actions: Left (0), Right (1), Up (2), Down (3), Stay (4)
Terminal state: {self._terminal_state} (row={N-1}, col={N-1})

True Parameters:
  Step penalty:     {self._step_penalty}
  Terminal reward:  {self._terminal_reward}
  Distance weight:  {self._distance_weight}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
