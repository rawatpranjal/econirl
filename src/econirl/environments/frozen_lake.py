"""FrozenLake DDC environment for IRL benchmarking.

This module implements the classic FrozenLake-v1 environment from Gymnasium
as a DDCEnvironment for structural estimation. A 4x4 grid represents a
frozen lake where the agent must navigate from the start cell to the goal
without falling into holes. On the slippery surface, each action moves the
agent in the intended direction with probability 1/3 and perpendicular
directions with probability 1/3 each.

This environment does NOT depend on Gymnasium at runtime. The transition
logic is hardcoded directly to avoid adding a dependency.

State space:
    16 states on a 4x4 grid. State index = row * 4 + col.
    Holes at positions 5, 7, 11, 12. Goal at position 15.
    Goal and holes are absorbing states.

Action space:
    4 actions: Left (0), Down (1), Right (2), Up (3).

Utility specification:
    U(s, a) = step_penalty * I(non-terminal)
              + goal_reward * I(next_state == goal)
              + hole_penalty * I(next_state in holes)

    The agent receives a step penalty at every non-terminal state, a goal
    reward for reaching the goal, and a hole penalty for falling in.

Reference:
    Gymnasium FrozenLake-v1 (Farama Foundation).
    Originally from Sutton & Barto, Reinforcement Learning (2018).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


# Default 4x4 map
# S = start, F = frozen, H = hole, G = goal
DEFAULT_MAP = [
    "SFFF",
    "FHFH",
    "FFFH",
    "HFFG",
]

# Action constants
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

ACTION_NAMES = ["Left", "Down", "Right", "Up"]


def _parse_map(map_desc: list[str]) -> tuple[int, int, set[int], int, int]:
    """Parse a map description into grid dimensions and special states."""
    nrow = len(map_desc)
    ncol = len(map_desc[0])
    holes = set()
    start = 0
    goal = 0
    for r, row in enumerate(map_desc):
        for c, cell in enumerate(row):
            s = r * ncol + c
            if cell == "H":
                holes.add(s)
            elif cell == "S":
                start = s
            elif cell == "G":
                goal = s
    return nrow, ncol, holes, start, goal


class FrozenLakeEnvironment(DDCEnvironment):
    """Slippery FrozenLake DDC environment (4x4 grid).

    The agent navigates a frozen lake to reach the goal without falling
    into holes. On the slippery surface, each action has a 1/3 chance
    of moving in the intended direction and 1/3 chance each of moving
    perpendicular. Holes and the goal are absorbing states.

    This is the simplest possible stochastic DDC testbed, useful for
    unit testing estimators and as a minimal IRL benchmark.

    State space:
        16 states on a 4x4 grid. Holes at 5, 7, 11, 12. Goal at 15.

    Action space:
        Left (0), Down (1), Right (2), Up (3).

    Example:
        >>> env = FrozenLakeEnvironment()
        >>> obs, info = env.reset()
        >>> print(f"Start state: {obs} (row={obs // 4}, col={obs % 4})")
    """

    def __init__(
        self,
        map_desc: list[str] | None = None,
        is_slippery: bool = True,
        step_penalty: float = -0.04,
        goal_reward: float = 1.0,
        hole_penalty: float = -1.0,
        discount_factor: float = 0.99,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the FrozenLake environment.

        Args:
            map_desc: Grid layout. If None, uses the default 4x4 map.
            is_slippery: If True, actions are stochastic (1/3 intended,
                1/3 each perpendicular). If False, deterministic.
            step_penalty: Per-step cost at non-terminal states.
            goal_reward: Reward for reaching the goal.
            hole_penalty: Penalty for falling into a hole.
            discount_factor: Time discount factor.
            scale_parameter: Logit scale parameter.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        if map_desc is None:
            map_desc = DEFAULT_MAP

        self._nrow, self._ncol, self._holes, self._start, self._goal = _parse_map(map_desc)
        self._n_states = self._nrow * self._ncol
        self._is_slippery = is_slippery
        self._step_penalty = step_penalty
        self._goal_reward = goal_reward
        self._hole_penalty = hole_penalty
        self._terminal_states = self._holes | {self._goal}

        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(4)

        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @property
    def num_states(self) -> int:
        return self._n_states

    @property
    def num_actions(self) -> int:
        return 4

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "step_penalty": self._step_penalty,
            "goal_reward": self._goal_reward,
            "hole_penalty": self._hole_penalty,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["step_penalty", "goal_reward", "hole_penalty"]

    @property
    def holes(self) -> set[int]:
        """Set of hole state indices."""
        return self._holes

    @property
    def goal(self) -> int:
        """Goal state index."""
        return self._goal

    @property
    def start(self) -> int:
        """Start state index."""
        return self._start

    def _get_next_state_deterministic(self, state: int, action: int) -> int:
        """Compute deterministic next state for one action."""
        if state in self._terminal_states:
            return state

        row = state // self._ncol
        col = state % self._ncol

        if action == LEFT:
            col = max(col - 1, 0)
        elif action == DOWN:
            row = min(row + 1, self._nrow - 1)
        elif action == RIGHT:
            col = min(col + 1, self._ncol - 1)
        elif action == UP:
            row = max(row - 1, 0)

        return row * self._ncol + col

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition matrices with optional slipperiness.

        For is_slippery=True, each action has 1/3 probability of executing
        as intended and 1/3 probability each for the two perpendicular
        directions. For is_slippery=False, transitions are deterministic.
        """
        n_states = self._n_states
        transitions = np.zeros((4, n_states, n_states), dtype=np.float32)

        for a in range(4):
            for s in range(n_states):
                if self._is_slippery and s not in self._terminal_states:
                    # Perpendicular actions: for Left/Right, perpendicular is Up/Down
                    # For Up/Down, perpendicular is Left/Right
                    if a in (LEFT, RIGHT):
                        candidates = [a, UP, DOWN]
                    else:
                        candidates = [a, LEFT, RIGHT]
                    for c in candidates:
                        ns = self._get_next_state_deterministic(s, c)
                        transitions[a, s, ns] += 1.0 / 3.0
                else:
                    ns = self._get_next_state_deterministic(s, a)
                    transitions[a, s, ns] = 1.0

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Three features:
        - step_penalty indicator (1.0 at non-terminal states)
        - goal_reward indicator (1.0 when next state is goal)
        - hole_penalty indicator (1.0 when next state is hole)
        """
        n_states = self._n_states
        features = np.zeros((n_states, 4, 3), dtype=np.float32)

        for s in range(n_states):
            is_terminal = s in self._terminal_states
            for a in range(4):
                if not is_terminal:
                    features[s, a, 0] = 1.0

                # For stochastic transitions, weight by transition probs
                if self._is_slippery and not is_terminal:
                    if a in (LEFT, RIGHT):
                        candidates = [a, UP, DOWN]
                    else:
                        candidates = [a, LEFT, RIGHT]
                    for c in candidates:
                        ns = self._get_next_state_deterministic(s, c)
                        if ns == self._goal:
                            features[s, a, 1] += 1.0 / 3.0
                        if ns in self._holes:
                            features[s, a, 2] += 1.0 / 3.0
                else:
                    ns = self._get_next_state_deterministic(s, a)
                    if ns == self._goal:
                        features[s, a, 1] = 1.0
                    if ns in self._holes:
                        features[s, a, 2] = 1.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Start at the start state (top-left corner by default)."""
        dist = np.zeros(self._n_states)
        dist[self._start] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        is_terminal = state in self._terminal_states
        if is_terminal:
            return 0.0

        utility = self._step_penalty

        ns = self._get_next_state_deterministic(state, action)
        if ns == self._goal:
            utility += self._goal_reward
        if ns in self._holes:
            utility += self._hole_penalty

        return utility

    def _sample_next_state(self, state: int, action: int) -> int:
        if state in self._terminal_states:
            return state

        if self._is_slippery:
            if action in (LEFT, RIGHT):
                candidates = [action, UP, DOWN]
            else:
                candidates = [action, LEFT, RIGHT]
            chosen = self._np_random.choice(candidates)
            return self._get_next_state_deterministic(state, chosen)
        else:
            return self._get_next_state_deterministic(state, action)

    def state_to_grid_position(self, state: int) -> tuple[int, int]:
        """Convert state index to (row, col)."""
        return state // self._ncol, state % self._ncol

    def describe(self) -> str:
        N = self._nrow
        return f"""FrozenLake Environment ({N}x{N})
{'=' * 40}
States: {self.num_states} ({N}x{N} grid)
Actions: Left (0), Down (1), Right (2), Up (3)
Holes: {sorted(self._holes)}
Goal: {self._goal}
Slippery: {self._is_slippery}

True Parameters:
  Step penalty: {self._step_penalty}
  Goal reward:  {self._goal_reward}
  Hole penalty: {self._hole_penalty}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
