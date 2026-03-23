"""Rust (1987) bus engine replacement environment.

This module implements the classic dynamic discrete choice model from
John Rust's 1987 Econometrica paper "Optimal Replacement of GMC Bus Engines."

The model:
- State: Discretized mileage (odometer reading in bins)
- Actions: Keep running (0) or Replace engine (1)
- Utility: Operating cost increases with mileage; replacement has fixed cost
- Transitions: Mileage increases stochastically; replacement resets to zero

This is the canonical example for DDC estimation methods and serves as
a benchmark for testing estimators.

Reference:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


class RustBusEnvironment(DDCEnvironment):
    """Rust (1987) bus engine replacement environment.

    Harold Zurcher's decision problem: Each period, observe bus mileage
    and decide whether to keep the current engine or replace it.

    State space:
        Mileage discretized into bins {0, 1, ..., num_mileage_bins - 1}
        Each bin represents approximately 5,000 miles.

    Action space:
        0 = Keep: Continue with current engine
        1 = Replace: Install new engine (mileage resets to 0)

    Utility specification:
        U(s, keep) = -operating_cost * mileage(s) + ε_keep
        U(s, replace) = -replacement_cost + ε_replace

    where ε are i.i.d. Type I Extreme Value (Gumbel) shocks.

    Transition dynamics:
        If keep: mileage increases by {0, 1, 2} with probabilities (θ_0, θ_1, θ_2)
        If replace: mileage resets to 0

    Example:
        >>> env = RustBusEnvironment(
        ...     operating_cost=0.001,
        ...     replacement_cost=3.0,
        ...     discount_factor=0.9999
        ... )
        >>> obs, info = env.reset()
        >>> print(f"Initial mileage bin: {obs}")
    """

    # Action constants for clarity
    KEEP = 0
    REPLACE = 1

    def __init__(
        self,
        operating_cost: float = 0.001,
        replacement_cost: float = 3.0,
        num_mileage_bins: int = 90,
        mileage_transition_probs: tuple[float, float, float] = (0.3919, 0.5953, 0.0128),
        discount_factor: float = 0.9999,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the Rust bus environment.

        Args:
            operating_cost: Cost per unit mileage for operating (θ_1 in Rust)
            replacement_cost: Fixed cost of engine replacement (RC in Rust)
            num_mileage_bins: Number of mileage discretization bins (default 90)
            mileage_transition_probs: Probabilities of mileage increase (0, 1, or 2 bins)
                                     Must sum to 1. Default from Rust (1987) estimates.
            discount_factor: Time discount factor β
            scale_parameter: Logit scale parameter σ
            seed: Random seed for reproducibility
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._operating_cost = operating_cost
        self._replacement_cost = replacement_cost
        self._num_mileage_bins = num_mileage_bins
        self._mileage_transition_probs = np.array(mileage_transition_probs)

        # Validate transition probabilities
        if len(self._mileage_transition_probs) != 3:
            raise ValueError("mileage_transition_probs must have exactly 3 elements")
        if not np.isclose(self._mileage_transition_probs.sum(), 1.0, atol=1e-4):
            raise ValueError("mileage_transition_probs must sum to 1")

        # Set up Gymnasium spaces
        self.observation_space = spaces.Discrete(num_mileage_bins)
        self.action_space = spaces.Discrete(2)  # Keep or Replace

        # Pre-compute transition matrices and features
        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @property
    def num_states(self) -> int:
        return self._num_mileage_bins

    @property
    def num_actions(self) -> int:
        return 2  # Keep, Replace

    @property
    def transition_matrices(self) -> torch.Tensor:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> torch.Tensor:
        return self._feature_matrix

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "operating_cost": self._operating_cost,
            "replacement_cost": self._replacement_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["operating_cost", "replacement_cost"]

    @property
    def mileage_transition_probs(self) -> np.ndarray:
        """Return the mileage transition probabilities."""
        return self._mileage_transition_probs.copy()

    def _build_transition_matrices(self) -> torch.Tensor:
        """Build transition probability matrices P(s'|s,a).

        Returns:
            Tensor of shape (2, num_states, num_states)
            - transitions[0, s, s'] = P(s' | s, keep)
            - transitions[1, s, s'] = P(s' | s, replace)
        """
        n = self._num_mileage_bins
        p = self._mileage_transition_probs

        # Initialize transition matrices
        transitions = torch.zeros((2, n, n), dtype=torch.float32)

        # Keep action: mileage increases by 0, 1, or 2 with given probabilities
        for s in range(n):
            for delta, prob in enumerate(p):
                next_s = min(s + delta, n - 1)  # Cap at max mileage
                transitions[self.KEEP, s, next_s] += prob

        # Replace action: always transition to state 0, then increase
        # After replacement, mileage starts at 0 and increases by 0, 1, or 2
        for delta, prob in enumerate(p):
            next_s = min(delta, n - 1)
            transitions[self.REPLACE, :, next_s] = prob

        return transitions

    def _build_feature_matrix(self) -> torch.Tensor:
        """Build feature matrix for utility computation.

        Features are structured so that:
            U(s, a) = θ · φ(s, a)

        where θ = [operating_cost, replacement_cost]

        For identification, we normalize the replacement action (anchor_action=1).

        Returns:
            Tensor of shape (num_states, num_actions, num_features)
        """
        n = self._num_mileage_bins
        features = torch.zeros((n, 2, 2), dtype=torch.float32)

        # Mileage values (state indices represent mileage bins)
        mileage = torch.arange(n, dtype=torch.float32)

        # Keep action (a=0): utility = -operating_cost * mileage
        # Feature: [-mileage, 0] so that θ·φ = -operating_cost * mileage
        features[:, self.KEEP, 0] = -mileage
        features[:, self.KEEP, 1] = 0.0

        # Replace action (a=1): utility = -replacement_cost
        # Feature: [0, -1] so that θ·φ = -replacement_cost
        features[:, self.REPLACE, 0] = 0.0
        features[:, self.REPLACE, 1] = -1.0

        return features

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return initial state distribution (start at mileage 0)."""
        dist = np.zeros(self._num_mileage_bins)
        dist[0] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility for state-action pair."""
        if action == self.KEEP:
            return -self._operating_cost * state
        else:  # REPLACE
            return -self._replacement_cost

    def _sample_next_state(self, state: int, action: int) -> int:
        """Sample next state given current state and action."""
        if action == self.REPLACE:
            # Reset to 0, then increase
            base_state = 0
        else:
            base_state = state

        # Sample mileage increase
        delta = self._np_random.choice(3, p=self._mileage_transition_probs)
        next_state = min(base_state + delta, self._num_mileage_bins - 1)

        return int(next_state)

    def mileage_to_state(self, mileage: float, bin_size: float = 5000.0) -> int:
        """Convert actual mileage to state index.

        Args:
            mileage: Actual odometer reading
            bin_size: Size of each mileage bin (default 5000 miles)

        Returns:
            State index (capped at num_mileage_bins - 1)
        """
        state = int(mileage / bin_size)
        return min(state, self._num_mileage_bins - 1)

    def state_to_mileage(self, state: int, bin_size: float = 5000.0) -> float:
        """Convert state index to midpoint mileage.

        Args:
            state: State index
            bin_size: Size of each mileage bin

        Returns:
            Midpoint mileage for this bin
        """
        return (state + 0.5) * bin_size

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        return f"""Rust (1987) Bus Engine Replacement Environment
===============================================
States: {self.num_states} mileage bins (0 to {self.num_states - 1})
Actions: Keep (0), Replace (1)

True Parameters:
  Operating cost (θ_c): {self._operating_cost}
  Replacement cost (RC): {self._replacement_cost}

Structural Parameters:
  Discount factor (β): {self._discount_factor}
  Scale parameter (σ): {self._scale_parameter}

Mileage Transition Probabilities:
  P(+0 bins): {self._mileage_transition_probs[0]:.4f}
  P(+1 bin):  {self._mileage_transition_probs[1]:.4f}
  P(+2 bins): {self._mileage_transition_probs[2]:.4f}
"""
