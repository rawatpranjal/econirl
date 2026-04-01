"""SCANIA Component X replacement environment.

This module implements a dynamic discrete choice model for heavy truck
component replacement, inspired by the SCANIA Component X dataset from
the IDA 2024 Industrial Challenge. The decision problem is structurally
identical to Rust (1987) but set in a predictive maintenance context
where degradation replaces mileage as the state variable.

The model:
- State: Discretized degradation level (composite index of operational wear)
- Actions: Keep operating (0) or Replace component (1)
- Utility: Operating cost increases with degradation; replacement has fixed cost
- Transitions: Degradation increases stochastically; replacement resets to zero

The default parameters are calibrated to produce a replacement rate of
roughly 10 percent, matching the observed repair frequency in the SCANIA
dataset (2,272 repairs out of 23,550 vehicles).

Reference:
    SCANIA Component X dataset, IDA 2024 Industrial Challenge.
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


class ScaniaComponentEnvironment(DDCEnvironment):
    """SCANIA Component X replacement environment.

    A fleet manager observes the degradation level of a heavy truck
    component each period and decides whether to continue operating
    or replace the component. Replacing incurs a fixed cost but resets
    degradation to zero. Continued operation incurs a per-period cost
    that grows with degradation.

    State space:
        Degradation discretized into bins {0, 1, ..., num_bins - 1}.
        Each bin represents a degradation increment derived from the
        14 anonymized operational readout variables in the SCANIA data.

    Action space:
        0 = Keep: Continue operating with current component
        1 = Replace: Install new component (degradation resets to 0)

    Utility specification:
        U(s, keep) = -operating_cost * degradation(s) + epsilon_keep
        U(s, replace) = -replacement_cost + epsilon_replace

    where epsilon are i.i.d. Type I Extreme Value shocks.

    Transition dynamics:
        If keep: degradation increases by {0, 1, 2} with probabilities
            (theta_0, theta_1, theta_2), capped at the maximum bin.
        If replace: degradation resets to 0, then increases by {0, 1, 2}.

    Example:
        >>> env = ScaniaComponentEnvironment()
        >>> obs, info = env.reset()
        >>> print(f"Initial degradation bin: {obs}")
    """

    KEEP = 0
    REPLACE = 1

    def __init__(
        self,
        operating_cost: float = 0.002,
        replacement_cost: float = 4.0,
        num_degradation_bins: int = 50,
        degradation_transition_probs: tuple[float, float, float] = (0.35, 0.55, 0.10),
        discount_factor: float = 0.9999,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the SCANIA component replacement environment.

        Args:
            operating_cost: Cost per unit degradation for operating.
            replacement_cost: Fixed cost of component replacement.
            num_degradation_bins: Number of degradation discretization bins.
            degradation_transition_probs: Probabilities of degradation
                increase (0, 1, or 2 bins per period). Must sum to 1.
                Default reflects slightly faster wear than Rust buses.
            discount_factor: Time discount factor beta.
            scale_parameter: Logit scale parameter sigma.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._operating_cost = operating_cost
        self._replacement_cost = replacement_cost
        self._num_degradation_bins = num_degradation_bins
        self._degradation_transition_probs = np.array(degradation_transition_probs)

        if len(self._degradation_transition_probs) != 3:
            raise ValueError("degradation_transition_probs must have exactly 3 elements")
        if not np.isclose(self._degradation_transition_probs.sum(), 1.0, atol=1e-4):
            raise ValueError("degradation_transition_probs must sum to 1")

        self.observation_space = spaces.Discrete(num_degradation_bins)
        self.action_space = spaces.Discrete(2)

        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @property
    def num_states(self) -> int:
        return self._num_degradation_bins

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
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
    def degradation_transition_probs(self) -> np.ndarray:
        """Return the degradation transition probabilities."""
        return self._degradation_transition_probs.copy()

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition probability matrices P(s'|s,a).

        Returns:
            Tensor of shape (2, num_states, num_states)
            - transitions[0, s, s'] = P(s' | s, keep)
            - transitions[1, s, s'] = P(s' | s, replace)
        """
        n = self._num_degradation_bins
        p = self._degradation_transition_probs

        transitions = np.zeros((2, n, n), dtype=np.float32)

        # Keep action: degradation increases by 0, 1, or 2
        for s in range(n):
            for delta, prob in enumerate(p):
                next_s = min(s + delta, n - 1)
                transitions[self.KEEP, s, next_s] += prob

        # Replace action: reset to 0, then increase by 0, 1, or 2
        for delta, prob in enumerate(p):
            next_s = min(delta, n - 1)
            transitions[self.REPLACE, :, next_s] = prob

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for utility computation.

        Features are structured so that U(s, a) = theta dot phi(s, a)
        where theta = [operating_cost, replacement_cost].

        Returns:
            Tensor of shape (num_states, num_actions, num_features)
        """
        n = self._num_degradation_bins

        features = np.zeros((n, 2, 2), dtype=np.float32)

        degradation = np.arange(n, dtype=np.float32)

        # Keep action (a=0): utility = -operating_cost * degradation
        features[:, self.KEEP, 0] = -degradation
        features[:, self.KEEP, 1] = 0.0

        # Replace action (a=1): utility = -replacement_cost
        features[:, self.REPLACE, 0] = 0.0
        features[:, self.REPLACE, 1] = -1.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return initial state distribution (start at degradation 0)."""
        dist = np.zeros(self._num_degradation_bins)
        dist[0] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility for state-action pair."""
        if action == self.KEEP:
            return -self._operating_cost * state
        else:
            return -self._replacement_cost

    def _sample_next_state(self, state: int, action: int) -> int:
        """Sample next state given current state and action."""
        if action == self.REPLACE:
            base_state = 0
        else:
            base_state = state

        delta = self._np_random.choice(3, p=self._degradation_transition_probs)
        next_state = min(base_state + delta, self._num_degradation_bins - 1)
        return int(next_state)

    def _state_to_record(self, state: int, action: int) -> dict[str, Any]:
        """Convert state-action pair to human-readable record fields."""
        return {
            "degradation_bin": state,
            "replaced": int(action == self.REPLACE),
        }

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        return f"""SCANIA Component X Replacement Environment
=============================================
States: {self.num_states} degradation bins (0 to {self.num_states - 1})
Actions: Keep (0), Replace (1)

True Parameters:
  Operating cost: {self._operating_cost}
  Replacement cost (RC): {self._replacement_cost}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}

Degradation Transition Probabilities:
  P(+0 bins): {self._degradation_transition_probs[0]:.4f}
  P(+1 bin):  {self._degradation_transition_probs[1]:.4f}
  P(+2 bins): {self._degradation_transition_probs[2]:.4f}
"""

    @classmethod
    def info(cls) -> dict:
        """Return metadata about this environment."""
        return {
            "name": "SCANIA Component X Replacement",
            "description": (
                "Heavy truck component replacement decision inspired by "
                "the SCANIA IDA 2024 Industrial Challenge dataset."
            ),
            "source": "SCANIA Component X, IDA 2024 Industrial Challenge",
            "n_states": 50,
            "n_actions": 2,
            "parameters": ["operating_cost", "replacement_cost"],
            "reference": "Rust (1987) structural framework applied to SCANIA data",
        }
