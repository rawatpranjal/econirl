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

Five cost function specifications are supported, following the ruspy
library (OpenSourceEconomics). The cost_type parameter selects the
functional form of the operating cost: linear, quadratic, cubic,
square root, or hyperbolic.

Reference:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment

# Valid cost function types
CostType = Literal["linear", "quadratic", "cubic", "sqrt", "hyperbolic"]
VALID_COST_TYPES: tuple[str, ...] = ("linear", "quadratic", "cubic", "sqrt", "hyperbolic")


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

    Utility specification (depends on cost_type):
        linear:     U(s, keep) = -theta_1 * s
        quadratic:  U(s, keep) = -theta_1 * s - theta_2 * s^2
        cubic:      U(s, keep) = -theta_1 * s - theta_2 * s^2 - theta_3 * s^3
        sqrt:       U(s, keep) = -theta_1 * sqrt(s)
        hyperbolic: U(s, keep) = -theta_1 / (N+1 - s)

        U(s, replace) = -replacement_cost + ε_replace

    where ε are i.i.d. Type I Extreme Value (Gumbel) shocks.

    Transition dynamics:
        If keep: mileage increases by {0, 1, 2} with probabilities (θ_0, θ_1, θ_2)
        If replace: mileage resets to 0

    Example:
        >>> env = RustBusEnvironment(
        ...     operating_cost=0.001,
        ...     replacement_cost=3.0,
        ...     discount_factor=0.9999,
        ... )
        >>> obs, info = env.reset()
        >>> print(f"Initial mileage bin: {obs}")

        >>> env_quad = RustBusEnvironment(
        ...     cost_type="quadratic",
        ...     operating_cost_params=(0.001, 0.00001),
        ...     replacement_cost=3.0,
        ... )
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
        cost_type: CostType = "linear",
        operating_cost_params: tuple[float, ...] | None = None,
    ):
        """Initialize the Rust bus environment.

        Args:
            operating_cost: Cost per unit mileage for operating (θ_1 in Rust).
                Used as the single operating cost coefficient for linear, sqrt,
                and hyperbolic cost types. Ignored when operating_cost_params
                is provided for quadratic or cubic types.
            replacement_cost: Fixed cost of engine replacement (RC in Rust)
            num_mileage_bins: Number of mileage discretization bins (default 90)
            mileage_transition_probs: Probabilities of mileage increase (0, 1, or 2 bins)
                                     Must sum to 1. Default from Rust (1987) estimates.
            discount_factor: Time discount factor β
            scale_parameter: Logit scale parameter σ
            seed: Random seed for reproducibility
            cost_type: Functional form of operating cost. One of "linear",
                "quadratic", "cubic", "sqrt", "hyperbolic".
            operating_cost_params: Tuple of operating cost coefficients for
                multi-parameter cost types (quadratic, cubic). For quadratic,
                provide (theta_1, theta_2). For cubic, provide (theta_1,
                theta_2, theta_3). If None, uses operating_cost as the
                single coefficient.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        if cost_type not in VALID_COST_TYPES:
            raise ValueError(
                f"cost_type must be one of {VALID_COST_TYPES}, got {cost_type!r}"
            )
        self._cost_type = cost_type

        # Determine operating cost coefficients based on cost_type
        if operating_cost_params is not None:
            self._operating_cost_params = tuple(operating_cost_params)
        else:
            self._operating_cost_params = (operating_cost,)

        expected_n_params = {"linear": 1, "quadratic": 2, "cubic": 3, "sqrt": 1, "hyperbolic": 1}
        n_expected = expected_n_params[cost_type]
        if len(self._operating_cost_params) != n_expected:
            raise ValueError(
                f"cost_type={cost_type!r} expects {n_expected} operating cost "
                f"parameter(s), got {len(self._operating_cost_params)}"
            )

        # Keep operating_cost for backward compatibility (first coefficient)
        self._operating_cost = self._operating_cost_params[0]
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
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix

    @property
    def cost_type(self) -> str:
        """Return the operating cost function type."""
        return self._cost_type

    @property
    def true_parameters(self) -> dict[str, float]:
        params = {}
        names = self._operating_cost_param_names()
        for name, val in zip(names, self._operating_cost_params):
            params[name] = val
        params["replacement_cost"] = self._replacement_cost
        return params

    @property
    def parameter_names(self) -> list[str]:
        return self._operating_cost_param_names() + ["replacement_cost"]

    def _operating_cost_param_names(self) -> list[str]:
        """Return parameter names for the operating cost coefficients."""
        if self._cost_type in ("linear", "sqrt", "hyperbolic"):
            return ["operating_cost"]
        elif self._cost_type == "quadratic":
            return ["operating_cost_1", "operating_cost_2"]
        else:  # cubic
            return ["operating_cost_1", "operating_cost_2", "operating_cost_3"]

    @property
    def mileage_transition_probs(self) -> np.ndarray:
        """Return the mileage transition probabilities."""
        return self._mileage_transition_probs.copy()

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition probability matrices P(s'|s,a).

        Returns:
            Tensor of shape (2, num_states, num_states)
            - transitions[0, s, s'] = P(s' | s, keep)
            - transitions[1, s, s'] = P(s' | s, replace)
        """
        n = self._num_mileage_bins
        p = self._mileage_transition_probs

        # Build with numpy then convert (JAX arrays are immutable)
        transitions = np.zeros((2, n, n), dtype=np.float32)

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

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for utility computation.

        Features are structured so that U(s, a) = theta dot phi(s, a).
        The number of features depends on the cost_type. Replacement
        cost is always the last feature column.

        Returns:
            Tensor of shape (num_states, num_actions, num_features)
        """
        n = self._num_mileage_bins
        n_params = len(self._operating_cost_params) + 1  # +1 for RC
        features = np.zeros((n, 2, n_params), dtype=np.float32)
        s = np.arange(n, dtype=np.float32)

        if self._cost_type == "linear":
            # U(s, keep) = -theta_1 * s
            features[:, self.KEEP, 0] = -s

        elif self._cost_type == "quadratic":
            # U(s, keep) = -theta_1 * s - theta_2 * s^2
            features[:, self.KEEP, 0] = -s
            features[:, self.KEEP, 1] = -s ** 2

        elif self._cost_type == "cubic":
            # U(s, keep) = -theta_1 * s - theta_2 * s^2 - theta_3 * s^3
            features[:, self.KEEP, 0] = -s
            features[:, self.KEEP, 1] = -s ** 2
            features[:, self.KEEP, 2] = -s ** 3

        elif self._cost_type == "sqrt":
            # U(s, keep) = -theta_1 * sqrt(s)
            features[:, self.KEEP, 0] = -np.sqrt(s)

        elif self._cost_type == "hyperbolic":
            # U(s, keep) = -theta_1 / (N+1 - s)
            features[:, self.KEEP, 0] = -1.0 / (n + 1.0 - s)

        # Replace action: utility = -replacement_cost (last feature column)
        features[:, self.REPLACE, -1] = -1.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return initial state distribution (start at mileage 0)."""
        dist = np.zeros(self._num_mileage_bins)
        dist[0] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility for state-action pair."""
        if action == self.REPLACE:
            return -self._replacement_cost

        # Keep action: cost depends on cost_type
        s = float(state)
        p = self._operating_cost_params
        if self._cost_type == "linear":
            return -p[0] * s
        elif self._cost_type == "quadratic":
            return -p[0] * s - p[1] * s ** 2
        elif self._cost_type == "cubic":
            return -p[0] * s - p[1] * s ** 2 - p[2] * s ** 3
        elif self._cost_type == "sqrt":
            return -p[0] * (s ** 0.5)
        else:  # hyperbolic
            n = self._num_mileage_bins
            return -p[0] / (n + 1.0 - s)

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
        param_lines = []
        for name, val in self.true_parameters.items():
            param_lines.append(f"  {name}: {val}")
        params_str = "\n".join(param_lines)

        return f"""Rust (1987) Bus Engine Replacement Environment
===============================================
States: {self.num_states} mileage bins (0 to {self.num_states - 1})
Actions: Keep (0), Replace (1)
Cost type: {self._cost_type}

True Parameters:
{params_str}

Structural Parameters:
  Discount factor (β): {self._discount_factor}
  Scale parameter (σ): {self._scale_parameter}

Mileage Transition Probabilities:
  P(+0 bins): {self._mileage_transition_probs[0]:.4f}
  P(+1 bin):  {self._mileage_transition_probs[1]:.4f}
  P(+2 bins): {self._mileage_transition_probs[2]:.4f}
"""
