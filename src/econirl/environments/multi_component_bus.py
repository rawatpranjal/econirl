"""Multi-component bus engine replacement environment.

This module extends Rust (1987) to K independent engine components, each
tracked with M mileage bins. The state space is M^K via mixed-radix
encoding.

State feature x(s) = sum_k(m_k / M), where m_k is the mileage bin for
component k.

Actions:
    0 = Keep: continue operating all components
    1 = Replace: replace all components (reset to state 0)

Features (3 per state-action pair):
    Keep:    [0, -x(s), -x(s)^2]
    Replace: [-1, 0, 0]

Parameters: ["replacement_cost", "operating_cost", "quadratic_cost"]

The transition matrix is built as a dense tensor of shape (2, M^K, M^K).
For K >= 4 this is prohibitively large and raises ValueError.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


class MultiComponentBusEnvironment(DDCEnvironment):
    """Multi-component bus engine replacement environment.

    Extends Rust (1987) to K independent components, each with M mileage
    bins. The total state space has M^K states encoded via mixed-radix.

    Example:
        >>> env = MultiComponentBusEnvironment(K=2, M=10)
        >>> obs, info = env.reset()
        >>> print(f"State space size: {env.num_states}")
        State space size: 100
    """

    KEEP = 0
    REPLACE = 1

    def __init__(
        self,
        K: int = 2,
        M: int = 20,
        operating_cost: float = 0.001,
        quadratic_cost: float = 0.0005,
        replacement_cost: float = 3.0,
        mileage_transition_probs: tuple[float, float, float] = (0.3919, 0.5953, 0.0128),
        discount_factor: float = 0.9999,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the multi-component bus environment.

        Args:
            K: Number of independent engine components (1-3).
            M: Number of mileage bins per component.
            operating_cost: Linear operating cost coefficient.
            quadratic_cost: Quadratic operating cost coefficient.
            replacement_cost: Fixed cost of replacing all components.
            mileage_transition_probs: Probabilities of mileage increase
                (0, 1, or 2 bins) per component per period. Must sum to 1.
            discount_factor: Time discount factor beta in [0, 1).
            scale_parameter: Logit scale parameter sigma > 0.
            seed: Random seed for reproducibility.

        Raises:
            ValueError: If K >= 4 (state space too large for dense tensor).
            ValueError: If K < 1 or M < 2.
            ValueError: If mileage_transition_probs has wrong length or
                does not sum to 1.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        if K >= 4:
            raise ValueError(
                f"K must be <= 3 for dense transition matrices, got K={K}"
            )
        if K < 1:
            raise ValueError(f"K must be >= 1, got K={K}")
        if M < 2:
            raise ValueError(f"M must be >= 2, got M={M}")

        self._K = K
        self._M = M
        self._operating_cost = operating_cost
        self._quadratic_cost = quadratic_cost
        self._replacement_cost = replacement_cost
        self._mileage_transition_probs = np.array(mileage_transition_probs)

        # Validate transition probabilities
        if len(self._mileage_transition_probs) != 3:
            raise ValueError("mileage_transition_probs must have exactly 3 elements")
        if not np.isclose(self._mileage_transition_probs.sum(), 1.0, atol=1e-4):
            raise ValueError("mileage_transition_probs must sum to 1")

        self._n_states = M ** K

        # Set up Gymnasium spaces
        self.observation_space = spaces.Discrete(self._n_states)
        self.action_space = spaces.Discrete(2)

        # Pre-compute transition matrices and features
        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    # ------------------------------------------------------------------
    # Mixed-radix encoding helpers
    # ------------------------------------------------------------------

    def state_to_components(self, state: int) -> list[int]:
        """Decode a flat state index into per-component mileage bins.

        Args:
            state: Flat state index in [0, M^K).

        Returns:
            List of K component mileage bins, each in [0, M).
        """
        components = []
        s = state
        for _ in range(self._K):
            components.append(s % self._M)
            s //= self._M
        return components

    def components_to_state(self, components: list[int]) -> int:
        """Encode per-component mileage bins into a flat state index.

        Args:
            components: List of K component mileage bins.

        Returns:
            Flat state index.
        """
        state = 0
        for k in reversed(range(len(components))):
            state = state * self._M + components[k]
        return state

    def _state_feature(self, state: int) -> float:
        """Compute the aggregate state feature x(s) = sum_k(m_k / M)."""
        components = self.state_to_components(state)
        return sum(m / self._M for m in components)

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._n_states

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def transition_matrices(self) -> torch.Tensor:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> torch.Tensor:
        return self._feature_matrix

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "replacement_cost": self._replacement_cost,
            "operating_cost": self._operating_cost,
            "quadratic_cost": self._quadratic_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["replacement_cost", "operating_cost", "quadratic_cost"]

    # ------------------------------------------------------------------
    # Build methods
    # ------------------------------------------------------------------

    def _build_single_component_transition(self) -> np.ndarray:
        """Build the M x M transition matrix for one component (keep action).

        Returns:
            Array of shape (M, M) where result[m, m'] = P(m' | m, keep).
        """
        M = self._M
        p = self._mileage_transition_probs
        T = np.zeros((M, M), dtype=np.float64)
        for m in range(M):
            for delta, prob in enumerate(p):
                next_m = min(m + delta, M - 1)
                T[m, next_m] += prob
        return T

    def _build_transition_matrices(self) -> torch.Tensor:
        """Build transition matrices P(s'|s,a).

        Returns:
            Tensor of shape (2, M^K, M^K).
        """
        N = self._n_states
        K = self._K
        M = self._M

        transitions = torch.zeros((2, N, N), dtype=torch.float32)

        # ------- Keep action -------
        # Each component evolves independently, so the joint transition
        # is the Kronecker product of per-component transitions.
        # np.kron(A, B) places B's index as fastest-varying. Our
        # mixed-radix encoding has component 0 as the fastest index,
        # so we build kron(T_{K-1}, kron(T_{K-2}, ... kron(T_1, T_0))).
        # With all components identical, this is simply:
        T1 = self._build_single_component_transition()

        joint_keep = T1.copy()
        for _ in range(1, K):
            joint_keep = np.kron(T1, joint_keep)

        transitions[self.KEEP] = torch.from_numpy(
            joint_keep.astype(np.float32)
        )

        # ------- Replace action -------
        # Reset to state 0 (all components at mileage 0), then apply one
        # step of mileage increase (same as the keep transition from state 0).
        # So the replace row for every source state s equals the keep row
        # for state 0.
        keep_from_zero = transitions[self.KEEP, 0, :]  # shape (N,)
        transitions[self.REPLACE] = keep_from_zero.unsqueeze(0).expand(N, -1)

        return transitions

    def _build_feature_matrix(self) -> torch.Tensor:
        """Build feature matrix phi(s, a).

        Returns:
            Tensor of shape (M^K, 2, 3).
            Features: [replacement_cost_indicator, operating_cost, quadratic_cost]
        """
        N = self._n_states
        features = torch.zeros((N, 2, 3), dtype=torch.float32)

        for s in range(N):
            x = self._state_feature(s)

            # Keep action: [0, -x(s), -x(s)^2]
            features[s, self.KEEP, 0] = 0.0
            features[s, self.KEEP, 1] = -x
            features[s, self.KEEP, 2] = -(x ** 2)

            # Replace action: [-1, 0, 0]
            features[s, self.REPLACE, 0] = -1.0
            features[s, self.REPLACE, 1] = 0.0
            features[s, self.REPLACE, 2] = 0.0

        return features

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Start at state 0 (all components at mileage 0)."""
        dist = np.zeros(self._n_states)
        dist[0] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility for a state-action pair."""
        if action == self.REPLACE:
            return -self._replacement_cost
        # Keep
        x = self._state_feature(state)
        return -self._operating_cost * x - self._quadratic_cost * (x ** 2)

    def _sample_next_state(self, state: int, action: int) -> int:
        """Sample next state given current state and action."""
        if action == self.REPLACE:
            components = [0] * self._K
        else:
            components = self.state_to_components(state)

        # Each component transitions independently
        new_components = []
        for m in components:
            delta = self._np_random.choice(3, p=self._mileage_transition_probs)
            new_components.append(min(m + delta, self._M - 1))

        return self.components_to_state(new_components)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def K(self) -> int:
        """Number of engine components."""
        return self._K

    @property
    def M(self) -> int:
        """Number of mileage bins per component."""
        return self._M

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        return f"""Multi-Component Bus Engine Replacement Environment
===================================================
Components (K): {self._K}
Bins per component (M): {self._M}
Total states: {self._n_states}
Actions: Keep (0), Replace (1)

True Parameters:
  Replacement cost: {self._replacement_cost}
  Operating cost: {self._operating_cost}
  Quadratic cost: {self._quadratic_cost}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}

Mileage Transition Probabilities (per component):
  P(+0 bins): {self._mileage_transition_probs[0]:.4f}
  P(+1 bin):  {self._mileage_transition_probs[1]:.4f}
  P(+2 bins): {self._mileage_transition_probs[2]:.4f}
"""
