"""Base class for dynamic discrete choice environments.

This module defines DDCEnvironment, a Gymnasium-compatible base class
for economic decision environments. It extends gym.Env with properties
and methods specific to structural estimation.

Key additions over standard Gym environments:
- Explicit transition matrices (for model-based estimation)
- Feature matrix for utility computation
- Problem specification (DDCProblem)
- True parameter access (for simulation studies)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, SupportsFloat, Union

import gymnasium as gym
import jax.numpy as jnp
import numpy as np
import pandas as pd
from gymnasium import spaces

from econirl.core.types import DDCProblem, Panel


class DDCEnvironment(gym.Env, ABC):
    """Base class for dynamic discrete choice environments.

    This abstract class defines the interface for environments used in
    structural estimation. Subclasses must implement the abstract properties
    and methods to define specific economic models.

    DDCEnvironment extends gym.Env with:
    - `problem_spec`: The DDCProblem specification
    - `transition_matrices`: Explicit P(s'|s,a) for model-based methods
    - `feature_matrix`: Features φ(s,a) for utility computation
    - `true_parameters`: Ground truth (for simulation studies)

    The environment can be used both for:
    1. Simulation: Generate data from known parameters
    2. Estimation: Provide structure needed by estimators

    Example:
        >>> env = MyDDCEnvironment(params)
        >>> obs, info = env.reset()
        >>> for _ in range(100):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        discount_factor: float = 0.9999,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the environment.

        Args:
            discount_factor: Time discount factor β ∈ [0, 1)
            scale_parameter: Logit scale parameter σ > 0
            seed: Random seed for reproducibility
        """
        super().__init__()
        self._discount_factor = discount_factor
        self._scale_parameter = scale_parameter
        self._np_random = np.random.default_rng(seed)

        # Subclasses should set these
        self._state: int | None = None
        self._current_period: int = 0

    @property
    @abstractmethod
    def num_states(self) -> int:
        """Number of discrete states in the environment."""
        ...

    @property
    @abstractmethod
    def num_actions(self) -> int:
        """Number of discrete actions available."""
        ...

    @property
    def problem_spec(self) -> DDCProblem:
        """Return the DDCProblem specification for this environment."""
        return DDCProblem(
            num_states=self.num_states,
            num_actions=self.num_actions,
            discount_factor=self._discount_factor,
            scale_parameter=self._scale_parameter,
            state_dim=self.state_dim,
            state_encoder=self.encode_states,
        )

    @property
    def state_dim(self) -> int:
        """Dimensionality of the continuous state representation."""
        return 1

    def encode_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Encode flat state indices to continuous features.

        Default: normalized scalar s/(S-1) with shape (batch, 1).
        Override for multi-dimensional environments.
        """
        denom = max(self.num_states - 1, 1)
        return jnp.expand_dims(states.astype(jnp.float32) / denom, axis=-1)

    @property
    @abstractmethod
    def transition_matrices(self) -> jnp.ndarray:
        """Return transition probability matrices.

        Returns:
            Tensor of shape (num_actions, num_states, num_states)
            where result[a, s, s'] = P(s' | s, a)
        """
        ...

    @property
    @abstractmethod
    def feature_matrix(self) -> jnp.ndarray:
        """Return feature matrix for utility computation.

        Features are the observable characteristics that enter the
        utility function: U(s,a;θ) = θ · φ(s,a)

        Returns:
            Tensor of shape (num_states, num_actions, num_features)
        """
        ...

    @property
    @abstractmethod
    def true_parameters(self) -> dict[str, float]:
        """Return the true utility parameters (for simulation studies).

        Returns:
            Dictionary mapping parameter names to values
        """
        ...

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        """Return names of utility parameters in order."""
        ...

    def get_true_parameter_vector(self) -> jnp.ndarray:
        """Return true parameters as a tensor in canonical order."""
        params = self.true_parameters
        return jnp.array(
            [params[name] for name in self.parameter_names], dtype=jnp.float32
        )

    @property
    def current_state(self) -> int | None:
        """Return the current state index."""
        return self._state

    @abstractmethod
    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return the initial state distribution for reset().

        Returns:
            Array of shape (num_states,) with probabilities
        """
        ...

    @abstractmethod
    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute the flow utility for a state-action pair.

        This should use the true_parameters to compute utility.

        Args:
            state: Current state index
            action: Chosen action index

        Returns:
            Flow utility value (before adding preference shock)
        """
        ...

    @abstractmethod
    def _sample_next_state(self, state: int, action: int) -> int:
        """Sample the next state given current state and action.

        Args:
            state: Current state index
            action: Chosen action index

        Returns:
            Next state index
        """
        ...

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            seed: Optional seed for random number generator
            options: Optional configuration options

        Returns:
            Tuple of (initial_observation, info_dict)
        """
        if seed is not None:
            self._np_random = np.random.default_rng(seed)

        # Sample initial state
        init_dist = self._get_initial_state_distribution()
        self._state = int(self._np_random.choice(self.num_states, p=init_dist))
        self._current_period = 0

        return self._state, {"period": self._current_period}

    def step(
        self, action: int
    ) -> tuple[int, SupportsFloat, bool, bool, dict[str, Any]]:
        """Take an action and observe the result.

        Args:
            action: The action to take

        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
            - next_state: The new state after transition
            - reward: The flow utility (reward in RL terms)
            - terminated: Always False (infinite horizon)
            - truncated: Always False (no time limit by default)
            - info: Additional information
        """
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        if not 0 <= action < self.num_actions:
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.num_actions})")

        # Compute flow utility (this is the "reward" in RL terms)
        utility = self._compute_flow_utility(self._state, action)

        # Sample next state
        prev_state = self._state
        self._state = self._sample_next_state(self._state, action)
        self._current_period += 1

        info = {
            "period": self._current_period,
            "prev_state": prev_state,
            "action": action,
            "flow_utility": utility,
        }

        # DDC models are infinite horizon, so never terminated
        return self._state, utility, False, False, info

    def compute_utility_matrix(self, parameters: jnp.ndarray | None = None) -> jnp.ndarray:
        """Compute the full utility matrix for all state-action pairs.

        Args:
            parameters: Optional parameter vector. If None, uses true_parameters.

        Returns:
            Tensor of shape (num_states, num_actions) with flow utilities
        """
        if parameters is None:
            parameters = self.get_true_parameter_vector()

        features = self.feature_matrix
        return jnp.einsum("sak,k->sa", features, parameters)

    def render(self) -> None:
        """Render the current state (optional)."""
        if self._state is not None:
            print(f"Period {self._current_period}: State = {self._state}")

    # --- Synthetic data generation ---

    def generate_panel(
        self,
        n_individuals: int = 1000,
        n_periods: int = 100,
        seed: int = 42,
        as_dataframe: bool = False,
    ) -> Union[Panel, pd.DataFrame]:
        """Generate synthetic panel data from this environment.

        Computes the optimal policy from the true parameters and
        simulates trajectories for multiple individuals.

        Args:
            n_individuals: Number of individuals to simulate.
            n_periods: Number of time periods per individual.
            seed: Random seed for reproducibility.
            as_dataframe: If True, return a DataFrame with
                human-readable columns via _state_to_record().

        Returns:
            Panel object, or DataFrame if as_dataframe=True.
        """
        from econirl.simulation.synthetic import simulate_panel

        panel = simulate_panel(
            self,
            n_individuals=n_individuals,
            n_periods=n_periods,
            seed=seed,
        )

        if not as_dataframe:
            return panel

        records = []
        for traj in panel.trajectories:
            tid = traj.individual_id
            for t in range(len(traj.states)):
                s = int(traj.states[t])
                a = int(traj.actions[t])
                ns = int(traj.next_states[t])
                record = {
                    "individual_id": tid,
                    "period": t,
                    "state": s,
                    "action": a,
                    "next_state": ns,
                }
                record.update(self._state_to_record(s, a))
                records.append(record)

        return pd.DataFrame(records)

    def _state_to_record(self, state: int, action: int) -> dict[str, Any]:
        """Convert a state-action pair to human-readable record fields.

        Subclasses override this to add domain-specific columns like
        profit_bin, incumbent_status, etc. The base implementation
        returns an empty dict.
        """
        return {}

    @classmethod
    def info(cls) -> dict:
        """Return metadata about this environment.

        Subclasses should override this to provide name, description,
        source, n_states, n_actions, parameter details, etc.
        """
        return {
            "name": cls.__name__,
            "description": cls.__doc__.split("\n")[0] if cls.__doc__ else "",
        }
