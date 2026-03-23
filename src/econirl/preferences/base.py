"""Base protocol for utility function specifications.

This module defines the UtilityFunction protocol that all utility
specifications must implement. Utility functions map state-action
pairs and parameters to flow utility values.

The design follows the discrete choice literature where:
    U(s, a; θ) = deterministic_utility(s, a; θ) + ε(a)

where ε(a) are i.i.d. preference shocks (typically Type I Extreme Value).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class UtilityFunction(Protocol):
    """Protocol defining the interface for utility functions.

    A UtilityFunction specifies how parameters θ map to flow utilities
    U(s, a; θ) for all state-action pairs. This is the object that
    estimators optimize over.

    The protocol requires:
    - `num_parameters`: Number of parameters to estimate
    - `parameter_names`: Human-readable names for each parameter
    - `compute()`: Evaluate utility for given parameters
    - `compute_gradient()`: Gradient of utility w.r.t. parameters

    Example:
        >>> utility = LinearUtility(feature_matrix, parameter_names=["cost", "benefit"])
        >>> U = utility.compute(theta)  # shape (num_states, num_actions)
    """

    @property
    def num_parameters(self) -> int:
        """Number of utility parameters to estimate."""
        ...

    @property
    def parameter_names(self) -> list[str]:
        """Names of utility parameters in order."""
        ...

    @property
    def num_states(self) -> int:
        """Number of states in the model."""
        ...

    @property
    def num_actions(self) -> int:
        """Number of actions in the model."""
        ...

    def compute(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute flow utility matrix for given parameters.

        Args:
            parameters: Parameter vector θ of shape (num_parameters,)

        Returns:
            Utility matrix of shape (num_states, num_actions)
            where result[s, a] = U(s, a; θ)
        """
        ...

    def compute_gradient(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute gradient of utility w.r.t. parameters.

        For linear utility U = θ·φ, the gradient is simply the
        feature matrix φ.

        Args:
            parameters: Parameter vector θ of shape (num_parameters,)

        Returns:
            Gradient tensor of shape (num_states, num_actions, num_parameters)
            where result[s, a, k] = ∂U(s,a;θ)/∂θ_k
        """
        ...


class BaseUtilityFunction(ABC):
    """Abstract base class for utility functions.

    Provides common functionality and enforces the UtilityFunction protocol.
    Subclasses must implement the abstract methods.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        parameter_names: list[str],
        anchor_action: int | None = None,
    ):
        """Initialize the utility function.

        Args:
            num_states: Number of states
            num_actions: Number of actions
            parameter_names: Names for each parameter
            anchor_action: Action to normalize for identification (optional).
                          If set, utility of this action is normalized to 0
                          in all states (standard practice in discrete choice).
        """
        self._num_states = num_states
        self._num_actions = num_actions
        self._parameter_names = list(parameter_names)
        self._anchor_action = anchor_action

    @property
    def num_parameters(self) -> int:
        return len(self._parameter_names)

    @property
    def parameter_names(self) -> list[str]:
        return self._parameter_names.copy()

    @property
    def num_states(self) -> int:
        return self._num_states

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def anchor_action(self) -> int | None:
        """Action used for normalization (if any)."""
        return self._anchor_action

    @abstractmethod
    def compute(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute utility matrix. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def compute_gradient(self, parameters: torch.Tensor) -> torch.Tensor:
        """Compute gradient tensor. Must be implemented by subclasses."""
        ...

    def get_initial_parameters(self) -> torch.Tensor:
        """Return reasonable initial parameter values for optimization.

        Default implementation returns zeros. Subclasses may override
        with better starting points.
        """
        return torch.zeros(self.num_parameters, dtype=torch.float32)

    def get_parameter_bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower and upper bounds for parameters.

        Default implementation returns (-inf, inf) for all parameters.
        Subclasses may override to impose constraints.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each of shape (num_parameters,)
        """
        lower = torch.full((self.num_parameters,), float("-inf"))
        upper = torch.full((self.num_parameters,), float("inf"))
        return lower, upper

    def validate_parameters(self, parameters: torch.Tensor) -> None:
        """Validate that parameters have correct shape.

        Args:
            parameters: Parameter tensor to validate

        Raises:
            ValueError: If parameters have wrong shape
        """
        if parameters.shape != (self.num_parameters,):
            raise ValueError(
                f"Expected parameters of shape ({self.num_parameters},), "
                f"got {parameters.shape}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_states={self.num_states}, "
            f"num_actions={self.num_actions}, "
            f"parameters={self.parameter_names})"
        )
