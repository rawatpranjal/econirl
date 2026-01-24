"""Sklearn-style utility classes for dynamic discrete choice models.

This module provides a simpler interface for defining utility functions
compared to the existing UtilityFunction protocol. These classes are
designed to work with the new sklearn-style estimators.

The main components are:
- `Utility`: Abstract base class defining the interface
- `LinearCost`: Built-in utility for the Rust bus replacement model
- `CallableUtility`: Wrapper for custom utility functions
- `make_utility`: Factory function for creating utilities from callables

Example:
    >>> from econirl.utilities import LinearCost, make_utility
    >>> import numpy as np
    >>>
    >>> # Use built-in utility
    >>> utility = LinearCost()
    >>> params = np.array([0.001, 3.0])
    >>> u = utility(state=10, action=0, params=params)
    >>>
    >>> # Or create custom utility
    >>> def my_utility(state, action, params):
    ...     return -params[0] * state - params[1] * action
    >>> utility = make_utility(my_utility, n_params=2, param_names=["cost", "action_cost"])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Utility(ABC):
    """Abstract base class for utility functions.

    A utility function maps (state, action, parameters) to a utility value.
    This is the core component that estimators optimize over.

    Subclasses must implement:
    - `n_params`: Number of parameters
    - `param_names`: Names of parameters
    - `__call__`: Compute utility for given state, action, and parameters

    Optional overrides:
    - `param_bounds`: Parameter bounds for optimization
    - `param_init`: Initial parameter values
    - `matrix`: Compute utility matrix (default uses __call__)
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Number of utility parameters."""
        ...

    @property
    @abstractmethod
    def param_names(self) -> list[str]:
        """Names of utility parameters in order."""
        ...

    @property
    def param_bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Lower and upper bounds for parameters.

        Returns:
            Tuple of (lower_bounds, upper_bounds), each of shape (n_params,).
            Default is (-inf, inf) for all parameters.
        """
        lower = np.full(self.n_params, float("-inf"))
        upper = np.full(self.n_params, float("inf"))
        return lower, upper

    @property
    def param_init(self) -> NDArray[np.floating]:
        """Initial parameter values for optimization.

        Returns:
            Array of shape (n_params,) with initial values.
            Default is zeros.
        """
        return np.zeros(self.n_params)

    @abstractmethod
    def __call__(
        self,
        state: ArrayLike,
        action: ArrayLike,
        params: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute utility for given state, action, and parameters.

        Args:
            state: State value(s). Can be scalar or array.
            action: Action value(s). Can be scalar or array.
            params: Parameter vector of shape (n_params,).

        Returns:
            Utility value(s). Same shape as broadcast of state and action.
        """
        ...

    def matrix(
        self,
        n_states: int,
        params: NDArray[np.floating],
        n_actions: int = 2,
    ) -> NDArray[np.floating]:
        """Compute utility matrix for all state-action pairs.

        Args:
            n_states: Number of states (states are 0, 1, ..., n_states-1).
            params: Parameter vector of shape (n_params,).
            n_actions: Number of actions (default 2 for binary choice).

        Returns:
            Utility matrix of shape (n_states, n_actions) where
            result[s, a] = u(s, a; params).
        """
        states = np.arange(n_states)
        result = np.zeros((n_states, n_actions))

        for a in range(n_actions):
            result[:, a] = self(state=states, action=a, params=params)

        return result


class LinearCost(Utility):
    """Linear cost utility for the Rust bus replacement model.

    Implements the utility specification from Rust (1987):
        u(s, a; theta_c, RC) = -theta_c * s * (1-a) - RC * a

    Where:
    - s is the mileage state
    - a is the action (0=don't replace, 1=replace)
    - theta_c is the per-period operating cost coefficient
    - RC is the replacement cost

    When a=0 (don't replace): u = -theta_c * s (operating cost increases with mileage)
    When a=1 (replace): u = -RC (pay replacement cost, mileage resets)
    """

    @property
    def n_params(self) -> int:
        """Two parameters: theta_c and RC."""
        return 2

    @property
    def param_names(self) -> list[str]:
        """Parameter names: theta_c (operating cost) and RC (replacement cost)."""
        return ["theta_c", "RC"]

    @property
    def param_bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Parameter bounds enforcing non-negativity for theta_c and RC."""
        lower = np.array([0.0, 0.0])
        upper = np.array([np.inf, np.inf])
        return lower, upper

    @property
    def param_init(self) -> NDArray[np.floating]:
        """Reasonable initial values for optimization."""
        return np.array([0.001, 5.0])

    def __call__(
        self,
        state: ArrayLike,
        action: ArrayLike,
        params: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute utility: u = -theta_c * s * (1-a) - RC * a.

        Args:
            state: Mileage state(s).
            action: Action(s) (0=don't replace, 1=replace).
            params: [theta_c, RC] parameter vector.

        Returns:
            Utility value(s).
        """
        state = np.asarray(state)
        action = np.asarray(action)

        theta_c = params[0]
        rc = params[1]

        # u = -theta_c * s * (1-a) - RC * a
        return -theta_c * state * (1 - action) - rc * action


class CallableUtility(Utility):
    """Wrapper that turns a callable into a Utility object.

    This allows users to define custom utility functions as simple
    Python functions while still conforming to the Utility interface.

    Example:
        >>> def my_utility(state, action, params):
        ...     return -params[0] * state - params[1] * action
        >>>
        >>> utility = CallableUtility(
        ...     fn=my_utility,
        ...     n_params=2,
        ...     param_names=["cost", "action_cost"],
        ... )
    """

    def __init__(
        self,
        fn: Callable[[ArrayLike, ArrayLike, NDArray[np.floating]], ArrayLike],
        n_params: int,
        param_names: list[str] | None = None,
        param_bounds: tuple[ArrayLike, ArrayLike] | None = None,
        param_init: ArrayLike | None = None,
    ):
        """Initialize CallableUtility.

        Args:
            fn: Utility function with signature (state, action, params) -> utility.
            n_params: Number of parameters.
            param_names: Names of parameters. If None, uses ["theta_0", "theta_1", ...].
            param_bounds: Optional (lower, upper) bounds tuple.
            param_init: Optional initial parameter values.
        """
        self._fn = fn
        self._n_params = n_params

        if param_names is None:
            self._param_names = [f"theta_{i}" for i in range(n_params)]
        else:
            if len(param_names) != n_params:
                raise ValueError(
                    f"param_names must have {n_params} elements, got {len(param_names)}"
                )
            self._param_names = list(param_names)

        if param_bounds is not None:
            self._param_bounds = (
                np.asarray(param_bounds[0]),
                np.asarray(param_bounds[1]),
            )
        else:
            self._param_bounds = None

        if param_init is not None:
            self._param_init = np.asarray(param_init)
        else:
            self._param_init = None

    @property
    def n_params(self) -> int:
        """Number of parameters."""
        return self._n_params

    @property
    def param_names(self) -> list[str]:
        """Parameter names."""
        return self._param_names.copy()

    @property
    def param_bounds(self) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Parameter bounds."""
        if self._param_bounds is not None:
            return self._param_bounds
        return super().param_bounds

    @property
    def param_init(self) -> NDArray[np.floating]:
        """Initial parameter values."""
        if self._param_init is not None:
            return self._param_init.copy()
        return super().param_init

    def __call__(
        self,
        state: ArrayLike,
        action: ArrayLike,
        params: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Compute utility using the wrapped function."""
        result = self._fn(state, action, params)
        return np.asarray(result)


def make_utility(
    fn: Callable[[ArrayLike, ArrayLike, NDArray[np.floating]], ArrayLike],
    n_params: int,
    param_names: list[str] | None = None,
    param_bounds: tuple[ArrayLike, ArrayLike] | None = None,
    param_init: ArrayLike | None = None,
) -> CallableUtility:
    """Factory function to create a Utility from a callable.

    This is a convenience function equivalent to constructing CallableUtility
    directly.

    Args:
        fn: Utility function with signature (state, action, params) -> utility.
        n_params: Number of parameters.
        param_names: Names of parameters. If None, uses ["theta_0", "theta_1", ...].
        param_bounds: Optional (lower, upper) bounds tuple.
        param_init: Optional initial parameter values.

    Returns:
        CallableUtility wrapping the provided function.

    Example:
        >>> def my_utility(state, action, params):
        ...     return -params[0] * state
        >>>
        >>> utility = make_utility(my_utility, n_params=1, param_names=["cost"])
    """
    return CallableUtility(
        fn=fn,
        n_params=n_params,
        param_names=param_names,
        param_bounds=param_bounds,
        param_init=param_init,
    )
