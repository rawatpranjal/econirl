"""Sklearn-style transition probability estimator.

This module provides TransitionEstimator, a scikit-learn style class for
estimating first-stage transition probabilities in Dynamic Discrete Choice models.

The estimator counts mileage bin transitions (excluding replacement periods)
and estimates theta = (theta_0, theta_1, theta_2) where:
- theta_0 = P(stay at same mileage bin)
- theta_1 = P(increase by 1 bin)
- theta_2 = P(increase by 2+ bins)

Reference:
    Rust (1987), Section 4.1, Table IV
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from econirl.core.types import Panel


class TransitionEstimator:
    """Sklearn-style estimator for mileage transition probabilities.

    Estimates the distribution of mileage increments from panel data,
    following the first-stage estimation in Rust (1987).

    Parameters
    ----------
    n_states : int, default=90
        Number of discrete mileage states.
    max_increase : int, default=2
        Maximum mileage bin increase per period. Larger increments are
        clamped to this value.

    Attributes
    ----------
    probs_ : tuple of float
        Estimated probabilities (theta_0, theta_1, theta_2) after fitting.
    matrix_ : ndarray of shape (n_states, n_states)
        Transition probability matrix P(s'|s, a=keep) after fitting.
    n_transitions_ : int
        Number of valid transitions used for estimation.

    Examples
    --------
    >>> from econirl.transitions import TransitionEstimator
    >>> from econirl.simulation.synthetic import simulate_panel
    >>> from econirl.environments.rust_bus import RustBusEnvironment
    >>> env = RustBusEnvironment()
    >>> panel = simulate_panel(env, n_individuals=100, n_periods=100)
    >>> estimator = TransitionEstimator(n_states=90, max_increase=2)
    >>> estimator.fit(panel)
    >>> print(f"theta = {estimator.probs_}")
    >>> print(estimator.summary())
    """

    def __init__(self, n_states: int = 90, max_increase: int = 2) -> None:
        self.n_states = n_states
        self.max_increase = max_increase

        # Fitted attributes (set after fit())
        self.probs_: Tuple[float, float, float] | None = None
        self.matrix_: np.ndarray | None = None
        self.n_transitions_: int | None = None

    def fit(self, data: Panel, state: str | None = None, id: str | None = None,
            action: str | None = None) -> "TransitionEstimator":
        """Fit the transition estimator to panel data.

        Counts state transitions for observations where action=0 (keep/no replacement)
        and estimates the probability distribution over increments.

        Parameters
        ----------
        data : Panel
            Panel data containing trajectories of state-action-next_state.
        state : str, optional
            Ignored. For API compatibility with DataFrame-based methods.
        id : str, optional
            Ignored. For API compatibility with DataFrame-based methods.
        action : str, optional
            Ignored. For API compatibility with DataFrame-based methods.

        Returns
        -------
        self : TransitionEstimator
            Returns self for method chaining.
        """
        # Count increments from valid transitions (action=0, i.e., keep)
        increment_counts = np.zeros(self.max_increase + 1)
        n_transitions = 0

        for traj in data.trajectories:
            states = traj.states.numpy()
            actions = traj.actions.numpy()
            next_states = traj.next_states.numpy()

            for t in range(len(states)):
                # Only count transitions where action is 0 (keep/no replacement)
                if actions[t] == 0:
                    increment = next_states[t] - states[t]
                    # Clamp to valid range [0, max_increase]
                    increment = max(0, min(increment, self.max_increase))
                    increment_counts[increment] += 1
                    n_transitions += 1

        self.n_transitions_ = n_transitions

        # Compute probabilities (handle edge case of no transitions)
        if n_transitions > 0:
            probs = increment_counts / n_transitions
        else:
            # Uniform distribution if no valid transitions
            probs = np.ones(self.max_increase + 1) / (self.max_increase + 1)

        self.probs_ = tuple(probs)

        # Build the transition matrix
        self.matrix_ = self._build_matrix(self.probs_)

        return self

    def _build_matrix(self, probs: Tuple[float, ...]) -> np.ndarray:
        """Build transition matrix from increment probabilities.

        Constructs the state transition matrix P(s'|s, a=keep) where
        the probability of moving from state s to s' depends on the
        increment distribution theta.

        Parameters
        ----------
        probs : tuple of float
            Probabilities (theta_0, theta_1, theta_2, ...) for each increment.

        Returns
        -------
        matrix : ndarray of shape (n_states, n_states)
            Transition probability matrix.
        """
        n = self.n_states
        matrix = np.zeros((n, n))

        for s in range(n):
            for inc, prob in enumerate(probs):
                s_next = s + inc
                if s_next >= n:
                    # Absorbing at last state: accumulate probability at boundary
                    s_next = n - 1
                matrix[s, s_next] += prob

        return matrix

    def summary(self) -> str:
        """Return a formatted summary of the estimated transition probabilities.

        Returns
        -------
        summary : str
            Human-readable summary of the estimation results.
        """
        if self.probs_ is None:
            return "TransitionEstimator: not fitted yet"

        lines = [
            "Transition Probability Estimation",
            "=" * 35,
            "",
            f"Number of transitions: {self.n_transitions_:,}",
            f"Number of states: {self.n_states}",
            f"Max increase: {self.max_increase}",
            "",
            "Estimated probabilities:",
        ]

        for i, prob in enumerate(self.probs_):
            lines.append(f"  theta_{i} (increment={i}): {prob:.4f}")

        lines.append("")
        lines.append(f"Sum of probabilities: {sum(self.probs_):.6f}")

        return "\n".join(lines)
