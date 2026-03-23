"""Core data types for dynamic discrete choice models.

This module defines the fundamental data structures used throughout econirl:
- DDCProblem: Specification of a discrete choice problem
- Trajectory: A single individual's state-action-state sequence
- Panel: Collection of trajectories (panel data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class DDCProblem:
    """Specification of a Dynamic Discrete Choice problem.

    This dataclass contains the structural parameters that define the
    decision environment, following the notation in Rust (1987).

    Attributes:
        num_states: Number of discrete states |S|
        num_actions: Number of discrete actions |A|
        discount_factor: Time discount factor β ∈ [0, 1)
        scale_parameter: Logit scale parameter σ > 0 (extreme value shock scale)

    Example:
        >>> problem = DDCProblem(
        ...     num_states=90,
        ...     num_actions=2,
        ...     discount_factor=0.9999,
        ...     scale_parameter=1.0
        ... )
    """

    num_states: int
    num_actions: int
    discount_factor: float = 0.9999
    scale_parameter: float = 1.0

    def __post_init__(self) -> None:
        if self.num_states < 1:
            raise ValueError(f"num_states must be positive, got {self.num_states}")
        if self.num_actions < 1:
            raise ValueError(f"num_actions must be positive, got {self.num_actions}")
        if not 0 <= self.discount_factor < 1:
            raise ValueError(
                f"discount_factor must be in [0, 1), got {self.discount_factor}"
            )
        if self.scale_parameter <= 0:
            raise ValueError(
                f"scale_parameter must be positive, got {self.scale_parameter}"
            )


@dataclass
class Trajectory:
    """A single individual's observed decision trajectory.

    Represents the sequence of states, actions, and next states observed
    for one decision-maker over time. This is the fundamental unit of
    observation in dynamic discrete choice estimation.

    Attributes:
        states: Tensor of shape (T,) containing state indices at each period
        actions: Tensor of shape (T,) containing chosen action at each period
        next_states: Tensor of shape (T,) containing state after transition
        individual_id: Optional identifier for the individual
        metadata: Optional dictionary for additional trajectory-level data

    Example:
        >>> traj = Trajectory(
        ...     states=torch.tensor([0, 5, 12, 18]),
        ...     actions=torch.tensor([0, 0, 0, 1]),
        ...     next_states=torch.tensor([5, 12, 18, 0]),
        ...     individual_id="bus_001"
        ... )
    """

    states: torch.Tensor
    actions: torch.Tensor
    next_states: torch.Tensor
    individual_id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.states) != len(self.actions):
            raise ValueError(
                f"states and actions must have same length, "
                f"got {len(self.states)} and {len(self.actions)}"
            )
        if len(self.states) != len(self.next_states):
            raise ValueError(
                f"states and next_states must have same length, "
                f"got {len(self.states)} and {len(self.next_states)}"
            )

    def __len__(self) -> int:
        """Return the number of time periods in this trajectory."""
        return len(self.states)

    @property
    def num_periods(self) -> int:
        """Number of time periods observed."""
        return len(self.states)

    def to(self, device: torch.device | str) -> Trajectory:
        """Move trajectory tensors to specified device."""
        return Trajectory(
            states=self.states.to(device),
            actions=self.actions.to(device),
            next_states=self.next_states.to(device),
            individual_id=self.individual_id,
            metadata=self.metadata,
        )


@dataclass
class Panel:
    """Collection of individual trajectories forming a panel dataset.

    A Panel represents the complete dataset used for estimation, containing
    trajectories from multiple individuals observed over (potentially varying)
    time periods. This is the primary data structure passed to estimators.

    Attributes:
        trajectories: List of Trajectory objects, one per individual
        metadata: Optional dictionary for panel-level metadata

    Example:
        >>> panel = Panel(trajectories=[traj1, traj2, traj3])
        >>> print(f"Panel with {panel.num_individuals} individuals")
        >>> print(f"Total observations: {panel.num_observations}")
    """

    trajectories: list[Trajectory]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.trajectories:
            raise ValueError("Panel must contain at least one trajectory")

    def __len__(self) -> int:
        """Return the number of individuals in the panel."""
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        """Get trajectory by index."""
        return self.trajectories[idx]

    def __iter__(self):
        """Iterate over trajectories."""
        return iter(self.trajectories)

    @property
    def num_individuals(self) -> int:
        """Number of individuals in the panel."""
        return len(self.trajectories)

    @property
    def num_observations(self) -> int:
        """Total number of state-action observations across all individuals."""
        return sum(len(traj) for traj in self.trajectories)

    @property
    def num_periods_per_individual(self) -> list[int]:
        """List of number of periods for each individual."""
        return [len(traj) for traj in self.trajectories]

    def get_all_states(self) -> torch.Tensor:
        """Concatenate all states into a single tensor."""
        return torch.cat([traj.states for traj in self.trajectories])

    def get_all_actions(self) -> torch.Tensor:
        """Concatenate all actions into a single tensor."""
        return torch.cat([traj.actions for traj in self.trajectories])

    def get_all_next_states(self) -> torch.Tensor:
        """Concatenate all next_states into a single tensor."""
        return torch.cat([traj.next_states for traj in self.trajectories])

    def to(self, device: torch.device | str) -> Panel:
        """Move all trajectory tensors to specified device."""
        return Panel(
            trajectories=[traj.to(device) for traj in self.trajectories],
            metadata=self.metadata,
        )

    def compute_state_frequencies(self, num_states: int) -> torch.Tensor:
        """Compute empirical state visit frequencies.

        Args:
            num_states: Total number of possible states

        Returns:
            Tensor of shape (num_states,) with visit counts
        """
        all_states = self.get_all_states()
        frequencies = torch.zeros(num_states, dtype=torch.float32)
        for state in all_states:
            frequencies[state.item()] += 1
        return frequencies / frequencies.sum()

    def compute_choice_frequencies(
        self, num_states: int, num_actions: int
    ) -> torch.Tensor:
        """Compute empirical choice frequencies by state.

        This gives the empirical conditional choice probabilities (CCPs)
        that can be used for CCP-based estimation methods.

        Args:
            num_states: Total number of possible states
            num_actions: Total number of possible actions

        Returns:
            Tensor of shape (num_states, num_actions) with empirical CCPs
        """
        all_states = self.get_all_states()
        all_actions = self.get_all_actions()

        counts = torch.zeros((num_states, num_actions), dtype=torch.float32)
        for state, action in zip(all_states, all_actions):
            counts[state.item(), action.item()] += 1

        # Normalize to get probabilities (add small epsilon to avoid division by zero)
        row_sums = counts.sum(dim=1, keepdim=True)
        row_sums = torch.where(row_sums > 0, row_sums, torch.ones_like(row_sums))
        return counts / row_sums

    @classmethod
    def from_numpy(
        cls,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
        individual_ids: np.ndarray | None = None,
    ) -> Panel:
        """Create Panel from numpy arrays with individual grouping.

        Args:
            states: Array of shape (N,) with state indices
            actions: Array of shape (N,) with action indices
            next_states: Array of shape (N,) with next state indices
            individual_ids: Array of shape (N,) with individual identifiers.
                           If None, all observations treated as one individual.

        Returns:
            Panel object with trajectories grouped by individual
        """
        if individual_ids is None:
            individual_ids = np.zeros(len(states), dtype=np.int64)

        unique_ids = np.unique(individual_ids)
        trajectories = []

        for ind_id in unique_ids:
            mask = individual_ids == ind_id
            traj = Trajectory(
                states=torch.tensor(states[mask], dtype=torch.long),
                actions=torch.tensor(actions[mask], dtype=torch.long),
                next_states=torch.tensor(next_states[mask], dtype=torch.long),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return cls(trajectories=trajectories)
