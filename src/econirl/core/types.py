"""Core data types for dynamic discrete choice models.

This module defines the fundamental data structures used throughout econirl:
- DDCProblem: Specification of a discrete choice problem
- Trajectory: A single individual's state-action-state sequence
- Panel: Collection of trajectories (panel data)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class DDCProblem:
    """Specification of a Dynamic Discrete Choice problem.

    This dataclass contains the structural parameters that define the
    decision environment, following the notation in Rust (1987).

    Attributes:
        num_states: Number of discrete states |S|
        num_actions: Number of discrete actions |A|
        discount_factor: Time discount factor beta in [0, 1)
        scale_parameter: Logit scale parameter sigma > 0 (extreme value shock scale)

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
    num_periods: int | None = None  # None = infinite horizon, int = finite horizon
    state_dim: int | None = None
    state_encoder: Callable | None = field(
        default=None, hash=False, compare=False
    )

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
        states: Array of shape (T,) containing state indices at each period
        actions: Array of shape (T,) containing chosen action at each period
        next_states: Array of shape (T,) containing state after transition
        individual_id: Optional identifier for the individual
        metadata: Optional dictionary for additional trajectory-level data

    Example:
        >>> traj = Trajectory(
        ...     states=jnp.array([0, 5, 12, 18]),
        ...     actions=jnp.array([0, 0, 0, 1]),
        ...     next_states=jnp.array([5, 12, 18, 0]),
        ...     individual_id="bus_001"
        ... )
    """

    states: jnp.ndarray
    actions: jnp.ndarray
    next_states: jnp.ndarray
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

    def get_all_states(self) -> jnp.ndarray:
        """Concatenate all states into a single array."""
        return jnp.concatenate([traj.states for traj in self.trajectories])

    def get_all_actions(self) -> jnp.ndarray:
        """Concatenate all actions into a single array."""
        return jnp.concatenate([traj.actions for traj in self.trajectories])

    def get_all_next_states(self) -> jnp.ndarray:
        """Concatenate all next_states into a single array."""
        return jnp.concatenate([traj.next_states for traj in self.trajectories])

    def compute_state_frequencies(self, num_states: int) -> jnp.ndarray:
        """Compute empirical state visit frequencies.

        Args:
            num_states: Total number of possible states

        Returns:
            Array of shape (num_states,) with visit frequencies
        """
        all_states = self.get_all_states()
        frequencies = jnp.zeros(num_states, dtype=jnp.float32)
        frequencies = frequencies.at[all_states].add(1.0)
        return frequencies / frequencies.sum()

    def compute_choice_frequencies(
        self, num_states: int, num_actions: int
    ) -> jnp.ndarray:
        """Compute empirical choice frequencies by state.

        This gives the empirical conditional choice probabilities (CCPs)
        that can be used for CCP-based estimation methods.

        Args:
            num_states: Total number of possible states
            num_actions: Total number of possible actions

        Returns:
            Array of shape (num_states, num_actions) with empirical CCPs
        """
        all_states = self.get_all_states()
        all_actions = self.get_all_actions()

        counts = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
        counts = counts.at[all_states, all_actions].add(1.0)

        # Normalize to get probabilities
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums_safe = jnp.where(row_sums > 0, row_sums, jnp.ones_like(row_sums))
        return counts / row_sums_safe

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
                states=jnp.array(states[mask], dtype=jnp.int32),
                actions=jnp.array(actions[mask], dtype=jnp.int32),
                next_states=jnp.array(next_states[mask], dtype=jnp.int32),
                individual_id=ind_id,
            )
            trajectories.append(traj)

        return cls(trajectories=trajectories)


class TrajectoryPanel(Panel):
    """Enhanced panel with efficient tensor operations and DataFrame I/O.

    TrajectoryPanel keeps the list-of-Trajectory interface from Panel but
    adds efficient stacked tensor operations, DataFrame conversion, bootstrap
    resampling, transition iteration, and sufficient statistics computation.

    All existing Panel methods and properties are inherited unchanged.
    """

    # ------------------------------------------------------------------
    # Lazy-cached stacked tensors
    # ------------------------------------------------------------------

    @functools.cached_property
    def all_states(self) -> jnp.ndarray:
        """Concatenated states array of shape (N,)."""
        return self.get_all_states()

    @functools.cached_property
    def all_actions(self) -> jnp.ndarray:
        """Concatenated actions array of shape (N,)."""
        return self.get_all_actions()

    @functools.cached_property
    def all_next_states(self) -> jnp.ndarray:
        """Concatenated next_states array of shape (N,)."""
        return self.get_all_next_states()

    @functools.cached_property
    def offsets(self) -> jnp.ndarray:
        """Cumulative individual lengths of shape (I+1,).

        ``offsets[i]`` is the start index of individual ``i`` in the
        concatenated tensors; ``offsets[-1] == num_observations``.
        """
        lengths = [len(traj) for traj in self.trajectories]
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        return jnp.array(offsets, dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Classmethods
    # ------------------------------------------------------------------

    @classmethod
    def from_dataframe(
        cls,
        df,  # pd.DataFrame — imported lazily
        state: str,
        action: str,
        id: str,
        next_state: str | None = None,
    ) -> TrajectoryPanel:
        """Create a TrajectoryPanel from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Panel data with at least ``state``, ``action``, and ``id`` columns.
        state : str
            Column name for state indices.
        action : str
            Column name for action indices.
        id : str
            Column name for individual identifiers.
        next_state : str or None
            Column name for next-state indices.  If ``None``, next states are
            inferred from sequential rows within each individual.

        Returns
        -------
        TrajectoryPanel
        """
        import pandas as pd  # noqa: F811 — lazy import

        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected pd.DataFrame, got {type(df).__name__}")

        trajectories: list[Trajectory] = []
        max_state = int(df[state].max())

        for ind_id, group in df.groupby(id, sort=True):
            group = group.sort_index()
            states_arr = jnp.array(group[state].values, dtype=jnp.int32)
            actions_arr = jnp.array(group[action].values, dtype=jnp.int32)

            if next_state is not None:
                next_states_arr = jnp.array(
                    group[next_state].values, dtype=jnp.int32
                )
            else:
                # Infer next_states from sequential rows
                n = len(states_arr)
                # Build as numpy first since JAX arrays are immutable
                ns = np.empty(n, dtype=np.int32)
                if n > 1:
                    ns[:-1] = np.asarray(states_arr[1:])
                # Last row: heuristic
                last_s = int(states_arr[-1])
                last_a = int(actions_arr[-1])
                if last_a == 0:
                    ns[-1] = min(last_s + 1, max_state)
                else:
                    ns[-1] = 0
                next_states_arr = jnp.array(ns)

            trajectories.append(
                Trajectory(
                    states=states_arr,
                    actions=actions_arr,
                    next_states=next_states_arr,
                    individual_id=ind_id,
                )
            )

        return cls(trajectories=trajectories)

    @classmethod
    def from_panel(cls, panel: Panel) -> TrajectoryPanel:
        """Wrap an existing Panel as a TrajectoryPanel.

        Parameters
        ----------
        panel : Panel
            Existing panel to wrap.

        Returns
        -------
        TrajectoryPanel
        """
        return cls(trajectories=panel.trajectories, metadata=panel.metadata)

    # ------------------------------------------------------------------
    # Sufficient statistics
    # ------------------------------------------------------------------

    def sufficient_stats(self, n_states: int, n_actions: int):
        """Compute sufficient statistics for tabular estimators.

        Parameters
        ----------
        n_states : int
            Total number of states in the MDP.
        n_actions : int
            Total number of actions in the MDP.

        Returns
        -------
        SufficientStats
            Pre-computed state-action counts, transitions, empirical CCPs,
            and initial state distribution.
        """
        from econirl.core.sufficient_stats import SufficientStats

        states = self.all_states
        actions = self.all_actions
        next_states = self.all_next_states

        # --- state-action counts ---
        state_action_counts = jnp.zeros(
            (n_states, n_actions), dtype=jnp.float64
        )
        state_action_counts = state_action_counts.at[states, actions].add(1.0)

        # --- empirical CCPs ---
        row_sums = state_action_counts.sum(axis=1, keepdims=True)
        row_sums_safe = jnp.where(
            row_sums > 0, row_sums, jnp.ones_like(row_sums)
        )
        empirical_ccps = state_action_counts / row_sums_safe
        # States with zero observations: uniform over actions
        zero_mask = (row_sums.squeeze(1) == 0)
        empirical_ccps = jnp.where(
            zero_mask[:, None],
            jnp.ones((n_states, n_actions)) / n_actions,
            empirical_ccps,
        )

        # --- transition matrix (A, S, S) ---
        # Build as numpy for the counting loop, then convert
        transition_counts_np = np.zeros(
            (n_actions, n_states, n_states), dtype=np.float64
        )
        states_np = np.asarray(states)
        actions_np = np.asarray(actions)
        next_states_np = np.asarray(next_states)
        for s, a, sp in zip(states_np, actions_np, next_states_np):
            transition_counts_np[a, s, sp] += 1
        transition_counts = jnp.array(transition_counts_np)

        # Normalize rows
        eps = 1e-10
        transition_row_sums = transition_counts.sum(axis=2, keepdims=True)
        transition_row_sums_safe = jnp.where(
            transition_row_sums > 0,
            transition_row_sums,
            jnp.ones_like(transition_row_sums),
        )
        transitions = transition_counts / transition_row_sums_safe
        # Zero-count rows get uniform
        zero_transition_mask = (transition_row_sums.squeeze(2) == 0)
        transitions = jnp.where(
            zero_transition_mask[:, :, None],
            jnp.ones((n_actions, n_states, n_states)) / n_states,
            transitions,
        )

        # Add epsilon smoothing, then re-normalize
        transitions = transitions + eps
        transitions = transitions / transitions.sum(axis=2, keepdims=True)

        # --- initial distribution ---
        initial_dist = jnp.zeros(n_states, dtype=jnp.float64)
        initial_states = jnp.array(
            [traj.states[0] for traj in self.trajectories], dtype=jnp.int32
        )
        initial_dist = initial_dist.at[initial_states].add(1.0)
        initial_total = initial_dist.sum()
        initial_dist = jnp.where(
            initial_total > 0,
            initial_dist / initial_total,
            jnp.ones(n_states) / n_states,
        )

        return SufficientStats(
            state_action_counts=state_action_counts.astype(jnp.float32),
            transitions=transitions.astype(jnp.float32),
            empirical_ccps=empirical_ccps.astype(jnp.float32),
            initial_distribution=initial_dist.astype(jnp.float32),
            n_observations=int(states.shape[0]),
            n_individuals=len(self.trajectories),
        )

    # ------------------------------------------------------------------
    # Bootstrap resampling
    # ------------------------------------------------------------------

    def resample_individuals(
        self, n: int | None = None, seed: int | None = None
    ) -> TrajectoryPanel:
        """Bootstrap resample of individuals (trajectories).

        Parameters
        ----------
        n : int or None
            Number of individuals in the resampled panel.  Defaults to the
            same number as the original panel.
        seed : int or None
            Random seed for reproducibility.

        Returns
        -------
        TrajectoryPanel
            New panel with resampled trajectories (sampled with replacement).
        """
        if n is None:
            n = len(self.trajectories)

        rng = np.random.RandomState(seed)
        indices = rng.choice(len(self.trajectories), size=n, replace=True)
        resampled = [self.trajectories[i] for i in indices]
        return TrajectoryPanel(trajectories=resampled, metadata=self.metadata)

    # ------------------------------------------------------------------
    # Mini-batch iteration
    # ------------------------------------------------------------------

    def iter_transitions(
        self, batch_size: int = 512, seed: int = 0
    ) -> Iterator[tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Iterate over (state, action, next_state) mini-batches.

        Shuffles all transitions and yields them in batches for SGD-style
        training loops.

        Parameters
        ----------
        batch_size : int
            Number of transitions per batch.
        seed : int
            Random seed for shuffling.

        Yields
        ------
        tuple[Array, Array, Array]
            ``(states, actions, next_states)`` each of shape ``(B,)`` where
            ``B <= batch_size``.
        """
        states = self.all_states
        actions = self.all_actions
        next_states = self.all_next_states
        n = states.shape[0]

        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            yield states[idx], actions[idx], next_states[idx]

    # ------------------------------------------------------------------
    # DataFrame conversion
    # ------------------------------------------------------------------

    def to_dataframe(self):
        """Convert panel to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``id``, ``period``, ``state``, ``action``,
            ``next_state``.
        """
        import pandas as pd  # noqa: F811 — lazy import

        rows = []
        for traj in self.trajectories:
            ind_id = traj.individual_id
            for t in range(len(traj)):
                rows.append(
                    {
                        "id": ind_id,
                        "period": t,
                        "state": int(traj.states[t]),
                        "action": int(traj.actions[t]),
                        "next_state": int(traj.next_states[t]),
                    }
                )
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_npz(self, path: str) -> None:
        """Save panel to a compressed .npz file.

        Stores all trajectories as flat arrays with an offset index so
        individual trajectories can be reconstructed on load. Metadata
        and individual IDs are preserved via pickle-compatible arrays.

        Parameters
        ----------
        path : str
            File path (should end in .npz).
        """
        states = np.concatenate([np.asarray(t.states) for t in self.trajectories])
        actions = np.concatenate([np.asarray(t.actions) for t in self.trajectories])
        next_states = np.concatenate([np.asarray(t.next_states) for t in self.trajectories])

        lengths = np.array([len(t) for t in self.trajectories], dtype=np.int32)
        ids = np.array(
            [t.individual_id if t.individual_id is not None else i
             for i, t in enumerate(self.trajectories)],
            dtype=object,
        )

        np.savez_compressed(
            path,
            states=states,
            actions=actions,
            next_states=next_states,
            lengths=lengths,
            ids=ids,
        )

    @classmethod
    def load_npz(cls, path: str) -> TrajectoryPanel:
        """Load a panel from a .npz file created by save_npz.

        Parameters
        ----------
        path : str
            Path to .npz file.

        Returns
        -------
        TrajectoryPanel
        """
        data = np.load(path, allow_pickle=True)
        states = data["states"]
        actions = data["actions"]
        next_states = data["next_states"]
        lengths = data["lengths"]
        ids = data["ids"]

        trajectories = []
        offset = 0
        for i, length in enumerate(lengths):
            end = offset + length
            traj = Trajectory(
                states=jnp.array(states[offset:end], dtype=jnp.int32),
                actions=jnp.array(actions[offset:end], dtype=jnp.int32),
                next_states=jnp.array(next_states[offset:end], dtype=jnp.int32),
                individual_id=ids[i],
            )
            trajectories.append(traj)
            offset = end

        return cls(trajectories=trajectories)


# Backward-compatible alias: Panel now points to TrajectoryPanel so new code
# that creates Panel(...) automatically gets the enhanced interface.
Panel = TrajectoryPanel  # type: ignore[misc]
