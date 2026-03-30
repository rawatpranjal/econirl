"""Trivago hotel search DDC dataset (RecSys Challenge 2019).

Models hotel search sessions as a sequential DDC: at each step, the user
decides to browse (view hotel details), refine (change filters/search),
clickout (book a hotel), or abandon (leave without booking).

Reference:
    RecSys Challenge 2019. https://recsys.trivago.cloud/
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

DEFAULT_DATA_PATH = "/Volumes/Expansion/datasets/trivago-2019/train.csv"

# Action mapping from raw action_type strings to simplified action indices
ACTION_BROWSE = 0
ACTION_REFINE = 1
ACTION_CLICKOUT = 2
ACTION_ABANDON = 3
N_ACTIONS = 4
ACTION_NAMES = ["browse", "refine", "clickout", "abandon"]

# Default state space configuration
N_STEP_BUCKETS = 3
N_VIEWED_BUCKETS = 4
N_DEVICES = 3
N_STATES = N_STEP_BUCKETS * N_VIEWED_BUCKETS * N_DEVICES + 1  # 37 (36 + absorbing)
ABSORBING_STATE = N_STATES - 1  # 36

DEVICE_MAP = {"mobile": 0, "desktop": 1, "tablet": 2}

# Mapping from raw action_type to simplified action
_BROWSE_ACTIONS = {
    "interaction item image",
    "interaction item info",
    "interaction item rating",
    "interaction item deals",
}
_REFINE_ACTIONS = {
    "filter selection",
    "change of sort order",
    "search for destination",
    "search for poi",
    "search for item",
}
_CLICKOUT_ACTION = "clickout item"


def _map_action_type(action_type: str) -> int:
    """Map a raw Trivago action_type string to a simplified action index.

    Parameters
    ----------
    action_type : str
        One of the 10 raw action types from the Trivago dataset.

    Returns
    -------
    int
        0 = browse, 1 = refine, 2 = clickout.
        Abandon (3) is not returned here; it is appended at session level.
    """
    if action_type in _BROWSE_ACTIONS:
        return ACTION_BROWSE
    elif action_type in _REFINE_ACTIONS:
        return ACTION_REFINE
    elif action_type == _CLICKOUT_ACTION:
        return ACTION_CLICKOUT
    else:
        # Unknown action types default to refine (search-related)
        return ACTION_REFINE


def _step_bucket(step: int) -> int:
    """Map a step number to a bucket index."""
    if step <= 3:
        return 0
    elif step <= 8:
        return 1
    else:
        return 2


def _viewed_bucket(n_viewed: int) -> int:
    """Map cumulative items viewed count to a bucket index."""
    if n_viewed == 0:
        return 0
    elif n_viewed <= 2:
        return 1
    elif n_viewed <= 5:
        return 2
    else:
        return 3


def _compute_state(
    step_b: int,
    viewed_b: int,
    device_code: int,
    n_viewed_buckets: int = N_VIEWED_BUCKETS,
) -> int:
    """Compute state index from bucketed features."""
    return step_b * (n_viewed_buckets * N_DEVICES) + viewed_b * N_DEVICES + device_code


def load_trivago_sessions(
    data_path: Optional[str] = None,
    n_sessions: Optional[int] = None,
) -> "pl.DataFrame":
    """Load raw Trivago session data using polars for speed.

    Parameters
    ----------
    data_path : str, optional
        Path to the train.csv file. Defaults to the external drive location.
    n_sessions : int, optional
        If specified, load only the first N unique session_ids.

    Returns
    -------
    pl.DataFrame
        DataFrame with all original columns.
    """
    import polars as pl

    path = data_path or DEFAULT_DATA_PATH
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Trivago data not found at {path}. "
            "Download from: https://recsys.trivago.cloud/"
        )

    df = pl.read_csv(path, infer_schema_length=10000)

    if n_sessions is not None:
        # Get the first n unique session_ids (preserving file order)
        unique_sessions = df.select("session_id").unique(maintain_order=True).head(n_sessions)
        df = df.join(unique_sessions, on="session_id", how="inner")

    return df


def build_trivago_mdp(
    sessions_df: "pl.DataFrame",
    n_step_buckets: int = N_STEP_BUCKETS,
    n_viewed_buckets: int = N_VIEWED_BUCKETS,
) -> dict:
    """Build MDP tuples (state, action, next_state) from raw session data.

    For each session, tracks cumulative items viewed, computes bucketed
    state features, maps actions, and handles terminal transitions.

    Parameters
    ----------
    sessions_df : pl.DataFrame
        Raw session data from ``load_trivago_sessions``.
    n_step_buckets : int
        Number of step depth buckets (default 3).
    n_viewed_buckets : int
        Number of items-viewed buckets (default 4).

    Returns
    -------
    dict
        Keys: all_states, all_actions, all_next_states, session_ids,
        n_states, n_actions, state_names, action_names.
    """
    import polars as pl

    n_non_absorbing = n_step_buckets * n_viewed_buckets * N_DEVICES
    absorbing = n_non_absorbing  # terminal state index

    # Build state names
    step_labels = ["early", "mid", "late"][:n_step_buckets]
    viewed_labels = ["none", "few", "some", "many"][:n_viewed_buckets]
    device_labels = ["mobile", "desktop", "tablet"]
    state_names = []
    for sb in range(n_step_buckets):
        for vb in range(n_viewed_buckets):
            for d in range(N_DEVICES):
                state_names.append(
                    f"step={step_labels[sb]}_viewed={viewed_labels[vb]}_dev={device_labels[d]}"
                )
    state_names.append("absorbing")

    # Sort by session_id and step to ensure correct ordering
    sessions_df = sessions_df.sort(["session_id", "step"])

    # Convert to Python for session-level processing
    session_ids_col = sessions_df["session_id"].to_list()
    steps_col = sessions_df["step"].to_list()
    action_types_col = sessions_df["action_type"].to_list()
    devices_col = sessions_df["device"].to_list()

    all_states = []
    all_actions = []
    all_next_states = []
    all_session_ids = []

    # Group by session
    prev_session = None
    session_start = 0

    # Build index of session boundaries
    boundaries = []
    for i in range(len(session_ids_col)):
        if session_ids_col[i] != prev_session:
            if prev_session is not None:
                boundaries.append((session_start, i))
            session_start = i
            prev_session = session_ids_col[i]
    if prev_session is not None:
        boundaries.append((session_start, len(session_ids_col)))

    for start, end in boundaries:
        sess_id = session_ids_col[start]

        # Extract session data
        sess_steps = steps_col[start:end]
        sess_action_types = action_types_col[start:end]
        sess_device_raw = devices_col[start]

        # Device code
        device_code = DEVICE_MAP.get(sess_device_raw, 0)

        # Track cumulative items viewed (browse actions increment the counter)
        n_items_viewed = 0
        sess_states = []
        sess_actions = []

        for j in range(end - start):
            step = sess_steps[j]
            action_type = sess_action_types[j]

            # Compute current state
            sb = _step_bucket(step)
            vb = _viewed_bucket(n_items_viewed)
            state = _compute_state(sb, vb, device_code, n_viewed_buckets)
            action = _map_action_type(action_type)

            sess_states.append(state)
            sess_actions.append(action)

            # Update items viewed if this was a browse action
            if action_type in _BROWSE_ACTIONS:
                n_items_viewed += 1

        # Check if session ends with clickout
        ends_with_clickout = sess_actions[-1] == ACTION_CLICKOUT

        # Build transitions
        for j in range(len(sess_states)):
            s = sess_states[j]
            a = sess_actions[j]

            if a == ACTION_CLICKOUT:
                # Terminal: clickout transitions to absorbing state
                ns = absorbing
            elif j == len(sess_states) - 1:
                # Last step and not clickout => this is the last observed action
                # We will add an abandon action below
                # This non-terminal action transitions to the state where abandon happens
                # But since there's no next observation, compute next state from
                # updated counters
                next_step = sess_steps[j] + 1
                next_sb = _step_bucket(next_step)
                # n_items_viewed was already updated if this was a browse
                next_vb = _viewed_bucket(n_items_viewed)
                ns = _compute_state(next_sb, next_vb, device_code, n_viewed_buckets)
            else:
                # Next state is the state at the next step
                ns = sess_states[j + 1]

            all_states.append(s)
            all_actions.append(a)
            all_next_states.append(ns)
            all_session_ids.append(sess_id)

        # If session doesn't end with clickout, append abandon action
        if not ends_with_clickout:
            # Abandon happens at the state after the last observed action
            last_step = sess_steps[-1] + 1
            last_sb = _step_bucket(last_step)
            last_vb = _viewed_bucket(n_items_viewed)
            abandon_state = _compute_state(last_sb, last_vb, device_code, n_viewed_buckets)

            all_states.append(abandon_state)
            all_actions.append(ACTION_ABANDON)
            all_next_states.append(absorbing)
            all_session_ids.append(sess_id)

    return {
        "all_states": all_states,
        "all_actions": all_actions,
        "all_next_states": all_next_states,
        "session_ids": all_session_ids,
        "n_states": n_non_absorbing + 1,
        "n_actions": N_ACTIONS,
        "state_names": state_names,
        "action_names": ACTION_NAMES,
    }


def build_trivago_panel(mdp_dict: dict) -> "Panel":
    """Build a Panel of trajectories from the MDP dict.

    Groups observations by session_id and creates one Trajectory per session.

    Parameters
    ----------
    mdp_dict : dict
        Output of ``build_trivago_mdp``.

    Returns
    -------
    Panel
        Panel object with one trajectory per session.
    """
    from econirl.core.types import Panel, Trajectory

    states = mdp_dict["all_states"]
    actions = mdp_dict["all_actions"]
    next_states = mdp_dict["all_next_states"]
    session_ids = mdp_dict["session_ids"]

    # Group by session_id preserving order
    session_groups: dict[str, list[int]] = {}
    for i, sid in enumerate(session_ids):
        if sid not in session_groups:
            session_groups[sid] = []
        session_groups[sid].append(i)

    trajectories = []
    for sid, indices in session_groups.items():
        traj = Trajectory(
            states=torch.tensor([states[i] for i in indices], dtype=torch.long),
            actions=torch.tensor([actions[i] for i in indices], dtype=torch.long),
            next_states=torch.tensor([next_states[i] for i in indices], dtype=torch.long),
            individual_id=sid,
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def build_trivago_features(
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
) -> torch.Tensor:
    """Build action-dependent feature matrix for Trivago hotel search.

    4 features per (state, action) pair:
    - step_cost: grows with search depth (negative for browse/refine)
    - browse_indicator: -1 for browse, 0 otherwise
    - refine_indicator: -1 for refine, 0 otherwise
    - clickout_indicator: +1 for clickout, 0 otherwise

    Parameters
    ----------
    n_states : int
        Number of states including absorbing (default 37).
    n_actions : int
        Number of actions (default 4).

    Returns
    -------
    torch.Tensor
        Feature matrix of shape (n_states, n_actions, 4).
    """
    n_features = 4
    features = torch.zeros(n_states, n_actions, n_features)

    n_non_absorbing = n_states - 1

    for s in range(n_non_absorbing):
        # Extract step_bucket from state index
        sb = s // (N_VIEWED_BUCKETS * N_DEVICES)

        # browse (a=0): step cost + browse indicator
        features[s, ACTION_BROWSE, 0] = -sb / 2.0  # step_cost
        features[s, ACTION_BROWSE, 1] = -1.0        # browse_indicator

        # refine (a=1): step cost + refine indicator
        features[s, ACTION_REFINE, 0] = -sb / 2.0   # step_cost
        features[s, ACTION_REFINE, 2] = -1.0         # refine_indicator

        # clickout (a=2): clickout indicator (booking value)
        features[s, ACTION_CLICKOUT, 3] = 1.0        # clickout_indicator

        # abandon (a=3): all zeros (normalized baseline)

    # Absorbing state: all zeros for all actions (already initialized)

    return features


def build_trivago_transitions(
    mdp_dict: dict,
    n_states: int = N_STATES,
    n_actions: int = N_ACTIONS,
    smoothing: float = 1e-8,
) -> torch.Tensor:
    """Build empirical transition matrix P(s'|s,a) from training data.

    Parameters
    ----------
    mdp_dict : dict
        Output of ``build_trivago_mdp``.
    n_states : int
        Number of states including absorbing (default 37).
    n_actions : int
        Number of actions (default 4).
    smoothing : float
        Smoothing constant for unobserved (s, a) pairs.

    Returns
    -------
    torch.Tensor
        Transition matrix of shape (n_actions, n_states, n_states).
    """
    absorbing = n_states - 1

    # Count transitions
    counts = torch.zeros(n_actions, n_states, n_states)
    for s, a, ns in zip(
        mdp_dict["all_states"],
        mdp_dict["all_actions"],
        mdp_dict["all_next_states"],
    ):
        counts[a, s, ns] += 1.0

    # Force terminal actions to absorbing state
    for a in [ACTION_CLICKOUT, ACTION_ABANDON]:
        counts[a, :, :] = 0.0
        counts[a, :, absorbing] = 1.0

    # Absorbing state self-loops for all actions
    for a in range(n_actions):
        counts[a, absorbing, :] = 0.0
        counts[a, absorbing, absorbing] = 1.0

    # Normalize rows; add smoothing for unobserved (s, a) pairs
    transitions = torch.zeros_like(counts)
    for a in range(n_actions):
        for s in range(n_states):
            row_sum = counts[a, s].sum()
            if row_sum > 0:
                transitions[a, s] = counts[a, s] / row_sum
            else:
                # Unobserved (s, a): uniform distribution with smoothing
                transitions[a, s] = torch.ones(n_states) / n_states

    return transitions


def load_trivago_search(
    n_sessions: Optional[int] = None,
    data_path: Optional[str] = None,
) -> "Panel":
    """Convenience function: load Trivago data and return a Panel.

    Parameters
    ----------
    n_sessions : int, optional
        Number of sessions to load. None = all.
    data_path : str, optional
        Path to train.csv.

    Returns
    -------
    Panel
        Panel of session trajectories.
    """
    sessions_df = load_trivago_sessions(data_path=data_path, n_sessions=n_sessions)
    mdp_dict = build_trivago_mdp(sessions_df)
    return build_trivago_panel(mdp_dict)


def get_trivago_info() -> dict:
    """Return metadata about the Trivago hotel search dataset."""
    return {
        "name": "Trivago Hotel Search Sessions (RecSys 2019)",
        "description": (
            "Sequential hotel search sessions modeled as DDC: "
            "browse, refine, clickout, or abandon."
        ),
        "source": "RecSys Challenge 2019",
        "url": "https://recsys.trivago.cloud/",
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "n_sessions": "~910K",
        "n_observations": "~16M raw rows",
        "state_description": "(step_bucket, n_items_viewed_bucket, device)",
        "action_description": "browse / refine / clickout / abandon",
        "action_names": ACTION_NAMES,
    }
