"""FrozenLake dataset for IRL benchmarking.

This module generates trajectory data on the classic FrozenLake-v1
environment (4x4 grid, slippery surface). The agent navigates from the
start cell to the goal without falling into holes. Known ground truth
parameters enable parameter recovery evaluation.

State space:
    16 states on a 4x4 grid. Holes at 5, 7, 11, 12. Goal at 15.

Action space:
    4 actions: Left (0), Down (1), Right (2), Up (3).

Reference:
    Gymnasium FrozenLake-v1 (Farama Foundation).
"""

from typing import Union

import numpy as np
import pandas as pd

from econirl.core.types import Panel, Trajectory
from econirl.environments.frozen_lake import FrozenLakeEnvironment
from econirl.simulation.synthetic import simulate_panel


def load_frozen_lake(
    n_individuals: int = 500,
    n_periods: int = 100,
    as_panel: bool = False,
    seed: int = 42,
    is_slippery: bool = True,
    step_penalty: float = -0.04,
    goal_reward: float = 1.0,
    hole_penalty: float = -1.0,
    discount_factor: float = 0.99,
) -> Union[pd.DataFrame, Panel]:
    """Generate FrozenLake trajectory data for IRL benchmarking.

    Creates synthetic trajectories on a 4x4 frozen lake where an agent
    navigates to the goal while avoiding holes. The surface is slippery
    by default, adding stochastic transitions. Known ground truth
    parameters enable parameter recovery evaluation.

    Args:
        n_individuals: Number of trajectories to simulate.
        n_periods: Number of steps per trajectory.
        as_panel: If True, return a Panel object for econirl estimators.
        seed: Random seed for reproducibility.
        is_slippery: If True, actions are stochastic (1/3 each direction).
        step_penalty: Per-step cost at non-terminal states.
        goal_reward: Reward for reaching the goal.
        hole_penalty: Penalty for falling into a hole.
        discount_factor: Time discount factor.

    Returns:
        DataFrame with columns: agent_id, period, state, action,
        next_state, row, col, next_row, next_col, at_hole, at_goal.

        If as_panel=True, returns Panel object.
    """
    env = FrozenLakeEnvironment(
        is_slippery=is_slippery,
        step_penalty=step_penalty,
        goal_reward=goal_reward,
        hole_penalty=hole_penalty,
        discount_factor=discount_factor,
        seed=seed,
    )

    panel = simulate_panel(
        env,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )

    if as_panel:
        return panel

    ncol = 4
    records = []
    for traj in panel.trajectories:
        tid = traj.individual_id
        states = traj.states
        actions = traj.actions
        next_states = traj.next_states

        for t in range(len(states)):
            s = int(states[t])
            ns = int(next_states[t])
            records.append({
                "agent_id": tid,
                "period": t,
                "state": s,
                "action": int(actions[t]),
                "next_state": ns,
                "row": s // ncol,
                "col": s % ncol,
                "next_row": ns // ncol,
                "next_col": ns % ncol,
                "at_hole": ns in env.holes,
                "at_goal": ns == env.goal,
            })

    return pd.DataFrame(records)


def get_frozen_lake_info() -> dict:
    """Return metadata about the FrozenLake dataset."""
    return {
        "name": "FrozenLake (Gymnasium-style)",
        "description": "Classic 4x4 frozen lake with slippery transitions",
        "source": "Hardcoded from Gymnasium FrozenLake-v1 (no dependency)",
        "n_states": 16,
        "n_actions": 4,
        "n_features": 3,
        "state_description": "4x4 grid: S=start, F=frozen, H=hole, G=goal",
        "action_description": "Left/Down/Right/Up",
        "true_parameters": {
            "step_penalty": -0.04,
            "goal_reward": 1.0,
            "hole_penalty": -1.0,
        },
        "ground_truth": True,
        "use_case": "Minimal stochastic IRL benchmark with parameter recovery",
    }
