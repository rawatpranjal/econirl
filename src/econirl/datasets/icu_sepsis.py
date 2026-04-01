"""ICU-Sepsis benchmark MDP dataset for clinical treatment IRL.

This module provides access to the ICU-Sepsis benchmark MDP derived from
MIMIC-III patient records by Komorowski et al. (2018). The MDP abstracts
ICU sepsis treatment into 716 discrete states (patient physiology clusters),
25 discrete actions (5 IV fluid levels times 5 vasopressor dose levels),
and transition probabilities estimated from real clinical data.

The dataset supports two use cases. First, load the raw MDP components
(transition matrices, rewards, expert policy) for direct policy evaluation
or custom analysis. Second, generate trajectory data by rolling out the
expert clinician policy through the MDP, producing Panel objects suitable
for IRL estimation with any econirl estimator.

The expert policy represents the aggregate treatment behavior of ICU
clinicians observed in MIMIC-III. Inverse reinforcement learning on this
data recovers the implicit reward function driving clinical decisions.

Reference:
    Komorowski, M., Celi, L.A., Badawi, O., Gordon, A.C., & Faisal, A.A.
    (2018). "The Artificial Intelligence Clinician learns optimal treatment
    strategies for sepsis in intensive care." Nature Medicine, 24, 1716-1720.

    Killian, T.W. et al. (2024). "ICU-Sepsis: A Benchmark MDP Built from
    Real Medical Data." NeurIPS Datasets and Benchmarks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from econirl.core.types import Panel, Trajectory

_DATA_PATH = Path(__file__).parent / "icu_sepsis_mdp.npz"

# Special state indices
DEATH_STATE = 713
SURVIVAL_STATE = 714
ABSORBING_STATE = 715


def load_icu_sepsis_mdp(data_path: str | Path | None = None) -> dict:
    """Load the raw ICU-Sepsis MDP components as numpy arrays.

    Returns a dictionary with the following keys:
    - transitions: shape (25, 716, 716), transitions[a, s, s'] = P(s'|s,a)
    - rewards: shape (716,), state rewards (+1 at survival state 714)
    - initial_distribution: shape (716,), starting state probabilities
    - expert_policy: shape (716, 25), clinician behavior policy pi(a|s)
    - sofa_scores: shape (716,), mean SOFA score per state cluster

    Args:
        data_path: Path to NPZ file. If None, uses bundled data.

    Returns:
        Dictionary of numpy arrays.
    """
    path = Path(data_path) if data_path is not None else _DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"ICU-Sepsis MDP data not found at {path}. "
            "Ensure the bundled icu_sepsis_mdp.npz file is present."
        )
    data = np.load(path)
    return {
        "transitions": np.array(data["transitions"]),
        "rewards": np.array(data["rewards"]),
        "initial_distribution": np.array(data["initial_distribution"]),
        "expert_policy": np.array(data["expert_policy"]),
        "sofa_scores": np.array(data["sofa_scores"]),
    }


def load_icu_sepsis(
    n_individuals: int = 500,
    max_steps: int = 20,
    as_panel: bool = False,
    seed: int = 42,
    data_path: str | Path | None = None,
) -> Union[pd.DataFrame, Panel]:
    """Generate ICU sepsis treatment trajectories from the expert policy.

    Simulates patient trajectories by rolling out the clinician behavior
    policy (from MIMIC-III) through the MDP transition dynamics. Each
    trajectory starts from the empirical initial state distribution and
    runs until the patient reaches the absorbing terminal state or the
    maximum number of steps is reached.

    Args:
        n_individuals: Number of patient trajectories to generate.
        max_steps: Maximum steps per trajectory. ICU stays in MIMIC-III
            are typically 5 to 20 four-hour windows.
        as_panel: If True, return a Panel object for econirl estimators.
        seed: Random seed for reproducibility.
        data_path: Path to NPZ file. If None, uses bundled data.

    Returns:
        DataFrame with columns: patient_id, period, state, action,
        next_state, sofa_score, fluid_level, vaso_level, reward,
        terminated.

        If as_panel=True, returns a Panel object.
    """
    mdp = load_icu_sepsis_mdp(data_path)
    transitions = mdp["transitions"]  # (25, 716, 716)
    expert_policy = mdp["expert_policy"]  # (716, 25)
    initial_dist = mdp["initial_distribution"]  # (716,)
    sofa = mdp["sofa_scores"]  # (716,)
    rewards = mdp["rewards"]  # (716,)

    rng = np.random.default_rng(seed)

    records = []
    trajectories = []

    for i in range(n_individuals):
        state = int(rng.choice(716, p=initial_dist))
        states_list = []
        actions_list = []
        next_states_list = []

        for t in range(max_steps):
            if state == ABSORBING_STATE:
                break

            policy = expert_policy[state]
            policy_sum = policy.sum()
            if policy_sum > 0:
                action = int(rng.choice(25, p=policy / policy_sum))
            else:
                action = int(rng.integers(25))

            probs = transitions[action, state, :]
            next_state = int(rng.choice(716, p=probs))
            terminated = next_state == ABSORBING_STATE

            fluid_level = action // 5
            vaso_level = action % 5

            records.append({
                "patient_id": i,
                "period": t,
                "state": state,
                "action": action,
                "next_state": next_state,
                "sofa_score": float(sofa[state]),
                "fluid_level": fluid_level,
                "vaso_level": vaso_level,
                "reward": float(rewards[state]),
                "terminated": terminated,
            })

            states_list.append(state)
            actions_list.append(action)
            next_states_list.append(next_state)

            state = next_state

        if states_list:
            trajectories.append(Trajectory(
                individual_id=i,
                states=np.array(states_list, dtype=np.int32),
                actions=np.array(actions_list, dtype=np.int32),
                next_states=np.array(next_states_list, dtype=np.int32),
            ))

    if as_panel:
        return Panel(trajectories=trajectories)

    return pd.DataFrame(records)


def get_icu_sepsis_info() -> dict:
    """Return metadata about the ICU-Sepsis dataset."""
    return {
        "name": "ICU-Sepsis Benchmark MDP",
        "description": (
            "Sepsis treatment MDP derived from MIMIC-III ICU records. "
            "716 states (patient physiology clusters), 25 actions "
            "(5 IV fluid x 5 vasopressor dose levels)."
        ),
        "source": "Komorowski et al. (2018) Nature Medicine; Killian et al. (2024) NeurIPS",
        "license": "MIT",
        "n_states": 716,
        "n_actions": 25,
        "n_features": 4,
        "state_description": "Patient physiology clusters from k-means on MIMIC-III vitals",
        "action_description": "5 IV fluid levels x 5 vasopressor dose levels",
        "special_states": {
            "death": DEATH_STATE,
            "survival": SURVIVAL_STATE,
            "absorbing": ABSORBING_STATE,
        },
        "ground_truth": False,
        "use_case": "Clinical treatment IRL, offline policy evaluation",
    }
