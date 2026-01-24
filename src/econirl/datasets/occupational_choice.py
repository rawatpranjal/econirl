"""
Synthetic Occupational Choice Dataset (Keane-Wolpin style).

This module provides a synthetic dataset for occupational choice problems,
inspired by Keane & Wolpin (1997) "The Career Decisions of Young Men."

The data represents individuals making career choices over their working lives:
- State: (education_level, experience, age) discretized
- Actions: 0=school, 1=white_collar, 2=blue_collar, 3=home

Reference:
    Keane, M. P., & Wolpin, K. I. (1997). "The Career Decisions of Young Men."
    Journal of Political Economy, 105(3), 473-522.
"""

import numpy as np
import pandas as pd


def load_occupational_choice(
    n_individuals: int = 500,
    n_periods: int = 40,
    as_panel: bool = False,
    seed: int = 1997,
) -> pd.DataFrame:
    """
    Load synthetic occupational choice data (Keane-Wolpin style).

    This dataset represents individuals making career choices over their working
    lives. The state space combines education level, work experience, and age
    into approximately 100 discrete states. Individuals choose between continuing
    school, working in white-collar or blue-collar jobs, or staying home.

    Args:
        n_individuals: Number of individuals to generate (default: 500)
        n_periods: Number of time periods per individual (default: 40)
        as_panel: If True, return data structured as a Panel object
            compatible with econirl estimators. If False (default),
            return as a pandas DataFrame.
        seed: Random seed for reproducibility (default: 1997)

    Returns:
        DataFrame with columns:
            - id: Individual identifier
            - period: Time period (0-indexed)
            - state: Discretized state index (0-99)
            - action: Chosen action (0=school, 1=white_collar, 2=blue_collar, 3=home)
            - education: Education level (0-4, representing years/degree levels)
            - experience: Work experience (0-9, discretized)
            - age: Age category (0-4, representing age groups)

    Example:
        >>> from econirl.datasets import load_occupational_choice
        >>> df = load_occupational_choice()
        >>> print(f"Observations: {len(df):,}")
        >>> print(f"Individuals: {df['id'].nunique()}")
        >>> print(f"States: {df['state'].nunique()}")

        >>> # Get as Panel for estimation
        >>> panel = load_occupational_choice(as_panel=True)
        >>> print(f"Panel with {panel.num_individuals} individuals")

    Notes:
        State encoding: state = education * 20 + experience * 2 + age // 10
        This gives approximately 5 * 10 * 2 = 100 discrete states.

        Action interpretation:
        - 0 (school): Continue education, increases education level
        - 1 (white_collar): Work in white-collar job, increases experience
        - 2 (blue_collar): Work in blue-collar job, increases experience
        - 3 (home): Stay home (unemployment, family care, etc.)
    """
    df = _generate_occupational_choice_data(n_individuals, n_periods, seed)

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import torch

        # Convert to Panel format
        individual_ids = df["id"].unique()
        trajectories = []

        for ind_id in individual_ids:
            ind_data = df[df["id"] == ind_id].sort_values("period")
            states = torch.tensor(ind_data["state"].values, dtype=torch.long)
            actions = torch.tensor(ind_data["action"].values, dtype=torch.long)
            # Compute next_states (shift states by 1, use 0 for last period)
            next_states = torch.cat([states[1:], torch.tensor([0])])

            traj = Trajectory(
                states=states,
                actions=actions,
                next_states=next_states,
                individual_id=int(ind_id),
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    return df


def _generate_occupational_choice_data(
    n_individuals: int,
    n_periods: int,
    seed: int,
) -> pd.DataFrame:
    """
    Generate synthetic occupational choice data.

    Creates a dataset with realistic choice patterns based on a simple
    model of career decisions. Choice probabilities depend on current
    state (education, experience, age) with reasonable economic patterns:
    - Higher education increases white-collar job probability
    - Experience increases employment probability
    - Age affects schooling eligibility and choice patterns
    """
    np.random.seed(seed)

    # Constants for state encoding
    # education: 0-4 (5 levels)
    # experience: 0-9 (10 levels)
    # age_group: 0-1 (2 groups: young/old)
    # Total states: 5 * 10 * 2 = 100

    records = []

    for ind_id in range(n_individuals):
        # Initial state: everyone starts young with no education/experience
        education = 0
        experience = 0
        age = 0  # Age in periods (0-39)

        for period in range(n_periods):
            # Compute discretized state
            age_group = min(age // 20, 1)  # 0 if age < 20, 1 otherwise
            exp_bin = min(experience, 9)
            edu_bin = min(education, 4)
            state = edu_bin * 20 + exp_bin * 2 + age_group

            # Compute choice probabilities based on state
            # Base probabilities (will be normalized)
            logits = _compute_choice_logits(education, experience, age)

            # Convert to probabilities using softmax
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            # Draw action
            action = np.random.choice(4, p=probs)

            # Record observation
            records.append({
                "id": ind_id,
                "period": period,
                "state": state,
                "action": action,
                "education": edu_bin,
                "experience": exp_bin,
                "age": age_group,
            })

            # State transition based on action
            if action == 0:  # School
                education = min(education + 1, 4)
            elif action in [1, 2]:  # White-collar or blue-collar work
                experience = min(experience + 1, 9)
            # action == 3 (home): no state change

            age += 1

    return pd.DataFrame(records)


def _compute_choice_logits(education: int, experience: int, age: int) -> np.ndarray:
    """
    Compute choice logits based on current state.

    Returns logits for [school, white_collar, blue_collar, home].
    These are designed to produce reasonable choice patterns:
    - School is attractive when young and less educated
    - White-collar requires more education
    - Blue-collar is more accessible
    - Home probability increases when other options are less attractive
    """
    logits = np.zeros(4)

    # School (action 0)
    # More attractive when young, less attractive when already educated
    if age < 20:  # Can only go to school when young enough
        logits[0] = 2.0 - 0.5 * education + 0.1 * np.random.randn()
    else:
        logits[0] = -10.0  # Effectively impossible after age threshold

    # White-collar (action 1)
    # More attractive with higher education
    logits[1] = -1.0 + 0.8 * education + 0.2 * experience + 0.1 * np.random.randn()

    # Blue-collar (action 2)
    # Less dependent on education, more on experience
    logits[2] = 0.5 + 0.1 * education + 0.3 * experience + 0.1 * np.random.randn()

    # Home (action 3)
    # Base option, less attractive with more human capital
    logits[3] = -0.5 - 0.2 * education - 0.2 * experience + 0.1 * np.random.randn()

    return logits


def get_occupational_choice_info() -> dict:
    """
    Get metadata about the occupational choice dataset.

    Returns:
        Dictionary with dataset information including number of states,
        actions, and description of the state/action spaces.
    """
    return {
        "name": "Synthetic Occupational Choice (Keane-Wolpin style)",
        "num_states": 100,
        "num_actions": 4,
        "state_description": {
            "education": "Education level (0-4)",
            "experience": "Work experience (0-9)",
            "age_group": "Age category (0-1: young/old)",
        },
        "action_description": {
            0: "school",
            1: "white_collar",
            2: "blue_collar",
            3: "home",
        },
        "reference": "Keane & Wolpin (1997). Journal of Political Economy, 105(3), 473-522.",
    }
