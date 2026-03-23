"""Keane & Wolpin (1994) Career Decisions Dataset.

This module provides the Keane-Wolpin career choice dataset, which tracks
individuals making choices between schooling, white-collar work, blue-collar
work, and home production.

Reference:
    Keane, M. P., & Wolpin, K. I. (1994). "The Solution and Estimation of
    Discrete Choice Dynamic Programming Models by Simulation and Interpolation:
    Monte Carlo Evidence." The Review of Economics and Statistics, 76(4), 648-672.

    Keane, M. P., & Wolpin, K. I. (1997). "The Career Decisions of Young Men."
    Journal of Political Economy, 105(3), 473-522.
"""

from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd


def load_keane_wolpin(
    version: Literal["kw_94", "kw_97"] = "kw_94",
    as_panel: bool = False,
    source: Literal["respy", "bundled"] = "bundled",
) -> pd.DataFrame:
    """Load the Keane & Wolpin career decisions dataset.

    This dataset tracks individuals choosing between:
    - 0: Schooling (accumulate education)
    - 1: White-collar work (accumulate white-collar experience)
    - 2: Blue-collar work (accumulate blue-collar experience)
    - 3: Home production (no state accumulation)

    State variables include:
    - schooling: Years of completed education
    - exp_white_collar: Years of white-collar experience
    - exp_blue_collar: Years of blue-collar experience
    - age: Current age

    Args:
        version: Which version of the KW model to load
            - "kw_94": Original 1994 REStat specification
            - "kw_97": Extended 1997 JPE specification
        as_panel: If True, return as Panel object for econirl estimators
        source: Data source
            - "respy": Load from respy package (if installed)
            - "bundled": Load bundled sample data

    Returns:
        DataFrame with columns:
            - id: Individual identifier
            - period: Decision period (1-indexed)
            - age: Current age
            - schooling: Years of education
            - exp_white_collar: White-collar experience
            - exp_blue_collar: Blue-collar experience
            - choice: Chosen action (0-3)

    Example:
        >>> from econirl.datasets import load_keane_wolpin
        >>> df = load_keane_wolpin()
        >>> print(f"Individuals: {df['id'].nunique()}")
        >>> print(f"Choice distribution:\\n{df['choice'].value_counts()}")

    Notes:
        For full replication of KW94/KW97, install respy:
        `pip install respy`

        The bundled sample data is suitable for testing and tutorials.
    """
    if source == "respy":
        try:
            return _load_from_respy(version, as_panel)
        except ImportError:
            import warnings
            warnings.warn(
                "respy not installed. Falling back to bundled data. "
                "Install with: pip install respy"
            )
            source = "bundled"

    if source == "bundled":
        return _load_bundled(as_panel)

    raise ValueError(f"Unknown source: {source}")


def _load_from_respy(version: str, as_panel: bool) -> pd.DataFrame:
    """Load data from respy package."""
    import respy

    # Map version to respy model name
    model_map = {
        "kw_94": "kw_94_one",
        "kw_97": "kw_97_basic",
    }

    model_name = model_map.get(version, "kw_94_one")

    # Get example model with data
    _, _, df = respy.get_example_model(model_name, with_data=True)

    # respy uses MultiIndex (Identifier, Period)
    df = df.reset_index()

    # Standardize column names
    rename_map = {
        'Identifier': 'id',
        'Period': 'period',
        'Age': 'age',
        'Years_Of_Schooling': 'schooling',
        'Experience_White_Collar': 'exp_white_collar',
        'Experience_Blue_Collar': 'exp_blue_collar',
        'Choice': 'choice',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure choice is 0-indexed
    if 'choice' in df.columns:
        if df['choice'].min() == 1:
            df['choice'] = df['choice'] - 1

    if as_panel:
        return _to_panel(df)

    return df


def _load_bundled(as_panel: bool) -> pd.DataFrame:
    """Load bundled sample data."""
    data_path = Path(__file__).parent / "keane_wolpin_sample.csv"

    if not data_path.exists():
        # Generate sample data if not present
        df = _generate_kw_sample()
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if as_panel:
        return _to_panel(df)

    return df


def _generate_kw_sample(
    n_individuals: int = 500,
    n_periods: int = 10,
    seed: int = 1994,
) -> pd.DataFrame:
    """Generate synthetic Keane-Wolpin style data."""
    np.random.seed(seed)

    records = []
    for i in range(1, n_individuals + 1):
        schooling = 10 + np.random.randint(0, 7)
        exp_white = 0
        exp_blue = 0

        for t in range(1, n_periods + 1):
            age = 16 + t

            # Simple choice model
            if age <= 22 and schooling < 16:
                p_school = 0.6 - 0.05 * (schooling - 10)
                p_school = max(0.1, min(0.8, p_school))
                if np.random.random() < p_school:
                    choice = 0
                else:
                    choice = np.random.choice([1, 2, 3], p=[0.4, 0.4, 0.2])
            else:
                # Experience-dependent probabilities
                total_exp = exp_white + exp_blue
                p_white = 0.3 + 0.02 * exp_white
                p_blue = 0.3 + 0.02 * exp_blue
                p_home = 0.15
                p_school = max(0, 1 - p_white - p_blue - p_home)
                probs = np.array([p_school, p_white, p_blue, p_home])
                probs = probs / probs.sum()
                choice = np.random.choice([0, 1, 2, 3], p=probs)

            records.append({
                'id': i,
                'period': t,
                'age': age,
                'schooling': schooling,
                'exp_white_collar': exp_white,
                'exp_blue_collar': exp_blue,
                'choice': choice,
            })

            # State transitions
            if choice == 0:
                schooling = min(schooling + 1, 20)
            elif choice == 1:
                exp_white += 1
            elif choice == 2:
                exp_blue += 1

    return pd.DataFrame(records)


def _to_panel(df: pd.DataFrame):
    """Convert DataFrame to Panel format."""
    from econirl.core.types import Panel, Trajectory
    import torch

    # Create composite state from individual state variables
    # For KW, state = (schooling, exp_white, exp_blue) encoded as single int
    # This is a simplification; full implementation would use multi-dimensional states

    def encode_state(row):
        """Encode state tuple as single integer."""
        # schooling: 0-20, exp_white: 0-30, exp_blue: 0-30
        return (row['schooling'] * 31 * 31 +
                row['exp_white_collar'] * 31 +
                row['exp_blue_collar'])

    df = df.copy()
    df['state'] = df.apply(encode_state, axis=1)

    trajectories = []
    for ind_id in df['id'].unique():
        ind_data = df[df['id'] == ind_id].sort_values('period')

        states = torch.tensor(ind_data['state'].values, dtype=torch.long)
        actions = torch.tensor(ind_data['choice'].values, dtype=torch.long)
        next_states = torch.cat([states[1:], states[-1:]])

        traj = Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=int(ind_id),
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def get_keane_wolpin_info() -> dict:
    """Get metadata about the Keane-Wolpin dataset."""
    df = load_keane_wolpin()

    return {
        "name": "Keane & Wolpin (1994/1997) Career Decisions",
        "n_observations": len(df),
        "n_individuals": df['id'].nunique(),
        "n_periods": df['period'].max(),
        "n_choices": df['choice'].nunique(),
        "choices": {
            0: "Schooling",
            1: "White-collar work",
            2: "Blue-collar work",
            3: "Home production",
        },
        "state_variables": ["schooling", "exp_white_collar", "exp_blue_collar"],
        "reference": "Keane & Wolpin (1994). REStat, 76(4), 648-672.",
    }
