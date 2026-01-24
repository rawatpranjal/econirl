"""Rust (1987) table replication functions."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from econirl.datasets import load_rust_bus
from econirl.estimation.transitions import estimate_transition_probs_by_group


def table_ii_descriptives(
    df: Optional[pd.DataFrame] = None,
    original: bool = True,
) -> pd.DataFrame:
    """Replicate Table II: Descriptive Statistics on Odometer Readings.

    This function computes descriptive statistics matching Table II from
    Rust (1987), including:
    - Number of buses per group
    - Number of engine replacements
    - Mean and std of mileage at replacement
    - Mean and std of final observed mileage

    Args:
        df: DataFrame with bus data. If None, loads from load_rust_bus().
        original: If df is None, whether to load original (True) or synthetic data.

    Returns:
        DataFrame indexed by group with descriptive statistics columns.

    Example:
        >>> from econirl.replication.rust1987 import table_ii_descriptives
        >>> table = table_ii_descriptives()
        >>> print(table)
    """
    if df is None:
        df = load_rust_bus(original=original)

    results = []

    for group in sorted(df['group'].unique()):
        group_df = df[df['group'] == group]

        n_buses = group_df['bus_id'].nunique()
        n_replacements = group_df['replaced'].sum()

        # Mileage at replacement
        replacement_obs = group_df[group_df['replaced'] == 1]
        if len(replacement_obs) > 0:
            mean_mileage = replacement_obs['mileage'].mean()
            std_mileage = replacement_obs['mileage'].std()
        else:
            mean_mileage = np.nan
            std_mileage = np.nan

        # Final mileage (last observation per bus)
        final_obs = group_df.groupby('bus_id').last()
        mean_final = final_obs['mileage'].mean()
        std_final = final_obs['mileage'].std()

        results.append({
            'group': group,
            'n_buses': n_buses,
            'n_replacements': int(n_replacements),
            'mean_mileage': mean_mileage,
            'std_mileage': std_mileage,
            'mean_final_mileage': mean_final,
            'std_final_mileage': std_final,
        })

    return pd.DataFrame(results).set_index('group')


def table_iv_transitions(
    df: Optional[pd.DataFrame] = None,
    original: bool = True,
) -> pd.DataFrame:
    """Replicate Table IV: Transition Probability Estimates.

    This function computes the mileage transition probability estimates
    (theta_0, theta_1, theta_2) for each bus group, matching Table IV
    from Rust (1987).

    Args:
        df: DataFrame with bus data. If None, loads from load_rust_bus().
        original: If df is None, whether to load original (True) or synthetic data.

    Returns:
        DataFrame indexed by group with transition probability columns:
        - theta_0: P(mileage increase = 0 bins)
        - theta_1: P(mileage increase = 1 bin)
        - theta_2: P(mileage increase = 2 bins)
        - n_transitions: Number of transitions used in estimation

    Example:
        >>> from econirl.replication.rust1987 import table_iv_transitions
        >>> table = table_iv_transitions()
        >>> print(table)
    """
    if df is None:
        df = load_rust_bus(original=original)

    probs_by_group = estimate_transition_probs_by_group(df)

    results = []
    for group, probs in probs_by_group.items():
        results.append({
            'group': group,
            'theta_0': probs[0],
            'theta_1': probs[1],
            'theta_2': probs[2],
            'n_transitions': len(df[(df['group'] == group) & (df['replaced'] == 0)]),
        })

    return pd.DataFrame(results).set_index('group')
