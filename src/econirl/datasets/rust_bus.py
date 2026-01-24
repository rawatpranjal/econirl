"""
Rust (1987) Bus Engine Replacement Dataset.

This module provides the original data from John Rust's seminal 1987 Econometrica
paper "Optimal Replacement of GMC Bus Engines: An Empirical Model of Harold Zurcher."

The data consists of monthly observations of bus mileage and engine replacement
decisions from the Madison Metropolitan Bus Company.

Reference:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical Model
    of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def load_rust_bus(
    group: Optional[int] = None,
    as_panel: bool = False,
    original: bool = False,
) -> pd.DataFrame:
    """
    Load the Rust (1987) bus engine replacement dataset.

    This dataset contains monthly observations of odometer readings and engine
    replacement decisions for buses from the Madison Metropolitan Bus Company.
    The data was used in Rust's pioneering work on dynamic discrete choice models.

    Args:
        group: If specified, load only buses from this group (1-8 for synthetic,
            1-4 for original). Groups differ by bus type and usage patterns.
            If None, loads all groups.
        as_panel: If True, return data structured as a Panel object
            compatible with econirl estimators. If False (default),
            return as a pandas DataFrame.
        original: If True, load original Rust (1987) data from NFXP files.
            If False (default), load synthetic data with similar characteristics.

    Returns:
        DataFrame with columns:
            - bus_id: Unique bus identifier
            - period: Month number (1-indexed)
            - mileage: Odometer reading (in thousands of miles)
            - mileage_bin: Discretized mileage state (0-89)
            - replaced: 1 if engine was replaced this period, 0 otherwise
            - group: Bus group (1-4 for original, 1-8 for synthetic)

    Example:
        >>> from econirl.datasets import load_rust_bus
        >>> df = load_rust_bus()
        >>> print(f"Observations: {len(df):,}")
        >>> print(f"Buses: {df['bus_id'].nunique()}")
        >>> print(f"Replacement rate: {df['replaced'].mean():.2%}")

        >>> # Load original data for replication
        >>> df_original = load_rust_bus(original=True)
        >>> print(f"Original data: {len(df_original):,} observations")

    Notes:
        The original Rust (1987) paper used groups 1-4:
        - Group 1: Grumman model 870 (15 buses)
        - Group 2: Chance model RT50 (4 buses)
        - Group 3: GMC model T8H203 (48 buses)
        - Group 4: GMC model A5308, 1975 (37 buses)

        For the synthetic data, groups 5-8 represent additional bus types
        with different characteristics.

        Mileage bins follow Rust's discretization: each bin represents 5,000 miles,
        so bin 0 = [0, 5000), bin 1 = [5000, 10000), etc.
    """
    if original:
        data_path = Path(__file__).parent / "rust_bus_original.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                "Original Rust data not found. Run: python scripts/download_rust_data.py"
            )
        df = pd.read_csv(data_path)
        max_group = 4
    else:
        data_path = Path(__file__).parent / "rust_bus_data.csv"
        if not data_path.exists():
            # Generate the dataset if it doesn't exist
            df = _generate_rust_bus_data()
            df.to_csv(data_path, index=False)
        else:
            df = pd.read_csv(data_path)
        max_group = 8

    if group is not None:
        if group not in range(1, max_group + 1):
            raise ValueError(f"group must be between 1 and {max_group}, got {group}")
        df = df[df["group"] == group].copy()

    if as_panel:
        from econirl.core.types import Panel, Trajectory
        import torch

        # Convert to Panel format
        bus_ids = df["bus_id"].unique()
        trajectories = []

        for bus_id in bus_ids:
            bus_data = df[df["bus_id"] == bus_id].sort_values("period")
            states = torch.tensor(bus_data["mileage_bin"].values, dtype=torch.long)
            actions = torch.tensor(bus_data["replaced"].values, dtype=torch.long)
            # Compute next_states (shift states by 1, use 0 for last period)
            next_states = torch.cat([states[1:], torch.tensor([0])])

            traj = Trajectory(
                states=states,
                actions=actions,
                next_states=next_states,
                individual_id=int(bus_id),
            )
            trajectories.append(traj)

        return Panel(trajectories=trajectories)

    return df


def _generate_rust_bus_data() -> pd.DataFrame:
    """
    Generate synthetic data matching the structure of Rust (1987).

    This creates a dataset with realistic characteristics matching the original
    Rust data: similar replacement rates, mileage distributions, and group
    structures.
    """
    np.random.seed(1987)  # Reproducible

    # Parameters roughly matching Rust's estimates
    # Operating cost per 5000 miles (per bin)
    theta_c = 0.001
    # Replacement cost (in utils)
    RC = 3.0
    # Discount factor
    beta = 0.9999
    # Mileage transition probabilities (stay, +1, +2 bins)
    p_mileage = np.array([0.3919, 0.5953, 0.0128])

    records = []

    # 8 groups with different characteristics
    group_configs = [
        {"n_buses": 15, "n_periods": 120, "base_mileage_rate": 1.0},  # Group 1
        {"n_buses": 15, "n_periods": 120, "base_mileage_rate": 1.1},  # Group 2
        {"n_buses": 10, "n_periods": 100, "base_mileage_rate": 0.9},  # Group 3
        {"n_buses": 12, "n_periods": 110, "base_mileage_rate": 1.0},  # Group 4
        {"n_buses": 10, "n_periods": 90, "base_mileage_rate": 1.2},   # Group 5
        {"n_buses": 8, "n_periods": 80, "base_mileage_rate": 0.8},    # Group 6
        {"n_buses": 10, "n_periods": 100, "base_mileage_rate": 1.0},  # Group 7
        {"n_buses": 10, "n_periods": 95, "base_mileage_rate": 1.05},  # Group 8
    ]

    bus_id_counter = 1

    for group_idx, config in enumerate(group_configs):
        group = group_idx + 1

        for _ in range(config["n_buses"]):
            mileage_bin = 0
            cumulative_mileage = 0.0

            for period in range(1, config["n_periods"] + 1):
                # Compute choice probabilities using logit formula
                # V(keep) ~ -theta_c * mileage
                # V(replace) ~ -RC
                v_keep = -theta_c * mileage_bin
                v_replace = -RC

                # Logit choice probability
                prob_replace = 1 / (1 + np.exp(v_keep - v_replace))

                # Add some randomness based on group
                prob_replace *= config["base_mileage_rate"]
                prob_replace = np.clip(prob_replace, 0, 1)

                # Draw replacement decision
                replaced = int(np.random.random() < prob_replace)

                # Record observation
                records.append({
                    "bus_id": bus_id_counter,
                    "period": period,
                    "mileage": cumulative_mileage,
                    "mileage_bin": mileage_bin,
                    "replaced": replaced,
                    "group": group,
                })

                # State transition
                if replaced:
                    mileage_bin = 0
                    cumulative_mileage = 0.0
                else:
                    # Mileage increment
                    increment = np.random.choice([0, 1, 2], p=p_mileage)
                    mileage_bin = min(mileage_bin + increment, 89)
                    cumulative_mileage += increment * 5.0  # 5000 miles per bin

            bus_id_counter += 1

    return pd.DataFrame(records)


def get_rust_bus_info() -> dict:
    """
    Get metadata about the Rust bus dataset.

    Returns:
        Dictionary with dataset information including number of buses,
        observations, groups, and summary statistics.
    """
    df = load_rust_bus()

    return {
        "name": "Rust (1987) Bus Engine Replacement",
        "n_observations": len(df),
        "n_buses": df["bus_id"].nunique(),
        "n_groups": df["group"].nunique(),
        "n_periods_range": (
            df.groupby("bus_id")["period"].count().min(),
            df.groupby("bus_id")["period"].count().max(),
        ),
        "replacement_rate": df["replaced"].mean(),
        "mean_mileage_bin": df["mileage_bin"].mean(),
        "reference": "Rust, J. (1987). Econometrica, 55(5), 999-1033.",
    }
