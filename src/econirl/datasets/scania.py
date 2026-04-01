"""SCANIA Component X replacement dataset.

This module provides a loader for the SCANIA Component X dataset from
the IDA 2024 Industrial Challenge. The original dataset tracks 23,550
heavy trucks with 105 anonymized operational readout features grouped
under 14 sensor families and records whether Component X was repaired
during each vehicle's observation window.

The loader converts the raw survival-style data into a DDC panel
suitable for econirl estimators. The 105 operational features are
reduced to a scalar degradation index via PCA. The first principal
component explains 97 percent of variance across all 105 features,
which means the sensor readings are nearly collinear and a single
degradation axis captures almost all useful signal. The PC1 score
is then discretized into bins to produce a finite state space for
tabular estimators like NFXP and CCP.

This is a single-spell optimal stopping model with right censoring,
not a renewal replacement problem like Rust (1987). Each vehicle is
observed from entry until either repair or end of study. Vehicles
that are not repaired during the study window are right-censored.
After a repair event, no further observations are recorded for that
vehicle.

When the real SCANIA data is not available locally, the loader falls
back to a synthetic dataset that mimics the structure and replacement
rate of the original data.

To use the real data, download the SCANIA Component X dataset from
Kaggle (tapanbatla/scania-component-x-dataset-2025) and pass the
directory path to load_scania(data_dir=...).

Expected files in data_dir:
    train_operational_readouts.csv  (vehicle_id, time_step, 105 features)
    train_tte.csv                   (vehicle_id, length_of_study_time_step,
                                     in_study_repair)

Reference:
    SCANIA Component X dataset, IDA 2024 Industrial Challenge.
    Kaggle: tapanbatla/scania-component-x-dataset-2025
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_scania(
    data_dir: Optional[str | Path] = None,
    as_panel: bool = False,
    num_degradation_bins: int = 50,
    max_vehicles: Optional[int] = None,
) -> Union[pd.DataFrame, "Panel"]:
    """Load the SCANIA Component X replacement dataset.

    If data_dir is provided and contains the real SCANIA CSV files,
    loads and transforms the real data into a DDC panel. Otherwise,
    generates a synthetic dataset that mimics the SCANIA data structure.

    The real data transformation computes a degradation index via PCA
    on the 105 operational readout features. The first principal
    component captures 97 percent of variance and is discretized
    into percentile-based bins. The replacement action is set to 1
    at the vehicle's final observed time step if in_study_repair is
    1, and 0 at all other time steps.

    Args:
        data_dir: Path to directory containing SCANIA CSV files.
            If None, uses synthetic data.
        as_panel: If True, return a Panel object compatible with
            econirl estimators. If False (default), return a DataFrame.
        num_degradation_bins: Number of bins for degradation
            discretization. Default 50.
        max_vehicles: If set, limit to this many vehicles (for
            quick testing).

    Returns:
        DataFrame with columns:
            - vehicle_id: Unique vehicle identifier
            - period: Observation index within each vehicle (0-indexed)
            - time_step: Original continuous time stamp
            - degradation: PC1 score (continuous degradation index)
            - degradation_bin: Discretized degradation state
            - replaced: 1 if component replaced this period, 0 otherwise

        Or Panel if as_panel=True.

    Example:
        >>> from econirl.datasets import load_scania
        >>> df = load_scania()
        >>> print(f"Vehicles: {df['vehicle_id'].nunique()}")
        >>> print(f"Replacement rate: {df['replaced'].mean():.2%}")

        >>> # With real data from Kaggle
        >>> df = load_scania(data_dir="data/scania/Dataset/")
    """
    if data_dir is not None:
        data_dir = Path(data_dir)
        readouts_path = data_dir / "train_operational_readouts.csv"
        tte_path = data_dir / "train_tte.csv"

        if readouts_path.exists() and tte_path.exists():
            df = _load_real_scania(
                data_dir, num_degradation_bins, max_vehicles
            )
        else:
            raise FileNotFoundError(
                f"Expected SCANIA data files in {data_dir}. "
                "Need train_operational_readouts.csv and train_tte.csv. "
                "Download: kaggle datasets download -d "
                "tapanbatla/scania-component-x-dataset-2025"
            )
    else:
        df = _generate_synthetic_scania(num_degradation_bins, max_vehicles)

    if as_panel:
        return _to_panel(df)

    return df


def _load_real_scania(
    data_dir: Path,
    num_degradation_bins: int,
    max_vehicles: Optional[int],
) -> pd.DataFrame:
    """Load and transform real SCANIA data into DDC panel format.

    The pipeline:
    1. Load 1.1M operational readout rows (105 features per row)
    2. Clip outliers at 1st/99th percentile per feature
    3. Standardize with robust scaling (median/IQR)
    4. PCA to extract first principal component as degradation index
    5. Discretize PC1 into percentile-based bins
    6. Construct replacement action from time-to-event data
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    readouts = pd.read_csv(data_dir / "train_operational_readouts.csv")
    tte = pd.read_csv(data_dir / "train_tte.csv")

    # Identify feature columns (everything except vehicle_id and time_step)
    feature_cols = [c for c in readouts.columns
                    if c not in ("vehicle_id", "time_step")]

    # Fill missing values
    readouts[feature_cols] = readouts[feature_cols].fillna(0)

    # Clip outliers at 1st/99th percentile per feature
    for col in feature_cols:
        lo, hi = readouts[col].quantile([0.01, 0.99])
        if hi > lo:
            readouts[col] = readouts[col].clip(lo, hi)

    # Robust standardization (median/IQR, resistant to remaining outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(readouts[feature_cols].values)

    # PCA: first component captures ~97% of variance
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X_scaled).ravel()
    readouts["degradation"] = pc1

    # Percentile-based binning (equal-count bins, not equal-width)
    readouts["degradation_bin"] = pd.qcut(
        readouts["degradation"],
        q=num_degradation_bins,
        labels=False,
        duplicates="drop",
    )

    if max_vehicles is not None:
        vehicle_ids = readouts["vehicle_id"].unique()[:max_vehicles]
        readouts = readouts[readouts["vehicle_id"].isin(vehicle_ids)]

    # Merge with time-to-event data
    merged = readouts[["vehicle_id", "time_step", "degradation",
                        "degradation_bin"]].copy()
    merged = merged.merge(tte, on="vehicle_id", how="left")
    merged = merged.sort_values(["vehicle_id", "time_step"])

    # Replacement action: a_t = 1 at the last observation if repaired
    merged["replaced"] = 0
    last_ts = merged.groupby("vehicle_id")["time_step"].transform("max")
    merged.loc[
        (merged["time_step"] == last_ts) & (merged["in_study_repair"] == 1),
        "replaced"
    ] = 1

    # Period index within each vehicle
    merged["period"] = merged.groupby("vehicle_id").cumcount()

    return merged[["vehicle_id", "period", "time_step", "degradation",
                    "degradation_bin", "replaced"]].reset_index(drop=True)


def _generate_synthetic_scania(
    num_degradation_bins: int = 50,
    max_vehicles: Optional[int] = None,
) -> pd.DataFrame:
    """Generate synthetic data matching SCANIA structure.

    Creates a dataset with roughly 500 vehicles observed over
    varying time horizons (40-80 periods). Parameters are set
    so that the forward-looking agent replaces the component
    when degradation is high enough that expected future operating
    costs exceed the one-time replacement cost.
    """
    rng = np.random.default_rng(2024)

    theta_c = 0.002
    rc = 4.0
    p_degradation = np.array([0.35, 0.55, 0.10])

    n_vehicles = max_vehicles if max_vehicles is not None else 500

    records = []
    vid = 1

    for _ in range(n_vehicles):
        n_periods = rng.integers(40, 81)
        degradation_bin = 0

        for t in range(n_periods):
            degradation = degradation_bin / max(num_degradation_bins - 1, 1)

            v_keep = -theta_c * degradation_bin
            v_replace = -rc
            prob_replace = 1.0 / (1.0 + np.exp(v_keep - v_replace))

            replaced = int(rng.random() < prob_replace)

            records.append({
                "vehicle_id": vid,
                "period": t,
                "time_step": float(t),
                "degradation": degradation,
                "degradation_bin": degradation_bin,
                "replaced": replaced,
            })

            if replaced:
                degradation_bin = 0
            else:
                delta = rng.choice(3, p=p_degradation)
                degradation_bin = min(
                    degradation_bin + delta, num_degradation_bins - 1
                )

        vid += 1

    return pd.DataFrame(records)


def _to_panel(df: pd.DataFrame) -> "Panel":
    """Convert SCANIA DataFrame to Panel object."""
    from econirl.core.types import Panel, Trajectory
    import jax.numpy as jnp

    vehicle_ids = df["vehicle_id"].unique()
    trajectories = []

    for vid in vehicle_ids:
        vdata = df[df["vehicle_id"] == vid].sort_values("period")
        states = jnp.array(vdata["degradation_bin"].values, dtype=jnp.int32)
        actions = jnp.array(vdata["replaced"].values, dtype=jnp.int32)
        next_states = jnp.concatenate([states[1:], jnp.array([0])])

        traj = Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=int(vid),
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def get_scania_info() -> dict:
    """Get metadata about the SCANIA dataset.

    Returns:
        Dictionary with dataset information including number of
        vehicles, observations, and summary statistics.
    """
    df = load_scania()

    return {
        "name": "SCANIA Component X Replacement",
        "source": "IDA 2024 Industrial Challenge (synthetic fallback)",
        "n_observations": len(df),
        "n_vehicles": df["vehicle_id"].nunique(),
        "n_periods_range": (
            df.groupby("vehicle_id")["period"].count().min(),
            df.groupby("vehicle_id")["period"].count().max(),
        ),
        "replacement_rate": df["replaced"].mean(),
        "mean_degradation_bin": df["degradation_bin"].mean(),
        "reference": "SCANIA Component X, IDA 2024 Industrial Challenge",
    }
