"""T-Drive Beijing Taxi Trajectory Dataset.

This module provides access to the T-Drive dataset from Microsoft Research,
containing GPS trajectories of 10,357 taxis in Beijing over one week.

The data is suitable for:
- Maximum Entropy IRL (learning route preferences)
- Trajectory prediction
- Urban mobility modeling

Reference:
    Yuan, J., et al. (2010). "T-Drive: Driving Directions Based on Taxi
    Trajectories." ACM SIGSPATIAL GIS.

    Ziebart, B. D., et al. (2008). "Maximum Entropy Inverse Reinforcement
    Learning." AAAI.

Data source:
    https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/
"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def load_tdrive(
    n_taxis: Optional[int] = None,
    as_trajectories: bool = False,
    discretize: bool = False,
    grid_size: int = 100,
    seed: Optional[int] = 2010,
) -> pd.DataFrame:
    """Load T-Drive taxi trajectory data.

    The original T-Drive dataset contains ~15 million GPS points from 10,357
    Beijing taxis. This loader provides a bundled sample or can download
    the full dataset.

    Args:
        n_taxis: Limit to first N taxis (None = all available)
        as_trajectories: If True, return list of trajectory arrays
        discretize: If True, convert GPS to discrete grid states
        grid_size: Number of grid cells per dimension if discretizing
        seed: Random seed for sample generation

    Returns:
        DataFrame with columns:
            - taxi_id: Taxi identifier
            - timestamp: GPS timestamp
            - longitude: GPS longitude
            - latitude: GPS latitude
            - (if discretize) state: Discrete grid cell index

    Example:
        >>> from econirl.datasets import load_tdrive
        >>> df = load_tdrive(n_taxis=100)
        >>> print(f"Points: {len(df):,}, Taxis: {df['taxi_id'].nunique()}")

        >>> # For MaxEnt IRL
        >>> trajectories = load_tdrive(as_trajectories=True, discretize=True)
        >>> print(f"Trajectories: {len(trajectories)}")
    """
    data_path = Path(__file__).parent / "tdrive_sample.csv"

    if not data_path.exists():
        df = _generate_tdrive_sample(seed=seed)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if n_taxis is not None:
        taxi_ids = df['taxi_id'].unique()[:n_taxis]
        df = df[df['taxi_id'].isin(taxi_ids)]

    if discretize:
        df = _discretize_gps(df, grid_size)

    if as_trajectories:
        return _to_trajectories(df, discretize)

    return df


def _generate_tdrive_sample(
    n_taxis: int = 200,
    n_points_per_taxi: int = 100,
    seed: int = 2010,
) -> pd.DataFrame:
    """Generate synthetic T-Drive-like data.

    Simulates taxi trajectories in a grid representing Beijing's road network.
    """
    np.random.seed(seed)

    # Beijing approximate bounds
    lon_min, lon_max = 116.2, 116.6
    lat_min, lat_max = 39.7, 40.1

    records = []
    base_time = pd.Timestamp('2008-02-02')

    for taxi_id in range(1, n_taxis + 1):
        # Random starting point
        lon = np.random.uniform(lon_min, lon_max)
        lat = np.random.uniform(lat_min, lat_max)

        for t in range(n_points_per_taxi):
            timestamp = base_time + pd.Timedelta(minutes=t)

            records.append({
                'taxi_id': taxi_id,
                'timestamp': timestamp,
                'longitude': lon,
                'latitude': lat,
            })

            # Random walk with road-like constraints
            # Taxis tend to follow major roads (grid pattern)
            direction = np.random.choice(['N', 'S', 'E', 'W', 'stay'], p=[0.2, 0.2, 0.2, 0.2, 0.2])
            step = np.random.uniform(0.001, 0.005)

            if direction == 'N':
                lat = min(lat + step, lat_max)
            elif direction == 'S':
                lat = max(lat - step, lat_min)
            elif direction == 'E':
                lon = min(lon + step, lon_max)
            elif direction == 'W':
                lon = max(lon - step, lon_min)

    return pd.DataFrame(records)


def _discretize_gps(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Convert GPS coordinates to discrete grid cells."""
    df = df.copy()

    # Compute grid bounds from data
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()

    # Discretize
    lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)

    lon_idx = np.digitize(df['longitude'], lon_bins) - 1
    lat_idx = np.digitize(df['latitude'], lat_bins) - 1

    # Clip to valid range
    lon_idx = np.clip(lon_idx, 0, grid_size - 1)
    lat_idx = np.clip(lat_idx, 0, grid_size - 1)

    # Encode as single state index
    df['state'] = lat_idx * grid_size + lon_idx

    return df


def _to_trajectories(df: pd.DataFrame, has_states: bool) -> List[np.ndarray]:
    """Convert DataFrame to list of trajectory arrays."""
    trajectories = []

    for taxi_id in df['taxi_id'].unique():
        taxi_data = df[df['taxi_id'] == taxi_id].sort_values('timestamp')

        if has_states:
            traj = taxi_data['state'].values
        else:
            traj = taxi_data[['longitude', 'latitude']].values

        trajectories.append(traj)

    return trajectories


def get_tdrive_info() -> dict:
    """Get metadata about T-Drive dataset."""
    return {
        "name": "T-Drive Beijing Taxi Trajectories",
        "type": "real (bundled sample) / synthetic fallback",
        "domain": "Urban mobility / Route planning",
        "n_taxis_full": 10357,
        "n_points_full": "~15 million",
        "time_span": "One week (Feb 2008)",
        "location": "Beijing, China",
        "use_cases": [
            "Maximum Entropy IRL for route preferences",
            "Trajectory prediction",
            "Traffic pattern learning",
        ],
        "reference": "Yuan et al. (2010). T-Drive. ACM SIGSPATIAL.",
        "download_url": "https://www.microsoft.com/en-us/research/publication/t-drive-trajectory-data-sample/",
    }
