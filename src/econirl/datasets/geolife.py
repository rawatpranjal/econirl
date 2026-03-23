"""GeoLife GPS Trajectory Dataset.

This module provides access to the GeoLife dataset from Microsoft Research,
containing GPS trajectories from 182 users over 5 years (2007-2012).

The data is suitable for:
- Human mobility pattern learning via IRL
- Transportation mode inference
- Activity recognition

Reference:
    Zheng, Y., et al. (2008-2010). GeoLife GPS Trajectory Dataset.
    Microsoft Research.

Data source:
    https://www.microsoft.com/en-us/download/details.aspx?id=52367
"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd


def load_geolife(
    n_users: Optional[int] = None,
    include_labels: bool = False,
    as_trajectories: bool = False,
    discretize: bool = False,
    grid_size: int = 100,
    seed: Optional[int] = 2008,
) -> pd.DataFrame:
    """Load GeoLife GPS trajectory data.

    The original GeoLife dataset contains 17,621 trajectories from 182 users,
    representing 1.2 million kilometers of travel. Some trajectories include
    transportation mode labels.

    Args:
        n_users: Limit to first N users (None = all available)
        include_labels: Include transportation mode labels where available
        as_trajectories: If True, return list of trajectory arrays
        discretize: If True, convert GPS to discrete grid states
        grid_size: Number of grid cells per dimension if discretizing
        seed: Random seed for sample generation

    Returns:
        DataFrame with columns:
            - user_id: User identifier
            - trajectory_id: Trajectory identifier (trip)
            - timestamp: GPS timestamp
            - latitude: GPS latitude
            - longitude: GPS longitude
            - altitude: Altitude in meters
            - (if include_labels) mode: Transportation mode

    Example:
        >>> from econirl.datasets import load_geolife
        >>> df = load_geolife(n_users=50)
        >>> print(f"Users: {df['user_id'].nunique()}, Trips: {df['trajectory_id'].nunique()}")

        >>> # For mobility IRL
        >>> trajectories = load_geolife(as_trajectories=True, discretize=True)
    """
    data_path = Path(__file__).parent / "geolife_sample.csv"

    if not data_path.exists():
        df = _generate_geolife_sample(seed=seed)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if n_users is not None:
        user_ids = df['user_id'].unique()[:n_users]
        df = df[df['user_id'].isin(user_ids)]

    if not include_labels and 'mode' in df.columns:
        df = df.drop(columns=['mode'])

    if discretize:
        df = _discretize_gps(df, grid_size)

    if as_trajectories:
        return _to_trajectories(df, discretize)

    return df


def _generate_geolife_sample(
    n_users: int = 50,
    trajectories_per_user: int = 5,
    points_per_trajectory: int = 50,
    seed: int = 2008,
) -> pd.DataFrame:
    """Generate synthetic GeoLife-like data."""
    np.random.seed(seed)

    # Beijing area (GeoLife was collected there)
    lon_min, lon_max = 116.2, 116.6
    lat_min, lat_max = 39.7, 40.1

    # Transportation modes
    modes = ['walk', 'bike', 'bus', 'car', 'subway']
    mode_speeds = {'walk': 0.0005, 'bike': 0.001, 'bus': 0.002, 'car': 0.003, 'subway': 0.004}

    records = []
    traj_id = 0

    for user_id in range(1, n_users + 1):
        # User's home location
        home_lon = np.random.uniform(lon_min, lon_max)
        home_lat = np.random.uniform(lat_min, lat_max)

        for _ in range(trajectories_per_user):
            traj_id += 1
            mode = np.random.choice(modes, p=[0.3, 0.2, 0.2, 0.2, 0.1])
            speed = mode_speeds[mode]

            # Start from home or previous endpoint
            lon, lat = home_lon, home_lat
            alt = np.random.uniform(20, 100)

            base_time = pd.Timestamp('2008-01-01') + pd.Timedelta(
                days=np.random.randint(0, 365),
                hours=np.random.randint(6, 22)
            )

            for t in range(points_per_trajectory):
                timestamp = base_time + pd.Timedelta(seconds=t * 5)

                records.append({
                    'user_id': user_id,
                    'trajectory_id': traj_id,
                    'timestamp': timestamp,
                    'latitude': lat,
                    'longitude': lon,
                    'altitude': alt,
                    'mode': mode,
                })

                # Move based on mode
                direction = np.random.uniform(0, 2 * np.pi)
                lon += speed * np.cos(direction)
                lat += speed * np.sin(direction)

                # Keep in bounds
                lon = np.clip(lon, lon_min, lon_max)
                lat = np.clip(lat, lat_min, lat_max)

    return pd.DataFrame(records)


def _discretize_gps(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Convert GPS coordinates to discrete grid cells."""
    df = df.copy()

    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()

    lon_bins = np.linspace(lon_min, lon_max, grid_size + 1)
    lat_bins = np.linspace(lat_min, lat_max, grid_size + 1)

    lon_idx = np.clip(np.digitize(df['longitude'], lon_bins) - 1, 0, grid_size - 1)
    lat_idx = np.clip(np.digitize(df['latitude'], lat_bins) - 1, 0, grid_size - 1)

    df['state'] = lat_idx * grid_size + lon_idx

    return df


def _to_trajectories(df: pd.DataFrame, has_states: bool) -> List[np.ndarray]:
    """Convert DataFrame to list of trajectory arrays."""
    trajectories = []

    for traj_id in df['trajectory_id'].unique():
        traj_data = df[df['trajectory_id'] == traj_id].sort_values('timestamp')

        if has_states:
            traj = traj_data['state'].values
        else:
            traj = traj_data[['longitude', 'latitude']].values

        trajectories.append(traj)

    return trajectories


def get_geolife_info() -> dict:
    """Get metadata about GeoLife dataset."""
    return {
        "name": "GeoLife GPS Trajectories",
        "type": "real (bundled sample) / synthetic fallback",
        "domain": "Human mobility / Activity recognition",
        "n_users_full": 182,
        "n_trajectories_full": 17621,
        "time_span": "5 years (2007-2012)",
        "location": "Beijing, China (primarily)",
        "labeled_portion": "~30% have transportation mode labels",
        "use_cases": [
            "Human mobility IRL",
            "Transportation mode inference",
            "Activity pattern learning",
        ],
        "reference": "Zheng et al. GeoLife. Microsoft Research.",
        "download_url": "https://www.microsoft.com/en-us/download/details.aspx?id=52367",
    }
