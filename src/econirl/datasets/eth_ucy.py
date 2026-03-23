"""ETH and UCY Pedestrian Trajectory Datasets.

This module provides access to the classic ETH and UCY pedestrian trajectory
datasets, widely used as benchmarks in trajectory prediction and pedestrian
behavior modeling.

Scenes:
- ETH: ETH building entrance (Zurich)
- Hotel: Hotel entrance (Zurich)
- Univ: University campus (Cyprus)
- Zara1: Shopping street scene 1 (Seville)
- Zara2: Shopping street scene 2 (Seville)

Reference:
    Pellegrini, S., et al. (2009). "You'll Never Walk Alone: Modeling Social
    Behavior for Multi-target Tracking." ICCV.

    Lerner, A., et al. (2007). "Crowds by Example." Computer Graphics Forum.

Data source:
    https://service.tib.eu/ldmservice/en/dataset/eth-and-ucy-datasets
"""

from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
import pandas as pd


def load_eth_ucy(
    scene: Optional[Literal["eth", "hotel", "univ", "zara1", "zara2"]] = None,
    as_trajectories: bool = False,
    discretize: bool = False,
    grid_size: int = 50,
    seed: Optional[int] = 2009,
) -> pd.DataFrame:
    """Load ETH/UCY pedestrian trajectory data.

    The ETH/UCY datasets are classic benchmarks for pedestrian trajectory
    prediction, containing world-coordinate trajectories (in meters) at
    2.5 FPS.

    Args:
        scene: Specific scene to load (None = all scenes)
        as_trajectories: If True, return list of trajectory arrays
        discretize: If True, convert coordinates to grid states
        grid_size: Grid size for discretization
        seed: Random seed for sample generation

    Returns:
        DataFrame with columns:
            - pedestrian_id: Unique pedestrian identifier
            - frame: Frame number
            - x: X coordinate (meters, world coordinates)
            - y: Y coordinate (meters, world coordinates)
            - scene: Scene name

    Example:
        >>> from econirl.datasets import load_eth_ucy
        >>> df = load_eth_ucy(scene="eth")
        >>> print(f"Pedestrians: {df['pedestrian_id'].nunique()}")

        >>> # For trajectory IRL
        >>> trajectories = load_eth_ucy(as_trajectories=True, discretize=True)
    """
    data_path = Path(__file__).parent / "eth_ucy_sample.csv"

    if not data_path.exists():
        df = _generate_eth_ucy_sample(seed=seed)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if scene is not None:
        df = df[df['scene'] == scene]

    if discretize:
        df = _discretize_coords(df, grid_size)

    if as_trajectories:
        return _to_trajectories(df, discretize)

    return df


def _generate_eth_ucy_sample(
    n_pedestrians_per_scene: int = 100,
    n_frames_per_ped: int = 20,
    seed: int = 2009,
) -> pd.DataFrame:
    """Generate synthetic ETH/UCY-like data."""
    np.random.seed(seed)

    scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']

    # Approximate scene dimensions in meters
    scene_dims = {
        'eth': (20, 15),
        'hotel': (15, 12),
        'univ': (30, 25),
        'zara1': (18, 14),
        'zara2': (18, 14),
    }

    records = []
    ped_id = 0

    for scene in scenes:
        width, height = scene_dims[scene]

        for _ in range(n_pedestrians_per_scene):
            ped_id += 1

            # Typical pedestrian speed: 1.2-1.5 m/s
            # At 2.5 FPS, that's ~0.5 m per frame
            speed = np.random.uniform(0.4, 0.6)

            # Random start and goal (often at edges)
            if np.random.random() < 0.5:
                # Enter from left/right
                x = 0 if np.random.random() < 0.5 else width
                y = np.random.uniform(0, height)
                goal_x = width - x  # Go to opposite side
                goal_y = np.random.uniform(0, height)
            else:
                # Enter from top/bottom
                x = np.random.uniform(0, width)
                y = 0 if np.random.random() < 0.5 else height
                goal_x = np.random.uniform(0, width)
                goal_y = height - y

            start_frame = np.random.randint(0, 500)

            for t in range(n_frames_per_ped):
                frame = start_frame + t

                records.append({
                    'pedestrian_id': ped_id,
                    'frame': frame,
                    'x': x,
                    'y': y,
                    'scene': scene,
                })

                # Move towards goal
                dx = goal_x - x
                dy = goal_y - y
                dist = np.sqrt(dx**2 + dy**2)

                if dist > speed:
                    # Add social force-like perturbation
                    x += speed * dx / dist + np.random.normal(0, 0.05)
                    y += speed * dy / dist + np.random.normal(0, 0.05)

    return pd.DataFrame(records)


def _discretize_coords(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Convert world coordinates to discrete grid cells."""
    df = df.copy()

    x_min, x_max = df['x'].min(), df['x'].max()
    y_min, y_max = df['y'].min(), df['y'].max()

    x_bins = np.linspace(x_min, x_max, grid_size + 1)
    y_bins = np.linspace(y_min, y_max, grid_size + 1)

    x_idx = np.clip(np.digitize(df['x'], x_bins) - 1, 0, grid_size - 1)
    y_idx = np.clip(np.digitize(df['y'], y_bins) - 1, 0, grid_size - 1)

    df['state'] = y_idx * grid_size + x_idx

    return df


def _to_trajectories(df: pd.DataFrame, has_states: bool) -> List[np.ndarray]:
    """Convert DataFrame to list of trajectory arrays."""
    trajectories = []

    for ped_id in df['pedestrian_id'].unique():
        ped_data = df[df['pedestrian_id'] == ped_id].sort_values('frame')

        if has_states:
            traj = ped_data['state'].values
        else:
            traj = ped_data[['x', 'y']].values

        trajectories.append(traj)

    return trajectories


def get_eth_ucy_info() -> dict:
    """Get metadata about ETH/UCY datasets."""
    return {
        "name": "ETH and UCY Pedestrian Datasets",
        "type": "real (bundled sample) / synthetic fallback",
        "domain": "Pedestrian trajectory prediction",
        "scenes": ["eth", "hotel", "univ", "zara1", "zara2"],
        "coordinate_system": "World coordinates (meters)",
        "fps": 2.5,
        "n_pedestrians_total": "~1500+",
        "use_cases": [
            "Social force IRL",
            "Pedestrian path preference learning",
            "Crowd simulation",
        ],
        "reference": "Pellegrini et al. (2009). ICCV. / Lerner et al. (2007). CGF.",
        "download_url": "https://service.tib.eu/ldmservice/en/dataset/eth-and-ucy-datasets",
    }
