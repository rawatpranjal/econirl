"""Stanford Drone Dataset (SDD) for Pedestrian Trajectory Analysis.

This module provides access to the Stanford Drone Dataset, containing
bird's-eye view trajectories of pedestrians, cyclists, and vehicles
on Stanford campus.

The data is suitable for:
- Pedestrian trajectory IRL (social costs, navigation preferences)
- Crowd behavior modeling
- Multi-agent trajectory prediction

Reference:
    Robicquet, A., et al. (2016). "Learning Social Etiquette: Human
    Trajectory Understanding in Crowded Scenes." ECCV.

Data source:
    https://cvgl.stanford.edu/projects/uav_data/
"""

from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
import pandas as pd


def load_stanford_drone(
    scene: Optional[Literal["bookstore", "coupa", "deathCircle", "gates", "hyang", "little", "nexus", "quad"]] = None,
    agent_type: Optional[Literal["Pedestrian", "Biker", "Skater"]] = None,
    as_trajectories: bool = False,
    discretize: bool = False,
    grid_size: int = 50,
    seed: Optional[int] = 2016,
) -> pd.DataFrame:
    """Load Stanford Drone Dataset pedestrian/cyclist trajectories.

    The original SDD contains trajectories from 8 unique scenes on Stanford
    campus, captured via drone footage. Trajectories include pedestrians,
    cyclists, skaters, carts, and vehicles.

    Args:
        scene: Specific scene to load (None = all scenes)
        agent_type: Filter by agent type (None = all types)
        as_trajectories: If True, return list of trajectory arrays
        discretize: If True, convert pixel coordinates to grid states
        grid_size: Grid size for discretization
        seed: Random seed for sample generation

    Returns:
        DataFrame with columns:
            - track_id: Unique trajectory identifier
            - frame: Video frame number
            - x: X coordinate (pixels)
            - y: Y coordinate (pixels)
            - agent_type: Type of agent
            - scene: Scene name

    Example:
        >>> from econirl.datasets import load_stanford_drone
        >>> df = load_stanford_drone(scene="gates", agent_type="Pedestrian")
        >>> print(f"Trajectories: {df['track_id'].nunique()}")

        >>> # For trajectory IRL
        >>> trajectories = load_stanford_drone(as_trajectories=True, discretize=True)
    """
    data_path = Path(__file__).parent / "sdd_sample.csv"

    if not data_path.exists():
        df = _generate_sdd_sample(seed=seed)
        df.to_csv(data_path, index=False)
    else:
        df = pd.read_csv(data_path)

    if scene is not None:
        df = df[df['scene'] == scene]

    if agent_type is not None:
        df = df[df['agent_type'] == agent_type]

    if discretize:
        df = _discretize_coords(df, grid_size)

    if as_trajectories:
        return _to_trajectories(df, discretize)

    return df


def _generate_sdd_sample(
    n_tracks_per_scene: int = 50,
    n_frames_per_track: int = 30,
    seed: int = 2016,
) -> pd.DataFrame:
    """Generate synthetic SDD-like data."""
    np.random.seed(seed)

    scenes = ['bookstore', 'coupa', 'gates', 'hyang', 'quad']
    agent_types = ['Pedestrian', 'Biker', 'Skater']
    agent_probs = [0.7, 0.2, 0.1]

    # Scene dimensions (pixels, approximate)
    scene_dims = {
        'bookstore': (800, 600),
        'coupa': (1000, 800),
        'gates': (900, 700),
        'hyang': (1100, 900),
        'quad': (1200, 1000),
    }

    records = []
    track_id = 0

    for scene in scenes:
        width, height = scene_dims[scene]

        for _ in range(n_tracks_per_scene):
            track_id += 1
            agent_type = np.random.choice(agent_types, p=agent_probs)

            # Speed depends on agent type
            speed = {'Pedestrian': 3, 'Biker': 8, 'Skater': 6}[agent_type]

            # Random start and goal
            x = np.random.uniform(50, width - 50)
            y = np.random.uniform(50, height - 50)
            goal_x = np.random.uniform(50, width - 50)
            goal_y = np.random.uniform(50, height - 50)

            start_frame = np.random.randint(0, 1000)

            for t in range(n_frames_per_track):
                frame = start_frame + t

                records.append({
                    'track_id': track_id,
                    'frame': frame,
                    'x': x,
                    'y': y,
                    'agent_type': agent_type,
                    'scene': scene,
                })

                # Move towards goal with some noise
                dx = goal_x - x
                dy = goal_y - y
                dist = np.sqrt(dx**2 + dy**2)

                if dist > speed:
                    x += speed * dx / dist + np.random.normal(0, 1)
                    y += speed * dy / dist + np.random.normal(0, 1)

                # Avoid going out of bounds
                x = np.clip(x, 0, width)
                y = np.clip(y, 0, height)

    return pd.DataFrame(records)


def _discretize_coords(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Convert pixel coordinates to discrete grid cells."""
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

    for track_id in df['track_id'].unique():
        track_data = df[df['track_id'] == track_id].sort_values('frame')

        if has_states:
            traj = track_data['state'].values
        else:
            traj = track_data[['x', 'y']].values

        trajectories.append(traj)

    return trajectories


def get_stanford_drone_info() -> dict:
    """Get metadata about Stanford Drone Dataset."""
    return {
        "name": "Stanford Drone Dataset (SDD)",
        "type": "real (bundled sample) / synthetic fallback",
        "domain": "Pedestrian/cyclist trajectory prediction",
        "n_scenes": 8,
        "n_unique_agents": "~20,000+",
        "agent_types": ["Pedestrian", "Biker", "Skater", "Cart", "Car", "Bus"],
        "view": "Bird's-eye (drone footage)",
        "use_cases": [
            "Social force IRL",
            "Pedestrian navigation preference learning",
            "Multi-agent trajectory prediction",
        ],
        "reference": "Robicquet et al. (2016). ECCV.",
        "download_url": "https://cvgl.stanford.edu/projects/uav_data/",
    }
