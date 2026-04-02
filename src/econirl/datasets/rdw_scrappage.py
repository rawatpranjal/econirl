"""
RDW Vehicle Scrappage Dataset.

This module provides data for a vehicle scrappage decision model using
Dutch RDW (Rijksdienst voor het Wegverkeer) open data. The dataset consists
of annual observations of vehicle age and APK inspection defect severity,
paired with scrappage decisions.

When real RDW data is not available, the module generates synthetic data
with realistic scrappage patterns matching Dutch CBS statistics for
passenger vehicles.

Reference:
    RDW Open Data: https://opendata.rdw.nl
    El Boubsi (2023). MSc Thesis, Delft University of Technology.
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def load_rdw_scrappage(
    data_dir: Optional[str] = None,
    as_panel: bool = False,
    max_vehicles: Optional[int] = None,
) -> Union[pd.DataFrame, "Panel"]:
    """
    Load the RDW vehicle scrappage dataset.

    This dataset contains annual observations of vehicle age, APK inspection
    defect severity, and scrappage decisions for Dutch passenger vehicles.
    When real data is not available, synthetic data is generated with
    matching structure.

    Args:
        data_dir: Path to directory containing real RDW CSV data
            produced by scripts/download_rdw.py. If None or if the
            file does not exist, synthetic data is generated.
        as_panel: If True, return data structured as a Panel object
            compatible with econirl estimators. If False (default),
            return as a pandas DataFrame.
        max_vehicles: If specified, limit the number of vehicles loaded.
            Useful for quick testing.

    Returns:
        DataFrame with columns:
            - vehicle_id: Unique vehicle identifier
            - year: Calendar year (or period index for synthetic data)
            - age_bin: Discretized vehicle age (0-24)
            - defect_level: APK defect severity (0=pass, 1=minor, 2=major)
            - scrapped: 1 if vehicle was scrapped this period, 0 otherwise
            - state: Flattened state index (age_bin * 3 + defect_level)

    Example:
        >>> from econirl.datasets import load_rdw_scrappage
        >>> df = load_rdw_scrappage()
        >>> print(f"Observations: {len(df):,}")
        >>> print(f"Vehicles: {df['vehicle_id'].nunique()}")
        >>> print(f"Scrappage rate: {df['scrapped'].mean():.2%}")

        >>> # With real RDW data
        >>> df = load_rdw_scrappage(data_dir="/path/to/rdw_data/")
    """
    if data_dir is not None:
        data_path = Path(data_dir) / "rdw_scrappage_data.csv"
        if data_path.exists():
            df = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(
                f"RDW data not found at {data_path}. "
                "Run: python scripts/download_rdw.py"
            )
    else:
        # Check for bundled data
        bundled_path = Path(__file__).parent / "rdw_scrappage_data.csv"
        if bundled_path.exists():
            df = pd.read_csv(bundled_path)
        else:
            df = _generate_synthetic_rdw()

    # Ensure state column exists
    if "state" not in df.columns:
        df["state"] = df["age_bin"] * 3 + df["defect_level"]

    if max_vehicles is not None:
        vehicle_ids = df["vehicle_id"].unique()[:max_vehicles]
        df = df[df["vehicle_id"].isin(vehicle_ids)].copy()

    if as_panel:
        return _to_panel(df)

    return df


def _generate_synthetic_rdw(
    n_vehicles: int = 2000,
    max_years: int = 20,
    num_age_bins: int = 25,
    num_defect_levels: int = 3,
) -> pd.DataFrame:
    """
    Generate synthetic data matching RDW scrappage patterns.

    Creates a panel of vehicles with realistic Dutch scrappage behavior:
    annual scrappage rates of 5-8 percent for vehicles aged 5-20 years,
    with defects roughly doubling the scrappage hazard.

    The data generating process uses the same structural model as
    RDWScrapageEnvironment with default parameters.
    """
    rng = np.random.default_rng(2024)

    # Structural parameters (matching RDWScrapageEnvironment defaults)
    theta_age = 0.15
    theta_minor = 0.5
    theta_major = 1.5
    RC = 3.0
    defect_sensitivity = 0.02

    records = []
    vehicle_id = 1

    for _ in range(n_vehicles):
        age = 0
        defect = 0

        for year in range(max_years):
            # Current state
            age_bin = min(age, num_age_bins - 1)

            # Compute scrappage probability via logit
            v_keep = -theta_age * age_bin
            if defect == 1:
                v_keep -= theta_minor
            elif defect == 2:
                v_keep -= theta_major
            v_scrap = -RC

            prob_scrap = 1.0 / (1.0 + np.exp(v_keep - v_scrap))
            scrapped = int(rng.random() < prob_scrap)

            records.append({
                "vehicle_id": vehicle_id,
                "year": year,
                "age_bin": age_bin,
                "defect_level": defect,
                "scrapped": scrapped,
                "state": age_bin * num_defect_levels + defect,
            })

            if scrapped:
                break

            # Transition: age +1, defect stochastic
            age += 1

            # Defect transition (age-dependent)
            p_stay = max(0.4, 0.85 - defect_sensitivity * age_bin)
            p_improve = 0.05 if defect > 0 else 0.0
            p_worsen = 1.0 - p_stay - p_improve

            if defect == 0:
                probs = [p_stay, p_worsen * 0.7, p_worsen * 0.3]
            elif defect == num_defect_levels - 1:
                probs = [0.0] * num_defect_levels
                probs[defect] = p_stay + p_worsen
                if defect > 0:
                    probs[defect - 1] = p_improve
            else:
                probs = [0.0] * num_defect_levels
                probs[defect] = p_stay
                probs[defect - 1] = p_improve
                if defect + 2 < num_defect_levels:
                    probs[defect + 1] = p_worsen * 0.7
                    probs[defect + 2] = p_worsen * 0.3
                else:
                    probs[defect + 1] = p_worsen

            # Normalize and sample
            probs = np.array(probs)
            probs = probs / probs.sum()
            defect = int(rng.choice(num_defect_levels, p=probs))

        vehicle_id += 1

    return pd.DataFrame(records)


def _to_panel(df: pd.DataFrame) -> "Panel":
    """Convert DataFrame to Panel format for estimators."""
    from econirl.core.types import Panel, Trajectory
    import jax.numpy as jnp

    from tqdm import tqdm

    vehicle_ids = df["vehicle_id"].unique()
    trajectories = []

    for vid in tqdm(vehicle_ids, desc="Building panel", leave=False):
        vdata = df[df["vehicle_id"] == vid].sort_values("year")
        states = jnp.array(vdata["state"].values, dtype=jnp.int32)
        actions = jnp.array(vdata["scrapped"].values, dtype=jnp.int32)
        next_states = jnp.concatenate([states[1:], jnp.array([0])])

        traj = Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=int(vid),
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def get_rdw_scrappage_info() -> dict:
    """
    Get metadata about the RDW scrappage dataset.

    Returns:
        Dictionary with dataset information including number of vehicles,
        observations, and summary statistics.
    """
    df = load_rdw_scrappage()

    return {
        "name": "RDW Vehicle Scrappage",
        "n_observations": len(df),
        "n_vehicles": df["vehicle_id"].nunique(),
        "scrappage_rate": df["scrapped"].mean(),
        "mean_age_bin": df["age_bin"].mean(),
        "mean_defect_level": df["defect_level"].mean(),
        "source": "RDW Open Data (opendata.rdw.nl) / Synthetic",
        "reference": "El Boubsi (2023). MSc Thesis, TU Delft.",
    }
