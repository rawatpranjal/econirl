"""State discretization utilities.

Provides functions for converting continuous state variables into discrete bins
suitable for DDC estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Literal, Union


def discretize_state(
    values: Union[np.ndarray, pd.Series],
    method: Literal["uniform", "quantile"] = "uniform",
    n_bins: int = 10,
    clip_to_range: bool = True,
) -> np.ndarray:
    """Discretize continuous state values into bins.

    This function provides transparent preprocessing for converting continuous
    state variables (like mileage, experience, inventory) into discrete bins
    required by DDC estimators.

    Args:
        values: Continuous values to discretize
        method: Binning method
            - "uniform": Equal-width bins (good for evenly distributed data)
            - "quantile": Equal-count bins (good for skewed distributions)
        n_bins: Number of discrete bins (0 to n_bins-1)
        clip_to_range: If True, clip values to [0, n_bins-1]

    Returns:
        Array of bin indices (0-indexed)

    Example:
        >>> from econirl.preprocessing import discretize_state
        >>> mileage = np.array([0, 10000, 25000, 50000, 100000])
        >>> bins = discretize_state(mileage, method="uniform", n_bins=20)
        >>> print(bins)  # [0, 2, 5, 10, 19]
    """
    arr = np.asarray(values)

    if method == "uniform":
        # Equal-width bins
        min_val, max_val = arr.min(), arr.max()
        if min_val == max_val:
            return np.zeros(len(arr), dtype=int)

        # Compute bin edges
        bin_width = (max_val - min_val) / n_bins
        binned = ((arr - min_val) / bin_width).astype(int)

    elif method == "quantile":
        # Equal-count bins using percentiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(arr, percentiles)
        binned = np.digitize(arr, bin_edges[1:-1])  # Exclude first/last

    else:
        raise ValueError(f"Unknown method: {method}. Use 'uniform' or 'quantile'.")

    if clip_to_range:
        binned = np.clip(binned, 0, n_bins - 1)

    return binned.astype(int)


def discretize_mileage(
    mileage: Union[np.ndarray, pd.Series],
    bin_width: float = 5000.0,
    max_bins: int = 90,
) -> np.ndarray:
    """Discretize mileage following Rust (1987) convention.

    Uses 5,000 mile bins as in the original Rust paper.

    Args:
        mileage: Mileage values (can be in miles or thousands of miles)
        bin_width: Width of each bin (default 5000 miles)
        max_bins: Maximum bin index (default 90, i.e., 450,000 miles)

    Returns:
        Array of bin indices (0 to max_bins-1)

    Example:
        >>> mileage = np.array([0, 12500, 250000])
        >>> bins = discretize_mileage(mileage)
        >>> print(bins)  # [0, 2, 50]
    """
    arr = np.asarray(mileage)

    # Auto-detect if values are in thousands (max < 1000)
    if arr.max() < 1000:
        arr = arr * 1000  # Convert to actual miles

    binned = (arr / bin_width).astype(int)
    return np.clip(binned, 0, max_bins - 1)
