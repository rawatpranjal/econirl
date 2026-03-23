# Rust (1987) Full Replication Package

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a complete academic replication package for Rust (1987) "Optimal Replacement of GMC Bus Engines" using the econirl package.

**Architecture:** Two-phase approach: (1) Acquire and process original Rust data into econirl format, (2) Build comprehensive replication notebook that reproduces key tables and adds Monte Carlo validation.

**Tech Stack:** Python, econirl, pandas, numpy, torch, matplotlib, Jupyter notebooks

---

## Overview

This replication package will:
1. Bundle the original Rust (1987) bus data (Groups 1-4)
2. Reproduce Tables II, IV, V, VI from the paper
3. Demonstrate NFXP, Hotz-Miller, and NPL estimation
4. Include Monte Carlo parameter recovery validation
5. Provide publication-ready output (LaTeX tables, figures)

---

## Task 1: Download and Process Original Rust Data

**Files:**
- Create: `scripts/download_rust_data.py`
- Create: `src/econirl/datasets/rust_bus_original.csv`
- Modify: `src/econirl/datasets/rust_bus.py`

**Step 1: Write the data download script**

```python
#!/usr/bin/env python3
"""Download and process original Rust (1987) bus data.

Sources:
- Kaggle: https://www.kaggle.com/datasets/erichschulman/bus1234.csv
- NFXP Software: http://bfroemel.webhost.uits.arizona.edu/NFXP.zip
"""

import pandas as pd
import numpy as np
from pathlib import Path


def download_rust_data():
    """Download bus data from Kaggle or process from NFXP format."""
    # Kaggle dataset has Groups 1-4 (GMC buses used in paper)
    # Format: bus_id, month, mileage, replacement

    url = "https://raw.githubusercontent.com/erichschulman/bus-engine-replacement/main/bus1234.csv"

    try:
        df = pd.read_csv(url)
        print(f"Downloaded {len(df)} observations")
        return df
    except Exception as e:
        print(f"Failed to download: {e}")
        print("Please download manually from Kaggle")
        return None


def process_rust_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw Rust data into econirl format.

    Original format varies, but typically:
    - Columns: bus_id, period/month, odometer, replacement (0/1)

    Target format:
    - bus_id: int
    - period: int (1-indexed)
    - mileage: float (in thousands)
    - mileage_bin: int (5000-mile bins, 0-89)
    - replaced: int (0/1)
    - group: int (1-4 for paper analysis)
    """
    # Standardize column names
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()

    # Rename to standard format
    rename_map = {
        'busid': 'bus_id',
        'bus': 'bus_id',
        'id': 'bus_id',
        'month': 'period',
        't': 'period',
        'time': 'period',
        'odometer': 'mileage',
        'odo': 'mileage',
        'miles': 'mileage',
        'replace': 'replaced',
        'replacement': 'replaced',
        'd': 'replaced',
        'decision': 'replaced',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure required columns
    required = ['bus_id', 'period', 'mileage', 'replaced']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Convert mileage to thousands if needed (Rust data is in 1000s)
    if df['mileage'].max() < 1000:
        # Already in thousands
        pass
    else:
        df['mileage'] = df['mileage'] / 1000.0

    # Create mileage bins (5000 miles = 5 units in thousands)
    df['mileage_bin'] = (df['mileage'] / 5.0).astype(int).clip(0, 89)

    # Assign groups based on bus_id ranges (from Rust paper)
    # Group 1: buses 4001-4015 (GMC T8H203, 1974)
    # Group 2: buses 4016-4030 (GMC T8H203, 1974)
    # Group 3: buses 4101-4115 (GMC A4523, 1972)
    # Group 4: buses 4116-4130 (GMC A4523, 1972)
    def assign_group(bus_id):
        if bus_id <= 15:
            return 1
        elif bus_id <= 30:
            return 2
        elif bus_id <= 45:
            return 3
        else:
            return 4

    if 'group' not in df.columns:
        df['group'] = df['bus_id'].apply(assign_group)

    # Sort and reset index
    df = df.sort_values(['bus_id', 'period']).reset_index(drop=True)

    return df[['bus_id', 'period', 'mileage', 'mileage_bin', 'replaced', 'group']]


def save_data(df: pd.DataFrame, output_path: Path):
    """Save processed data to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} observations to {output_path}")

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Buses: {df['bus_id'].nunique()}")
    print(f"  Observations: {len(df):,}")
    print(f"  Groups: {sorted(df['group'].unique())}")
    print(f"  Replacement rate: {df['replaced'].mean():.2%}")
    print(f"  Mean mileage bin: {df['mileage_bin'].mean():.1f}")


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "src/econirl/datasets/rust_bus_original.csv"

    df = download_rust_data()
    if df is not None:
        df = process_rust_data(df)
        save_data(df, output_path)
```

**Step 2: Run the download script**

Run: `python scripts/download_rust_data.py`
Expected: Creates `src/econirl/datasets/rust_bus_original.csv`

**Step 3: Update the dataset loader to support original data**

In `src/econirl/datasets/rust_bus.py`, add parameter to load original vs synthetic:

```python
def load_rust_bus(
    group: Optional[int] = None,
    as_panel: bool = False,
    original: bool = False,  # NEW: load original Rust data
) -> pd.DataFrame:
    """
    Load the Rust (1987) bus engine replacement dataset.

    Args:
        group: If specified, load only buses from this group (1-4 for original, 1-8 for synthetic).
        as_panel: If True, return data as Panel object.
        original: If True, load original Rust (1987) data. If False (default), load synthetic data.
    """
    if original:
        data_path = Path(__file__).parent / "rust_bus_original.csv"
        if not data_path.exists():
            raise FileNotFoundError(
                "Original Rust data not found. Run: python scripts/download_rust_data.py"
            )
    else:
        data_path = Path(__file__).parent / "rust_bus_data.csv"
        # ... existing logic for synthetic data
```

**Step 4: Run tests to verify data loads correctly**

Run: `python -c "from econirl.datasets import load_rust_bus; df = load_rust_bus(original=True); print(df.head())"`
Expected: Shows first rows of original Rust data

**Step 5: Commit**

```bash
git add scripts/download_rust_data.py src/econirl/datasets/rust_bus.py src/econirl/datasets/rust_bus_original.csv
git commit -m "feat: add original Rust (1987) bus data

- Add download script for original data
- Support original=True in load_rust_bus()
- Bundle processed original data

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Transition Probability Estimation (First Stage)

**Files:**
- Create: `src/econirl/estimation/transitions.py`
- Create: `tests/test_transition_estimation.py`

**Step 1: Write failing test for transition estimation**

```python
# tests/test_transition_estimation.py
"""Tests for transition probability estimation."""

import pytest
import torch
import numpy as np
from econirl.estimation.transitions import estimate_transition_probs
from econirl.datasets import load_rust_bus


class TestTransitionEstimation:
    """Tests for first-stage transition probability estimation."""

    def test_probs_sum_to_one(self):
        """Transition probabilities must sum to 1."""
        df = load_rust_bus(group=1)
        probs = estimate_transition_probs(df)

        assert np.isclose(probs.sum(), 1.0, atol=1e-6)

    def test_probs_non_negative(self):
        """All probabilities must be non-negative."""
        df = load_rust_bus(group=1)
        probs = estimate_transition_probs(df)

        assert (probs >= 0).all()

    def test_returns_three_probs(self):
        """Should return exactly 3 probabilities (0, 1, 2 bin increments)."""
        df = load_rust_bus(group=1)
        probs = estimate_transition_probs(df)

        assert len(probs) == 3

    def test_matches_rust_estimates(self):
        """Estimates should be close to Rust's Table IV values."""
        # Rust (1987) Table IV estimates for Group 4:
        # θ₀ = 0.3919, θ₁ = 0.5953, θ₂ = 0.0128
        df = load_rust_bus(group=4)
        probs = estimate_transition_probs(df)

        # Allow 10% relative tolerance due to data differences
        assert np.isclose(probs[0], 0.3919, rtol=0.1)
        assert np.isclose(probs[1], 0.5953, rtol=0.1)
        assert np.isclose(probs[2], 0.0128, rtol=0.2)  # Small prob, more tolerance
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_transition_estimation.py -v`
Expected: FAIL with "No module named 'econirl.estimation.transitions'"

**Step 3: Implement transition probability estimation**

```python
# src/econirl/estimation/transitions.py
"""First-stage transition probability estimation.

Estimates the mileage transition probabilities θ = (θ₀, θ₁, θ₂) from data.
These represent P(mileage increases by k bins | keep engine).

Reference:
    Rust (1987), Section 4.1, Table IV
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def estimate_transition_probs(
    df: pd.DataFrame,
    max_increment: int = 2,
    mileage_col: str = "mileage_bin",
    action_col: str = "replaced",
    bus_col: str = "bus_id",
    period_col: str = "period",
) -> np.ndarray:
    """Estimate mileage transition probabilities from panel data.

    Uses observations where the engine was NOT replaced to estimate
    the distribution of mileage increments.

    P(Δx = k) = #{transitions with increment k} / #{total transitions}

    Args:
        df: Panel data with mileage and replacement decisions
        max_increment: Maximum mileage increment to consider (default 2)
        mileage_col: Column name for mileage bin
        action_col: Column name for replacement decision
        bus_col: Column name for bus identifier
        period_col: Column name for time period

    Returns:
        Array of probabilities [P(Δx=0), P(Δx=1), P(Δx=2)]

    Example:
        >>> df = load_rust_bus(group=4)
        >>> probs = estimate_transition_probs(df)
        >>> print(f"θ = ({probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f})")
    """
    # Sort by bus and period
    df = df.sort_values([bus_col, period_col]).copy()

    # Compute mileage increments for non-replacement periods
    increments = []

    for bus_id in df[bus_col].unique():
        bus_data = df[df[bus_col] == bus_id].sort_values(period_col)

        for i in range(len(bus_data) - 1):
            current = bus_data.iloc[i]
            next_obs = bus_data.iloc[i + 1]

            # Only use transitions where engine was NOT replaced
            if current[action_col] == 0:
                # And next period is consecutive
                if next_obs[period_col] == current[period_col] + 1:
                    increment = next_obs[mileage_col] - current[mileage_col]
                    # Clamp to valid range (could be negative due to measurement error)
                    increment = max(0, min(increment, max_increment))
                    increments.append(increment)

    if len(increments) == 0:
        # Return uniform if no valid transitions
        return np.ones(max_increment + 1) / (max_increment + 1)

    # Count increments
    counts = np.zeros(max_increment + 1)
    for inc in increments:
        counts[int(inc)] += 1

    # Normalize to probabilities
    probs = counts / counts.sum()

    return probs


def estimate_transition_probs_by_group(
    df: pd.DataFrame,
    group_col: str = "group",
    **kwargs,
) -> dict[int, np.ndarray]:
    """Estimate transition probabilities separately for each bus group.

    Args:
        df: Panel data with group identifier
        group_col: Column name for group
        **kwargs: Additional arguments passed to estimate_transition_probs

    Returns:
        Dictionary mapping group ID to probability array

    Example:
        >>> df = load_rust_bus()
        >>> probs_by_group = estimate_transition_probs_by_group(df)
        >>> for g, p in probs_by_group.items():
        ...     print(f"Group {g}: θ = {p}")
    """
    result = {}
    for group in sorted(df[group_col].unique()):
        group_df = df[df[group_col] == group]
        result[group] = estimate_transition_probs(group_df, **kwargs)
    return result
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_transition_estimation.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add src/econirl/estimation/transitions.py tests/test_transition_estimation.py
git commit -m "feat: add first-stage transition probability estimation

- estimate_transition_probs() for single group
- estimate_transition_probs_by_group() for all groups
- Matches Rust (1987) Table IV methodology

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Create Table II Replication (Descriptive Statistics)

**Files:**
- Create: `src/econirl/replication/rust1987/tables.py`
- Create: `tests/test_rust_tables.py`

**Step 1: Write failing test for Table II**

```python
# tests/test_rust_tables.py
"""Tests for Rust (1987) table replication."""

import pytest
import pandas as pd
from econirl.replication.rust1987.tables import table_ii_descriptives


class TestTableII:
    """Tests for Table II replication."""

    def test_table_ii_structure(self):
        """Table II should have correct structure."""
        table = table_ii_descriptives()

        # Should have rows for each group
        assert len(table) >= 4

        # Should have required columns
        required_cols = ['n_buses', 'n_replacements', 'mean_mileage', 'std_mileage']
        for col in required_cols:
            assert col in table.columns

    def test_table_ii_values_reasonable(self):
        """Table II values should be in reasonable ranges."""
        table = table_ii_descriptives()

        # Mean mileage at replacement should be positive
        assert (table['mean_mileage'] > 0).all()

        # Should have some replacements
        assert table['n_replacements'].sum() > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rust_tables.py::TestTableII -v`
Expected: FAIL with "No module named 'econirl.replication'"

**Step 3: Implement Table II**

```python
# src/econirl/replication/rust1987/tables.py
"""Rust (1987) table replication functions.

Reproduces key tables from:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

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

    Table II in Rust (1987) reports:
    - Number of buses per group
    - Number of engine replacements
    - Mean and std of odometer at replacement
    - Mean and std of final odometer (right-censored obs)

    Args:
        df: Bus data (if None, loads original Rust data)
        original: If loading data, whether to use original

    Returns:
        DataFrame with descriptive statistics by group

    Example:
        >>> table = table_ii_descriptives()
        >>> print(table.to_string())
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

    Table IV reports maximum likelihood estimates of the mileage
    transition probabilities θ = (θ₀, θ₁, θ₂) for each bus group.

    Args:
        df: Bus data
        original: If loading data, whether to use original

    Returns:
        DataFrame with transition probability estimates by group
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
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rust_tables.py::TestTableII -v`
Expected: PASS

**Step 5: Add `__init__.py` files for new package**

Create `src/econirl/replication/__init__.py`:
```python
"""Replication packages for classic papers."""
```

Create `src/econirl/replication/rust1987/__init__.py`:
```python
"""Rust (1987) replication package.

Reference:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from econirl.replication.rust1987.tables import (
    table_ii_descriptives,
    table_iv_transitions,
)

__all__ = [
    "table_ii_descriptives",
    "table_iv_transitions",
]
```

**Step 6: Commit**

```bash
git add src/econirl/replication/ tests/test_rust_tables.py
git commit -m "feat: add Rust (1987) Table II and IV replication

- table_ii_descriptives() for descriptive stats
- table_iv_transitions() for first-stage estimates
- New replication subpackage structure

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Create Table V Replication (Structural Estimates)

**Files:**
- Modify: `src/econirl/replication/rust1987/tables.py`
- Modify: `tests/test_rust_tables.py`

**Step 1: Write failing test for Table V**

```python
# Add to tests/test_rust_tables.py

class TestTableV:
    """Tests for Table V replication (structural estimates)."""

    def test_table_v_runs(self):
        """Table V estimation should complete without error."""
        from econirl.replication.rust1987.tables import table_v_structural

        # Use small sample for speed
        table = table_v_structural(groups=[4], n_bootstrap=0)

        assert table is not None
        assert 'operating_cost' in table.columns or 'theta_c' in table.columns

    def test_table_v_has_standard_errors(self):
        """Table V should include standard errors."""
        from econirl.replication.rust1987.tables import table_v_structural

        table = table_v_structural(groups=[4], n_bootstrap=0)

        # Should have SE columns
        se_cols = [c for c in table.columns if 'se' in c.lower()]
        assert len(se_cols) > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_rust_tables.py::TestTableV -v`
Expected: FAIL with "cannot import name 'table_v_structural'"

**Step 3: Implement Table V**

```python
# Add to src/econirl/replication/rust1987/tables.py

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.core.types import Panel, Trajectory
import torch


def _df_to_panel(df: pd.DataFrame) -> Panel:
    """Convert DataFrame to Panel format."""
    trajectories = []

    for bus_id in df['bus_id'].unique():
        bus_data = df[df['bus_id'] == bus_id].sort_values('period')

        states = torch.tensor(bus_data['mileage_bin'].values, dtype=torch.long)
        actions = torch.tensor(bus_data['replaced'].values, dtype=torch.long)
        next_states = torch.cat([states[1:], torch.tensor([0])])

        traj = Trajectory(
            states=states,
            actions=actions,
            next_states=next_states,
            individual_id=int(bus_id),
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)


def table_v_structural(
    df: Optional[pd.DataFrame] = None,
    groups: Optional[list[int]] = None,
    estimators: list[str] = ["NFXP", "Hotz-Miller", "NPL"],
    discount_factor: float = 0.9999,
    n_bootstrap: int = 0,
    original: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Replicate Table V: Structural Parameter Estimates.

    Table V reports estimates of the cost parameters:
    - θ_c: Operating cost (cost per mileage unit)
    - RC: Replacement cost

    for different specifications and estimators.

    Args:
        df: Bus data
        groups: Which groups to estimate (default: [4] as in paper)
        estimators: Which estimators to use
        discount_factor: β for dynamic programming
        n_bootstrap: Number of bootstrap replications for SEs
        original: If loading data, whether to use original
        verbose: Print progress

    Returns:
        DataFrame with structural estimates by group and estimator
    """
    if df is None:
        df = load_rust_bus(original=original)

    if groups is None:
        groups = [4]  # Rust focuses on Group 4

    # Get transition probabilities from first stage
    probs_by_group = estimate_transition_probs_by_group(df)

    results = []

    for group in groups:
        group_df = df[df['group'] == group]
        panel = _df_to_panel(group_df)

        # Set up environment with estimated transitions
        trans_probs = tuple(probs_by_group[group])

        env = RustBusEnvironment(
            operating_cost=0.001,  # Initial guess
            replacement_cost=3.0,  # Initial guess
            mileage_transition_probs=trans_probs,
            discount_factor=discount_factor,
        )

        utility = LinearUtility.from_environment(env)
        problem = env.problem_spec
        transitions = env.transition_matrices

        for est_name in estimators:
            if verbose:
                print(f"Estimating Group {group} with {est_name}...")

            if est_name == "NFXP":
                estimator = NFXPEstimator(verbose=False, outer_max_iter=200)
            elif est_name == "Hotz-Miller":
                estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
            elif est_name == "NPL":
                estimator = CCPEstimator(num_policy_iterations=10, verbose=False)
            else:
                raise ValueError(f"Unknown estimator: {est_name}")

            result = estimator.estimate(panel, utility, problem, transitions)

            results.append({
                'group': group,
                'estimator': est_name,
                'theta_c': result.parameters[0].item(),
                'theta_c_se': result.standard_errors[0].item() if result.standard_errors is not None else np.nan,
                'RC': result.parameters[1].item(),
                'RC_se': result.standard_errors[1].item() if result.standard_errors is not None else np.nan,
                'log_likelihood': result.log_likelihood,
                'converged': result.converged,
            })

    return pd.DataFrame(results)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_rust_tables.py::TestTableV -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/econirl/replication/rust1987/tables.py tests/test_rust_tables.py
git commit -m "feat: add Rust (1987) Table V structural estimation

- table_v_structural() replicates cost parameter estimates
- Supports NFXP, Hotz-Miller, and NPL estimators
- Uses first-stage transition estimates

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Create Monte Carlo Validation Module

**Files:**
- Create: `src/econirl/replication/rust1987/monte_carlo.py`
- Create: `tests/test_monte_carlo.py`

**Step 1: Write failing test for Monte Carlo**

```python
# tests/test_monte_carlo.py
"""Tests for Monte Carlo parameter recovery."""

import pytest
from econirl.replication.rust1987.monte_carlo import run_monte_carlo


class TestMonteCarlo:
    """Tests for Monte Carlo validation."""

    def test_monte_carlo_runs(self):
        """Monte Carlo should complete without error."""
        results = run_monte_carlo(
            n_simulations=2,
            n_individuals=50,
            n_periods=20,
            estimators=["NFXP"],
            seed=42,
        )

        assert results is not None
        assert len(results) == 2  # 2 simulations

    def test_monte_carlo_recovers_params(self):
        """Monte Carlo should recover true parameters on average."""
        results = run_monte_carlo(
            n_simulations=5,
            n_individuals=200,
            n_periods=50,
            estimators=["NFXP"],
            seed=42,
        )

        # Mean estimate should be within 50% of true value
        mean_theta_c = results['theta_c'].mean()
        mean_RC = results['RC'].mean()

        assert 0.0005 < mean_theta_c < 0.002  # True: 0.001
        assert 1.5 < mean_RC < 4.5  # True: 3.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_monte_carlo.py -v`
Expected: FAIL with "No module named 'econirl.replication.rust1987.monte_carlo'"

**Step 3: Implement Monte Carlo module**

```python
# src/econirl/replication/rust1987/monte_carlo.py
"""Monte Carlo parameter recovery validation.

Validates estimators by:
1. Simulating data from known parameters
2. Estimating parameters
3. Comparing estimates to truth
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Optional
from tqdm import tqdm

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.simulation.synthetic import simulate_panel


def run_monte_carlo(
    n_simulations: int = 100,
    n_individuals: int = 500,
    n_periods: int = 100,
    true_operating_cost: float = 0.001,
    true_replacement_cost: float = 3.0,
    discount_factor: float = 0.9999,
    estimators: list[str] = ["NFXP", "Hotz-Miller", "NPL"],
    seed: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Run Monte Carlo parameter recovery experiment.

    Args:
        n_simulations: Number of Monte Carlo replications
        n_individuals: Buses per simulation
        n_periods: Time periods per bus
        true_operating_cost: True θ_c parameter
        true_replacement_cost: True RC parameter
        discount_factor: β for dynamic programming
        estimators: Which estimators to test
        seed: Random seed for reproducibility
        verbose: Print progress

    Returns:
        DataFrame with estimates from each simulation

    Example:
        >>> results = run_monte_carlo(n_simulations=100, seed=42)
        >>> print(results.groupby('estimator').agg({
        ...     'theta_c': ['mean', 'std'],
        ...     'RC': ['mean', 'std'],
        ... }))
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # True environment
    env = RustBusEnvironment(
        operating_cost=true_operating_cost,
        replacement_cost=true_replacement_cost,
        discount_factor=discount_factor,
    )

    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    results = []
    iterator = range(n_simulations)
    if verbose:
        iterator = tqdm(iterator, desc="Monte Carlo")

    for sim in iterator:
        sim_seed = seed + sim if seed is not None else None

        # Simulate data
        panel = simulate_panel(
            env,
            n_individuals=n_individuals,
            n_periods=n_periods,
            seed=sim_seed,
        )

        for est_name in estimators:
            if est_name == "NFXP":
                estimator = NFXPEstimator(verbose=False, outer_max_iter=200)
            elif est_name == "Hotz-Miller":
                estimator = CCPEstimator(num_policy_iterations=1, verbose=False)
            elif est_name == "NPL":
                estimator = CCPEstimator(num_policy_iterations=10, verbose=False)
            else:
                raise ValueError(f"Unknown estimator: {est_name}")

            try:
                result = estimator.estimate(panel, utility, problem, transitions)

                results.append({
                    'simulation': sim,
                    'estimator': est_name,
                    'theta_c': result.parameters[0].item(),
                    'theta_c_se': result.standard_errors[0].item() if result.standard_errors is not None else np.nan,
                    'RC': result.parameters[1].item(),
                    'RC_se': result.standard_errors[1].item() if result.standard_errors is not None else np.nan,
                    'log_likelihood': result.log_likelihood,
                    'converged': result.converged,
                    'true_theta_c': true_operating_cost,
                    'true_RC': true_replacement_cost,
                })
            except Exception as e:
                if verbose:
                    print(f"Sim {sim}, {est_name} failed: {e}")
                results.append({
                    'simulation': sim,
                    'estimator': est_name,
                    'theta_c': np.nan,
                    'RC': np.nan,
                    'converged': False,
                    'true_theta_c': true_operating_cost,
                    'true_RC': true_replacement_cost,
                })

    return pd.DataFrame(results)


def summarize_monte_carlo(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize Monte Carlo results.

    Computes bias, RMSE, and coverage for each estimator.

    Args:
        results: Output from run_monte_carlo()

    Returns:
        Summary statistics by estimator
    """
    summary = []

    for est_name in results['estimator'].unique():
        est_results = results[results['estimator'] == est_name]

        # Filter converged results
        converged = est_results[est_results['converged'] == True]

        for param in ['theta_c', 'RC']:
            true_val = converged[f'true_{param}'].iloc[0]
            estimates = converged[param].dropna()

            if len(estimates) == 0:
                continue

            bias = estimates.mean() - true_val
            rmse = np.sqrt(((estimates - true_val) ** 2).mean())
            std = estimates.std()

            # Coverage: fraction of 95% CIs containing true value
            if f'{param}_se' in converged.columns:
                ses = converged[f'{param}_se'].dropna()
                if len(ses) == len(estimates):
                    lower = estimates - 1.96 * ses
                    upper = estimates + 1.96 * ses
                    coverage = ((lower <= true_val) & (true_val <= upper)).mean()
                else:
                    coverage = np.nan
            else:
                coverage = np.nan

            summary.append({
                'estimator': est_name,
                'parameter': param,
                'true_value': true_val,
                'mean_estimate': estimates.mean(),
                'bias': bias,
                'std': std,
                'rmse': rmse,
                'coverage_95': coverage,
                'n_converged': len(estimates),
                'n_total': len(est_results),
            })

    return pd.DataFrame(summary)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_monte_carlo.py -v`
Expected: PASS

**Step 5: Update `__init__.py`**

Add to `src/econirl/replication/rust1987/__init__.py`:
```python
from econirl.replication.rust1987.monte_carlo import (
    run_monte_carlo,
    summarize_monte_carlo,
)
```

**Step 6: Commit**

```bash
git add src/econirl/replication/rust1987/monte_carlo.py tests/test_monte_carlo.py src/econirl/replication/rust1987/__init__.py
git commit -m "feat: add Monte Carlo parameter recovery validation

- run_monte_carlo() for simulation study
- summarize_monte_carlo() for bias/RMSE/coverage
- Tests verify parameter recovery

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Create Replication Jupyter Notebook

**Files:**
- Create: `examples/rust_1987_replication.ipynb`

**Step 1: Create notebook structure**

The notebook will have these sections:
1. Introduction & Setup
2. Data Overview (Table II)
3. First-Stage Estimation (Table IV)
4. Structural Estimation (Table V)
5. Model Comparison (NFXP vs CCP vs NPL)
6. Monte Carlo Validation
7. Counterfactual Analysis
8. Conclusions

**Step 2: Write the notebook**

Create `examples/rust_1987_replication.ipynb` with cells implementing each section. Key cells:

```python
# Cell 1: Introduction
"""
# Rust (1987) Replication Package

This notebook replicates key results from:

> Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical Model
> of Harold Zurcher." *Econometrica*, 55(5), 999-1033.

Using the `econirl` package for dynamic discrete choice estimation.
"""

# Cell 2: Setup
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from econirl.datasets import load_rust_bus
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.replication.rust1987 import (
    table_ii_descriptives,
    table_iv_transitions,
    table_v_structural,
    run_monte_carlo,
    summarize_monte_carlo,
)

# Cell 3: Load Data
df = load_rust_bus(original=True)
print(f"Loaded {len(df):,} observations from {df['bus_id'].nunique()} buses")

# Cell 4: Table II
table_ii = table_ii_descriptives(df)
print("Table II: Descriptive Statistics")
print(table_ii.to_string())

# Cell 5: Table IV
table_iv = table_iv_transitions(df)
print("Table IV: Transition Probability Estimates")
print(table_iv.to_string())

# Cell 6: Table V
table_v = table_v_structural(df, groups=[1, 2, 3, 4], verbose=True)
print("Table V: Structural Parameter Estimates")
print(table_v.to_string())

# Cell 7: Monte Carlo
mc_results = run_monte_carlo(n_simulations=100, seed=1987, verbose=True)
mc_summary = summarize_monte_carlo(mc_results)
print("Monte Carlo Summary")
print(mc_summary.to_string())

# Cell 8: Visualization
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# ... plotting code
```

**Step 3: Run notebook to verify it executes**

Run: `jupyter nbconvert --execute examples/rust_1987_replication.ipynb --to html`
Expected: Notebook executes successfully, produces HTML output

**Step 4: Commit**

```bash
git add examples/rust_1987_replication.ipynb
git commit -m "feat: add Rust (1987) replication notebook

- Complete walkthrough of paper replication
- Tables II, IV, V reproduced
- Monte Carlo validation
- Publication-ready figures

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Add LaTeX Export for Tables

**Files:**
- Create: `src/econirl/replication/rust1987/export.py`

**Step 1: Write failing test**

```python
# Add to tests/test_rust_tables.py

class TestExport:
    """Tests for LaTeX export."""

    def test_table_ii_latex(self):
        """Table II should export to LaTeX."""
        from econirl.replication.rust1987.export import table_to_latex

        table = table_ii_descriptives()
        latex = table_to_latex(table, caption="Table II: Descriptive Statistics")

        assert "\\begin{table}" in latex
        assert "Descriptive Statistics" in latex
```

**Step 2: Implement LaTeX export**

```python
# src/econirl/replication/rust1987/export.py
"""Export functions for publication-ready output."""

from __future__ import annotations

import pandas as pd
from typing import Optional


def table_to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
) -> str:
    """Convert DataFrame to LaTeX table.

    Args:
        df: Table to convert
        caption: Table caption
        label: LaTeX label for referencing
        float_format: Format string for floats

    Returns:
        LaTeX table string
    """
    latex = df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
    )
    return latex


def save_all_tables(
    output_dir: str = "output/tables",
    original: bool = True,
):
    """Generate and save all replication tables.

    Args:
        output_dir: Directory for output files
        original: Use original Rust data
    """
    from pathlib import Path
    from econirl.replication.rust1987.tables import (
        table_ii_descriptives,
        table_iv_transitions,
        table_v_structural,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Table II
    t2 = table_ii_descriptives(original=original)
    t2.to_csv(output_path / "table_ii.csv")
    with open(output_path / "table_ii.tex", "w") as f:
        f.write(table_to_latex(t2, caption="Table II: Descriptive Statistics", label="tab:table_ii"))

    # Table IV
    t4 = table_iv_transitions(original=original)
    t4.to_csv(output_path / "table_iv.csv")
    with open(output_path / "table_iv.tex", "w") as f:
        f.write(table_to_latex(t4, caption="Table IV: Transition Probabilities", label="tab:table_iv"))

    # Table V
    t5 = table_v_structural(original=original, verbose=True)
    t5.to_csv(output_path / "table_v.csv")
    with open(output_path / "table_v.tex", "w") as f:
        f.write(table_to_latex(t5, caption="Table V: Structural Estimates", label="tab:table_v"))

    print(f"Tables saved to {output_path}")
```

**Step 3: Run test, then commit**

```bash
git add src/econirl/replication/rust1987/export.py tests/test_rust_tables.py
git commit -m "feat: add LaTeX export for replication tables

- table_to_latex() for individual tables
- save_all_tables() for batch export
- CSV and LaTeX output

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Add Integration Tests

**Files:**
- Create: `tests/integration/test_rust_replication.py`

**Step 1: Write integration test**

```python
# tests/integration/test_rust_replication.py
"""Integration tests for Rust (1987) replication package."""

import pytest
import pandas as pd


class TestRustReplicationIntegration:
    """End-to-end tests for replication package."""

    @pytest.mark.slow
    def test_full_replication_pipeline(self):
        """Test complete replication from data to tables."""
        from econirl.datasets import load_rust_bus
        from econirl.replication.rust1987 import (
            table_ii_descriptives,
            table_iv_transitions,
            table_v_structural,
        )

        # Load data
        df = load_rust_bus(original=False)  # Use synthetic for CI
        assert len(df) > 0

        # Table II
        t2 = table_ii_descriptives(df, original=False)
        assert len(t2) > 0

        # Table IV
        t4 = table_iv_transitions(df, original=False)
        assert len(t4) > 0
        assert (t4[['theta_0', 'theta_1', 'theta_2']].sum(axis=1) - 1.0).abs().max() < 0.01

        # Table V (single group for speed)
        t5 = table_v_structural(df, groups=[1], estimators=["NFXP"], original=False)
        assert len(t5) > 0
        assert t5['converged'].all()

    @pytest.mark.slow
    def test_monte_carlo_smoke(self):
        """Smoke test for Monte Carlo."""
        from econirl.replication.rust1987 import run_monte_carlo

        results = run_monte_carlo(
            n_simulations=3,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        assert len(results) == 3
        assert results['converged'].all()
```

**Step 2: Run integration test**

Run: `pytest tests/integration/test_rust_replication.py -v -m slow`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_rust_replication.py
git commit -m "test: add integration tests for Rust replication

- Full pipeline test
- Monte Carlo smoke test
- Marked as slow for CI

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 9: Update Package Exports and Documentation

**Files:**
- Modify: `src/econirl/__init__.py`
- Create: `docs/tutorials/rust_1987_replication.rst`

**Step 1: Update main package exports**

Add to `src/econirl/__init__.py`:
```python
from econirl import replication
```

**Step 2: Create documentation page**

```rst
.. _rust-1987-replication:

Rust (1987) Replication
=======================

This tutorial demonstrates how to replicate the results from Rust's seminal
1987 paper on bus engine replacement using econirl.

.. contents:: Table of Contents
   :local:

Overview
--------

Rust (1987) introduced the Nested Fixed Point (NFXP) algorithm for estimating
dynamic discrete choice models...

Quick Start
-----------

.. code-block:: python

    from econirl.datasets import load_rust_bus
    from econirl.replication.rust1987 import table_v_structural

    # Load original Rust data
    df = load_rust_bus(original=True)

    # Replicate Table V
    table = table_v_structural(df, groups=[4])
    print(table)

...
```

**Step 3: Commit**

```bash
git add src/econirl/__init__.py docs/tutorials/rust_1987_replication.rst
git commit -m "docs: add Rust (1987) replication tutorial

- Export replication module from main package
- Sphinx documentation page
- Quick start examples

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Final Testing and README Update

**Files:**
- Modify: `README.md`

**Step 1: Run full test suite**

Run: `pytest tests/ -v --cov=econirl`
Expected: All tests pass, coverage > 80%

**Step 2: Update README with replication example**

Add to `README.md`:

```markdown
## Replication Packages

### Rust (1987) - Bus Engine Replacement

Replicate the classic dynamic discrete choice paper:

```python
from econirl.replication.rust1987 import table_v_structural

# Reproduce Table V structural estimates
table = table_v_structural(groups=[4])
print(table)
```

See the [full replication notebook](examples/rust_1987_replication.ipynb) for details.
```

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add Rust (1987) replication to README

- Quick example of replication package
- Link to full notebook

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

This plan creates a complete Rust (1987) replication package with:

| Component | Description |
|-----------|-------------|
| Original data | Download and process actual Rust bus data |
| Table II | Descriptive statistics on odometer readings |
| Table IV | First-stage transition probability estimates |
| Table V | Structural cost parameter estimates |
| Monte Carlo | Parameter recovery validation |
| Notebook | Interactive walkthrough of all results |
| LaTeX export | Publication-ready tables |
| Documentation | Tutorial and API docs |

**Total: 10 tasks, ~40 steps**
