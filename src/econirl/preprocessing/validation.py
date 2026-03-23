"""Panel data validation utilities.

Provides functions for validating panel data structure before estimation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PanelValidationResult:
    """Result of panel structure validation."""

    valid: bool
    n_individuals: int
    n_observations: int
    n_periods_per_individual: dict[int, int]
    is_balanced: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __getitem__(self, key):
        """Allow dict-like access for backwards compatibility."""
        return getattr(self, key)


def check_panel_structure(
    df: pd.DataFrame,
    id_col: str = "id",
    period_col: str = "period",
    state_col: Optional[str] = None,
    action_col: Optional[str] = None,
    require_balanced: bool = False,
    require_consecutive: bool = False,
) -> PanelValidationResult:
    """Validate panel data structure for DDC estimation.

    Checks for common data issues that can cause estimation problems:
    - Missing values in key columns
    - Gaps in period sequences
    - Unbalanced panels
    - State/action value issues

    Args:
        df: Panel data
        id_col: Column name for individual identifier
        period_col: Column name for time period
        state_col: Optional column name for state (if provided, checks for missing)
        action_col: Optional column name for action (if provided, checks for missing)
        require_balanced: If True, treat unbalanced panel as error
        require_consecutive: If True, treat period gaps as error

    Returns:
        PanelValidationResult with validation details

    Example:
        >>> from econirl.preprocessing import check_panel_structure
        >>> result = check_panel_structure(df, id_col='bus_id', period_col='period')
        >>> if not result.valid:
        ...     print("Errors:", result.errors)
    """
    errors = []
    warnings = []

    # Check required columns exist
    for col in [id_col, period_col]:
        if col not in df.columns:
            errors.append(f"Required column '{col}' not found")

    if errors:
        return PanelValidationResult(
            valid=False,
            n_individuals=0,
            n_observations=len(df),
            n_periods_per_individual={},
            is_balanced=False,
            errors=errors,
            warnings=warnings,
        )

    # Check for missing values
    for col in [id_col, period_col, state_col, action_col]:
        if col and col in df.columns:
            n_missing = df[col].isna().sum()
            if n_missing > 0:
                errors.append(f"Missing values in '{col}': {n_missing} observations")

    # Compute panel structure
    n_individuals = df[id_col].nunique()
    n_observations = len(df)

    # Periods per individual
    periods_per_id = df.groupby(id_col)[period_col].count().to_dict()

    # Check for balance
    unique_counts = set(periods_per_id.values())
    is_balanced = len(unique_counts) == 1

    if not is_balanced:
        min_periods = min(periods_per_id.values())
        max_periods = max(periods_per_id.values())
        msg = f"Unbalanced panel: {min_periods}-{max_periods} periods per individual"
        if require_balanced:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Check for period gaps (Markov property validation)
    for ind_id in df[id_col].unique():
        ind_data = df[df[id_col] == ind_id].sort_values(period_col)
        periods = ind_data[period_col].values

        if len(periods) > 1:
            diffs = np.diff(periods)
            if not np.all(diffs == 1):
                gap_locations = np.where(diffs != 1)[0]
                msg = f"Individual {ind_id}: period gaps at indices {gap_locations.tolist()}"
                if require_consecutive:
                    errors.append(msg)
                else:
                    warnings.append(msg)

    # Check state/action columns if provided
    if state_col and state_col in df.columns:
        if df[state_col].dtype not in [np.int32, np.int64, int]:
            warnings.append(f"State column '{state_col}' is not integer type")
        if df[state_col].min() < 0:
            warnings.append(f"State column '{state_col}' has negative values")

    if action_col and action_col in df.columns:
        if df[action_col].dtype not in [np.int32, np.int64, int]:
            warnings.append(f"Action column '{action_col}' is not integer type")
        unique_actions = df[action_col].nunique()
        if unique_actions < 2:
            warnings.append(f"Only {unique_actions} unique action(s) found")

    return PanelValidationResult(
        valid=len(errors) == 0,
        n_individuals=n_individuals,
        n_observations=n_observations,
        n_periods_per_individual=periods_per_id,
        is_balanced=is_balanced,
        errors=errors,
        warnings=warnings,
    )


def compute_next_states(
    df: pd.DataFrame,
    id_col: str = "id",
    period_col: str = "period",
    state_col: str = "state",
    fill_last: str = "same",
) -> np.ndarray:
    """Compute next_state column from state transitions.

    For each observation, computes the state in the next period for the same
    individual. Handles the last period for each individual according to
    fill_last parameter.

    Args:
        df: Panel data sorted by id and period
        id_col: Column name for individual identifier
        period_col: Column name for time period
        state_col: Column name for state
        fill_last: How to handle last period per individual
            - "same": Use same state (absorbing)
            - "zero": Use state 0 (reset)
            - "nan": Leave as NaN

    Returns:
        Array of next_state values

    Example:
        >>> df['next_state'] = compute_next_states(df, id_col='bus_id')
    """
    df = df.sort_values([id_col, period_col]).copy()

    # Shift state within each individual
    next_states = df.groupby(id_col)[state_col].shift(-1)

    # Handle last period
    if fill_last == "same":
        mask = next_states.isna()
        next_states.loc[mask] = df.loc[mask, state_col]
    elif fill_last == "zero":
        next_states = next_states.fillna(0)
    # "nan" leaves as-is

    return next_states.astype(int if fill_last != "nan" else float).values
