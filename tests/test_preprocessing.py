"""Tests for preprocessing utilities."""

import pytest
import numpy as np
import pandas as pd
from econirl.preprocessing import discretize_state


class TestDiscretizeState:
    """Tests for state discretization."""

    def test_uniform_binning(self):
        """Uniform binning should create equal-width bins."""
        values = np.array([0, 25, 50, 75, 100])
        binned = discretize_state(values, method="uniform", n_bins=4)

        assert binned.min() == 0
        assert binned.max() == 3
        assert len(np.unique(binned)) == 4

    def test_quantile_binning(self):
        """Quantile binning should create equal-count bins."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        binned = discretize_state(values, method="quantile", n_bins=5)

        # Each bin should have ~2 observations
        counts = np.bincount(binned)
        assert all(c == 2 for c in counts)

    def test_preserves_order(self):
        """Discretization should preserve relative ordering."""
        values = np.array([10, 30, 20, 50, 40])
        binned = discretize_state(values, method="uniform", n_bins=5)

        # Larger values should have larger or equal bin indices
        assert binned[3] >= binned[0]  # 50 >= 10
        assert binned[4] >= binned[2]  # 40 >= 20

    def test_handles_series(self):
        """Should work with pandas Series."""
        s = pd.Series([0, 50, 100])
        binned = discretize_state(s, method="uniform", n_bins=2)

        assert isinstance(binned, np.ndarray)
        assert len(binned) == 3


class TestDiscretizeMileage:
    """Tests for Rust-style mileage discretization."""

    def test_basic_discretization(self):
        """Should discretize mileage in 5000-mile bins."""
        from econirl.preprocessing import discretize_mileage

        mileage = np.array([0, 5000, 10000, 25000])
        binned = discretize_mileage(mileage)

        assert binned[0] == 0
        assert binned[1] == 1
        assert binned[2] == 2
        assert binned[3] == 5

    def test_thousands_autodetect(self):
        """Should autodetect if values are in thousands."""
        from econirl.preprocessing import discretize_mileage

        mileage_thousands = np.array([0, 5, 10, 25])  # In thousands
        binned = discretize_mileage(mileage_thousands)

        assert binned[0] == 0
        assert binned[1] == 1
        assert binned[2] == 2


class TestCheckPanelStructure:
    """Tests for panel validation."""

    def test_valid_panel_passes(self):
        """Valid panel should pass all checks."""
        from econirl.preprocessing import check_panel_structure

        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2, 2],
            'period': [1, 2, 3, 1, 2, 3],
            'state': [0, 1, 2, 0, 1, 1],
            'action': [0, 0, 1, 0, 0, 1],
        })

        result = check_panel_structure(df, id_col='id', period_col='period')

        assert result['valid']
        assert len(result['warnings']) == 0
        assert len(result['errors']) == 0

    def test_missing_values_detected(self):
        """Should detect missing values."""
        from econirl.preprocessing import check_panel_structure

        df = pd.DataFrame({
            'id': [1, 1, 2],
            'period': [1, 2, 1],
            'state': [0, np.nan, 1],
            'action': [0, 0, 1],
        })

        # Must specify state_col to check for missing values in state
        result = check_panel_structure(df, id_col='id', period_col='period', state_col='state')

        assert not result['valid']
        assert any('missing' in e.lower() for e in result['errors'])

    def test_period_gaps_detected(self):
        """Should detect gaps in periods."""
        from econirl.preprocessing import check_panel_structure

        df = pd.DataFrame({
            'id': [1, 1, 1],
            'period': [1, 2, 5],  # Gap at 3, 4
            'state': [0, 1, 2],
            'action': [0, 0, 1],
        })

        result = check_panel_structure(df, id_col='id', period_col='period')

        assert any('gap' in w.lower() for w in result['warnings'])

    def test_unbalanced_panel_warning(self):
        """Should warn about unbalanced panels."""
        from econirl.preprocessing import check_panel_structure

        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2],  # id=2 has fewer periods
            'period': [1, 2, 3, 1, 2],
            'state': [0, 1, 2, 0, 1],
            'action': [0, 0, 1, 0, 1],
        })

        result = check_panel_structure(df, id_col='id', period_col='period')

        assert any('unbalanced' in w.lower() for w in result['warnings'])


class TestComputeNextStates:
    """Tests for next state computation."""

    def test_basic_next_states(self):
        """Should compute next states correctly."""
        from econirl.preprocessing import compute_next_states

        df = pd.DataFrame({
            'id': [1, 1, 1, 2, 2],
            'period': [1, 2, 3, 1, 2],
            'state': [0, 5, 10, 0, 3],
        })

        next_states = compute_next_states(df, id_col='id', period_col='period', state_col='state')

        # For id=1: next states should be [5, 10, 10] (last filled with same)
        # For id=2: next states should be [3, 3] (last filled with same)
        assert next_states[0] == 5  # 0 -> 5
        assert next_states[1] == 10  # 5 -> 10
        assert next_states[2] == 10  # 10 -> 10 (absorbing)
        assert next_states[3] == 3  # 0 -> 3
        assert next_states[4] == 3  # 3 -> 3 (absorbing)
