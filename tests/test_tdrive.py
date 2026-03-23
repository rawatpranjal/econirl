"""Tests for T-Drive taxi trajectory dataset."""

import pytest
import pandas as pd
import numpy as np


class TestTDrive:
    """Tests for T-Drive dataset."""

    def test_loads_dataframe(self):
        """Should load as DataFrame."""
        from econirl.datasets import load_tdrive

        df = load_tdrive()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        """Should have trajectory columns."""
        from econirl.datasets import load_tdrive

        df = load_tdrive()

        required = ['taxi_id', 'timestamp', 'longitude', 'latitude']
        for col in required:
            assert col in df.columns

    def test_as_trajectories(self):
        """Should convert to trajectory format."""
        from econirl.datasets import load_tdrive

        trajectories = load_tdrive(as_trajectories=True)

        assert isinstance(trajectories, list)
        assert len(trajectories) > 0
        # Each trajectory is list of (state, action) or states
        assert all(len(t) > 0 for t in trajectories)

    def test_discretized_states(self):
        """Should support grid discretization."""
        from econirl.datasets import load_tdrive

        df = load_tdrive(discretize=True, grid_size=100)

        assert 'state' in df.columns
        assert df['state'].dtype in [np.int32, np.int64, int]

    def test_n_taxis_filter(self):
        """Should filter by number of taxis."""
        from econirl.datasets import load_tdrive

        df = load_tdrive(n_taxis=10)

        assert df['taxi_id'].nunique() == 10

    def test_get_info(self):
        """Should return dataset info."""
        from econirl.datasets import get_tdrive_info

        info = get_tdrive_info()

        assert 'name' in info
        assert 'use_cases' in info
