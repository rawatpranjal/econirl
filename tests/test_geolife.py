"""Tests for GeoLife GPS trajectory dataset."""

import pytest
import pandas as pd


class TestGeoLife:
    """Tests for GeoLife dataset."""

    def test_loads_dataframe(self):
        """Should load as DataFrame."""
        from econirl.datasets import load_geolife

        df = load_geolife()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        """Should have trajectory columns."""
        from econirl.datasets import load_geolife

        df = load_geolife()

        required = ['user_id', 'latitude', 'longitude', 'timestamp']
        for col in required:
            assert col in df.columns

    def test_transportation_mode(self):
        """Should include transportation mode labels."""
        from econirl.datasets import load_geolife

        df = load_geolife(include_labels=True)

        # Labels should be present
        assert 'mode' in df.columns

    def test_n_users_filter(self):
        """Should filter by number of users."""
        from econirl.datasets import load_geolife

        df = load_geolife(n_users=10)

        assert df['user_id'].nunique() == 10

    def test_as_trajectories(self):
        """Should convert to trajectory format."""
        from econirl.datasets import load_geolife

        trajectories = load_geolife(as_trajectories=True, discretize=True)

        assert isinstance(trajectories, list)
        assert len(trajectories) > 0

    def test_get_info(self):
        """Should return dataset info."""
        from econirl.datasets import get_geolife_info

        info = get_geolife_info()

        assert 'name' in info
        assert 'use_cases' in info
