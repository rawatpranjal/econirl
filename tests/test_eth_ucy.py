"""Tests for ETH/UCY pedestrian trajectory datasets."""

import pytest
import pandas as pd


class TestETHUCY:
    """Tests for ETH/UCY datasets."""

    def test_loads_dataframe(self):
        """Should load as DataFrame."""
        from econirl.datasets import load_eth_ucy

        df = load_eth_ucy()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_available_scenes(self):
        """Should have expected scenes."""
        from econirl.datasets import load_eth_ucy

        df = load_eth_ucy()

        expected_scenes = ['eth', 'hotel', 'univ', 'zara1', 'zara2']
        assert df['scene'].isin(expected_scenes).all()

    def test_scene_filter(self):
        """Should filter by scene."""
        from econirl.datasets import load_eth_ucy

        df = load_eth_ucy(scene='eth')

        assert df['scene'].unique().tolist() == ['eth']

    def test_required_columns(self):
        """Should have trajectory columns."""
        from econirl.datasets import load_eth_ucy

        df = load_eth_ucy()

        required = ['pedestrian_id', 'frame', 'x', 'y', 'scene']
        for col in required:
            assert col in df.columns

    def test_as_trajectories(self):
        """Should convert to trajectory format."""
        from econirl.datasets import load_eth_ucy

        trajectories = load_eth_ucy(as_trajectories=True, discretize=True)

        assert isinstance(trajectories, list)
        assert len(trajectories) > 0

    def test_get_info(self):
        """Should return dataset info."""
        from econirl.datasets import get_eth_ucy_info

        info = get_eth_ucy_info()

        assert 'name' in info
        assert 'scenes' in info
        assert 'use_cases' in info
