"""Tests for Stanford Drone Dataset."""

import pytest
import pandas as pd


class TestStanfordDrone:
    """Tests for Stanford Drone pedestrian dataset."""

    def test_loads_dataframe(self):
        """Should load as DataFrame."""
        from econirl.datasets import load_stanford_drone

        df = load_stanford_drone()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        """Should have trajectory columns."""
        from econirl.datasets import load_stanford_drone

        df = load_stanford_drone()

        required = ['track_id', 'frame', 'x', 'y', 'agent_type']
        for col in required:
            assert col in df.columns

    def test_agent_types(self):
        """Should have pedestrians and/or cyclists."""
        from econirl.datasets import load_stanford_drone

        df = load_stanford_drone()

        valid_types = ['Pedestrian', 'Biker', 'Skater', 'Cart', 'Car', 'Bus']
        assert df['agent_type'].isin(valid_types).all()

    def test_scene_filter(self):
        """Should filter by scene."""
        from econirl.datasets import load_stanford_drone

        df = load_stanford_drone(scene="gates")

        assert df['scene'].unique().tolist() == ['gates']

    def test_agent_type_filter(self):
        """Should filter by agent type."""
        from econirl.datasets import load_stanford_drone

        df = load_stanford_drone(agent_type="Pedestrian")

        assert df['agent_type'].unique().tolist() == ['Pedestrian']

    def test_as_trajectories(self):
        """Should convert to trajectory format."""
        from econirl.datasets import load_stanford_drone

        trajectories = load_stanford_drone(as_trajectories=True, discretize=True)

        assert isinstance(trajectories, list)
        assert len(trajectories) > 0

    def test_get_info(self):
        """Should return dataset info."""
        from econirl.datasets import get_stanford_drone_info

        info = get_stanford_drone_info()

        assert 'name' in info
        assert 'use_cases' in info
