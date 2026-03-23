"""Integration tests for dataset estimation pipelines."""

import pytest
import torch


class TestDatasetEstimationPipelines:
    """Test full estimation pipeline with each dataset."""

    @pytest.mark.slow
    def test_rust_bus_pipeline(self):
        """Full pipeline with Rust bus data."""
        from econirl.datasets import load_rust_bus
        from econirl.environments.rust_bus import RustBusEnvironment
        from econirl.estimation.ccp import CCPEstimator
        from econirl.preferences.linear import LinearUtility

        panel = load_rust_bus(group=1, as_panel=True)
        env = RustBusEnvironment()
        utility = LinearUtility.from_environment(env)

        est = CCPEstimator(num_policy_iterations=1)
        result = est.estimate(panel, utility, env.problem_spec, env.transition_matrices)

        assert result.converged
        assert len(result.parameters) == 2

    @pytest.mark.slow
    def test_keane_wolpin_data_validation(self):
        """Test Keane-Wolpin data loads and validates."""
        from econirl.datasets import load_keane_wolpin
        from econirl.preprocessing import check_panel_structure

        df = load_keane_wolpin()

        result = check_panel_structure(df, id_col='id', period_col='period')
        assert result.valid
        assert result.n_individuals > 0

    @pytest.mark.slow
    def test_robinson_crusoe_panel_conversion(self):
        """Test Robinson Crusoe panel conversion."""
        from econirl.datasets import load_robinson_crusoe
        from econirl.core.types import Panel

        panel = load_robinson_crusoe(n_individuals=50, n_periods=20, as_panel=True)

        assert isinstance(panel, Panel)
        assert panel.num_individuals == 50
        assert panel.num_observations == 50 * 20

    def test_tdrive_discretization(self):
        """Test T-Drive GPS discretization."""
        from econirl.datasets import load_tdrive
        import numpy as np

        df = load_tdrive(n_taxis=10, discretize=True, grid_size=50)

        assert 'state' in df.columns
        assert df['state'].min() >= 0
        assert df['state'].max() < 50 * 50

    def test_geolife_trajectory_format(self):
        """Test GeoLife trajectory conversion."""
        from econirl.datasets import load_geolife

        trajectories = load_geolife(n_users=5, as_trajectories=True, discretize=True)

        assert isinstance(trajectories, list)
        assert len(trajectories) > 0
        # Each trajectory should be a numpy array of states
        for traj in trajectories:
            assert hasattr(traj, '__len__')
            assert len(traj) > 0

    def test_stanford_drone_scene_filtering(self):
        """Test Stanford Drone scene and agent type filtering."""
        from econirl.datasets import load_stanford_drone

        # Filter by scene and agent type
        df = load_stanford_drone(scene="gates", agent_type="Pedestrian")

        assert (df['scene'] == 'gates').all()
        assert (df['agent_type'] == 'Pedestrian').all()

    def test_eth_ucy_world_coordinates(self):
        """Test ETH/UCY world coordinate system."""
        from econirl.datasets import load_eth_ucy

        df = load_eth_ucy(scene="eth")

        # Coordinates should be in meters (reasonable range)
        assert df['x'].min() >= -100
        assert df['x'].max() <= 100
        assert df['y'].min() >= -100
        assert df['y'].max() <= 100
