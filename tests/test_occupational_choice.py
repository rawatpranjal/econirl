"""Tests for synthetic occupational choice dataset."""

import pytest
import pandas as pd

from econirl.datasets import load_occupational_choice
from econirl.core.types import Panel


class TestLoadOccupationalChoice:
    """Tests for load_occupational_choice function."""

    def test_basic_loading_returns_dataframe(self) -> None:
        """Basic loading returns a DataFrame."""
        df = load_occupational_choice()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_correct_columns(self) -> None:
        """DataFrame has expected columns."""
        df = load_occupational_choice(n_individuals=10, n_periods=5)
        expected_columns = ["id", "period", "state", "action", "education", "experience", "age"]
        for col in expected_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_as_panel_returns_panel(self) -> None:
        """as_panel=True returns Panel object."""
        panel = load_occupational_choice(n_individuals=10, n_periods=5, as_panel=True)
        assert isinstance(panel, Panel)
        assert panel.num_individuals == 10

    def test_n_individuals_parameter(self) -> None:
        """Different n_individuals values work correctly."""
        for n in [10, 50, 100]:
            df = load_occupational_choice(n_individuals=n, n_periods=5)
            assert df["id"].nunique() == n

    def test_n_periods_parameter(self) -> None:
        """Different n_periods values work correctly."""
        for periods in [10, 20, 40]:
            df = load_occupational_choice(n_individuals=5, n_periods=periods)
            periods_per_individual = df.groupby("id")["period"].count()
            assert (periods_per_individual == periods).all()

    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical data."""
        df1 = load_occupational_choice(n_individuals=20, n_periods=10, seed=42)
        df2 = load_occupational_choice(n_individuals=20, n_periods=10, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self) -> None:
        """Different seeds produce different data."""
        df1 = load_occupational_choice(n_individuals=20, n_periods=10, seed=42)
        df2 = load_occupational_choice(n_individuals=20, n_periods=10, seed=123)
        # Actions should differ (with very high probability)
        assert not df1["action"].equals(df2["action"])


class TestStateSpace:
    """Tests for state space properties."""

    def test_approximately_100_states(self) -> None:
        """State space has approximately 100 discrete states."""
        # Use larger dataset to explore state space
        df = load_occupational_choice(n_individuals=500, n_periods=40)
        n_states = df["state"].nunique()
        # Should have a substantial number of states visited
        # (won't always hit all 100 due to sampling, but should be significant)
        assert n_states >= 20, f"Only {n_states} unique states visited"
        # State values should be in valid range [0, 99]
        assert df["state"].min() >= 0
        assert df["state"].max() < 100

    def test_state_encoding_consistency(self) -> None:
        """State encoding is consistent with education, experience, age."""
        df = load_occupational_choice(n_individuals=50, n_periods=20)
        # Check state = education * 20 + experience * 2 + age
        computed_state = df["education"] * 20 + df["experience"] * 2 + df["age"]
        pd.testing.assert_series_equal(
            df["state"], computed_state, check_names=False
        )


class TestActionSpace:
    """Tests for action space properties."""

    def test_four_actions(self) -> None:
        """There are exactly 4 possible actions."""
        df = load_occupational_choice(n_individuals=200, n_periods=40)
        unique_actions = df["action"].unique()
        # All actions should be in [0, 1, 2, 3]
        assert set(unique_actions).issubset({0, 1, 2, 3})
        # With enough data, all 4 actions should appear
        assert len(unique_actions) == 4, f"Only {len(unique_actions)} actions observed"

    def test_action_values_valid(self) -> None:
        """All action values are in valid range."""
        df = load_occupational_choice(n_individuals=50, n_periods=20)
        assert df["action"].min() >= 0
        assert df["action"].max() <= 3


class TestPanelConversion:
    """Tests for Panel conversion functionality."""

    def test_panel_num_observations(self) -> None:
        """Panel has correct total observations."""
        n_ind = 20
        n_per = 15
        panel = load_occupational_choice(
            n_individuals=n_ind, n_periods=n_per, as_panel=True
        )
        assert panel.num_observations == n_ind * n_per

    def test_panel_trajectory_lengths(self) -> None:
        """Each trajectory has correct length."""
        n_ind = 10
        n_per = 25
        panel = load_occupational_choice(
            n_individuals=n_ind, n_periods=n_per, as_panel=True
        )
        for traj in panel.trajectories:
            assert len(traj) == n_per

    def test_panel_states_actions_match_dataframe(self) -> None:
        """Panel states and actions match DataFrame values."""
        n_ind = 5
        n_per = 10
        seed = 999

        df = load_occupational_choice(
            n_individuals=n_ind, n_periods=n_per, seed=seed
        )
        panel = load_occupational_choice(
            n_individuals=n_ind, n_periods=n_per, seed=seed, as_panel=True
        )

        # Check that states and actions match
        for traj in panel.trajectories:
            ind_df = df[df["id"] == traj.individual_id].sort_values("period")
            assert list(traj.states.numpy()) == list(ind_df["state"].values)
            assert list(traj.actions.numpy()) == list(ind_df["action"].values)


class TestDataQuality:
    """Tests for data quality and realistic patterns."""

    def test_education_range(self) -> None:
        """Education values are in valid range."""
        df = load_occupational_choice(n_individuals=100, n_periods=40)
        assert df["education"].min() >= 0
        assert df["education"].max() <= 4

    def test_experience_range(self) -> None:
        """Experience values are in valid range."""
        df = load_occupational_choice(n_individuals=100, n_periods=40)
        assert df["experience"].min() >= 0
        assert df["experience"].max() <= 9

    def test_age_range(self) -> None:
        """Age group values are in valid range."""
        df = load_occupational_choice(n_individuals=100, n_periods=40)
        assert df["age"].min() >= 0
        assert df["age"].max() <= 1

    def test_period_zero_indexed(self) -> None:
        """Periods are 0-indexed and sequential."""
        df = load_occupational_choice(n_individuals=10, n_periods=20)
        for ind_id in df["id"].unique():
            periods = df[df["id"] == ind_id]["period"].values
            assert list(periods) == list(range(20))


class TestInfoFunction:
    """Tests for get_occupational_choice_info function."""

    def test_info_returns_dict(self) -> None:
        """Info function returns dictionary with expected keys."""
        from econirl.datasets.occupational_choice import get_occupational_choice_info

        info = get_occupational_choice_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "num_states" in info
        assert "num_actions" in info
        assert info["num_states"] == 100
        assert info["num_actions"] == 4
