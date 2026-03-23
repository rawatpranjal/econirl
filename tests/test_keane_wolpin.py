"""Tests for Keane & Wolpin (1994) dataset loader."""

import pytest
import pandas as pd
import numpy as np


class TestLoadKeaneWolpin:
    """Tests for Keane & Wolpin dataset loading."""

    def test_loads_dataframe(self):
        """Should load as DataFrame by default."""
        from econirl.datasets import load_keane_wolpin

        df = load_keane_wolpin()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        """Should have required columns for DDC."""
        from econirl.datasets import load_keane_wolpin

        df = load_keane_wolpin()

        required = ['id', 'period', 'choice']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_state_columns(self):
        """Should have state variables."""
        from econirl.datasets import load_keane_wolpin

        df = load_keane_wolpin()

        # KW94 uses experience and schooling as states
        state_cols = ['schooling', 'exp_white_collar', 'exp_blue_collar']
        for col in state_cols:
            assert col in df.columns, f"Missing state: {col}"

    def test_choice_values(self):
        """Choice should be 0-indexed integers."""
        from econirl.datasets import load_keane_wolpin

        df = load_keane_wolpin()

        assert df['choice'].min() >= 0
        assert df['choice'].max() <= 4  # 5 choices in KW94
        assert df['choice'].dtype in [np.int32, np.int64, int]

    def test_as_panel(self):
        """Should convert to Panel format."""
        from econirl.datasets import load_keane_wolpin
        from econirl.core.types import Panel

        panel = load_keane_wolpin(as_panel=True)

        assert isinstance(panel, Panel)
        assert panel.num_individuals > 0

    def test_bundled_fallback(self):
        """Should use bundled data if respy not installed."""
        from econirl.datasets import load_keane_wolpin

        # This should work regardless of respy installation
        df = load_keane_wolpin(source="bundled")

        assert len(df) > 0

    def test_get_info(self):
        """Should return dataset info."""
        from econirl.datasets import get_keane_wolpin_info

        info = get_keane_wolpin_info()

        assert 'name' in info
        assert 'n_observations' in info
        assert 'n_individuals' in info
