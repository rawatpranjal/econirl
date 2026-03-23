"""Tests for Robinson Crusoe synthetic dataset."""

import pytest
import pandas as pd
import numpy as np


class TestRobinsonCrusoe:
    """Tests for Robinson Crusoe dataset."""

    def test_loads_dataframe(self):
        """Should load as DataFrame."""
        from econirl.datasets import load_robinson_crusoe

        df = load_robinson_crusoe()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_required_columns(self):
        """Should have required columns."""
        from econirl.datasets import load_robinson_crusoe

        df = load_robinson_crusoe()

        required = ['id', 'period', 'inventory', 'choice']
        for col in required:
            assert col in df.columns

    def test_choice_values(self):
        """Choice should be fish (0) or leisure (1)."""
        from econirl.datasets import load_robinson_crusoe

        df = load_robinson_crusoe()

        assert df['choice'].min() == 0
        assert df['choice'].max() <= 2

    def test_inventory_non_negative(self):
        """Inventory should be non-negative."""
        from econirl.datasets import load_robinson_crusoe

        df = load_robinson_crusoe()

        assert (df['inventory'] >= 0).all()

    def test_include_hunt(self):
        """Should support 3 actions with include_hunt=True."""
        from econirl.datasets import load_robinson_crusoe

        df = load_robinson_crusoe(include_hunt=True)

        # With hunting, should see all 3 choices
        assert df['choice'].max() <= 2

    def test_as_panel(self):
        """Should convert to Panel format."""
        from econirl.datasets import load_robinson_crusoe
        from econirl.core.types import Panel

        panel = load_robinson_crusoe(n_individuals=50, n_periods=20, as_panel=True)

        assert isinstance(panel, Panel)
        assert panel.num_individuals == 50

    def test_get_info(self):
        """Should return dataset info."""
        from econirl.datasets import get_robinson_crusoe_info

        info = get_robinson_crusoe_info()

        assert 'name' in info
        assert 'state_variables' in info
        assert 'choices' in info
