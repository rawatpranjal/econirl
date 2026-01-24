"""Tests for Rust (1987) table replication."""

import pytest
import numpy as np
import pandas as pd
from econirl.replication.rust1987.tables import table_ii_descriptives, table_iv_transitions


class TestTableII:
    """Tests for Table II replication."""

    def test_table_ii_structure(self):
        """Table II should have correct structure."""
        table = table_ii_descriptives()

        # Should have rows for each group
        assert len(table) >= 4

        # Should have required columns
        required_cols = ['n_buses', 'n_replacements', 'mean_mileage', 'std_mileage']
        for col in required_cols:
            assert col in table.columns

    def test_table_ii_values_reasonable(self):
        """Table II values should be in reasonable ranges."""
        table = table_ii_descriptives()

        # Mean mileage at replacement should be positive (where defined)
        # Some groups may have no replacements, resulting in NaN
        valid_mileage = table['mean_mileage'].dropna()
        if len(valid_mileage) > 0:
            assert (valid_mileage > 0).all()

        # Should have some replacements across all groups
        assert table['n_replacements'].sum() > 0

    def test_table_ii_with_synthetic_data(self):
        """Table II should work with synthetic data."""
        table = table_ii_descriptives(original=False)

        # Synthetic data should have all 8 groups
        assert len(table) == 8

        # All groups should have positive mileage values
        assert (table['mean_mileage'] > 0).all()

        # Number of buses should be positive
        assert (table['n_buses'] > 0).all()


class TestTableIV:
    """Tests for Table IV replication."""

    def test_table_iv_structure(self):
        """Table IV should have correct structure."""
        table = table_iv_transitions(original=False)

        # Should have required columns
        required_cols = ['theta_0', 'theta_1', 'theta_2', 'n_transitions']
        for col in required_cols:
            assert col in table.columns

    def test_table_iv_probabilities_valid(self):
        """Transition probabilities should be valid."""
        table = table_iv_transitions(original=False)

        # Probabilities should be non-negative
        for col in ['theta_0', 'theta_1', 'theta_2']:
            assert (table[col] >= 0).all()

        # Probabilities should sum to 1 for each group
        prob_sums = table['theta_0'] + table['theta_1'] + table['theta_2']
        assert np.allclose(prob_sums, 1.0, atol=1e-6)

    def test_table_iv_n_transitions_positive(self):
        """Number of transitions should be positive."""
        table = table_iv_transitions(original=False)

        assert (table['n_transitions'] > 0).all()
