"""Integration tests for Rust (1987) replication package."""

import pytest
import pandas as pd


class TestRustReplicationIntegration:
    """End-to-end tests for replication package."""

    @pytest.mark.slow
    def test_full_replication_pipeline(self):
        """Test complete replication from data to tables."""
        from econirl.datasets import load_rust_bus
        from econirl.replication.rust1987 import (
            table_ii_descriptives,
            table_iv_transitions,
            table_v_structural,
        )

        # Load data
        df = load_rust_bus(original=False)  # Use synthetic for CI
        assert len(df) > 0

        # Table II
        t2 = table_ii_descriptives(df, original=False)
        assert len(t2) > 0

        # Table IV
        t4 = table_iv_transitions(df, original=False)
        assert len(t4) > 0
        assert (t4[['theta_0', 'theta_1', 'theta_2']].sum(axis=1) - 1.0).abs().max() < 0.01

        # Table V (single group for speed)
        t5 = table_v_structural(df, groups=[1], estimators=["Hotz-Miller"], original=False)
        assert len(t5) > 0
        assert t5['converged'].all()

    @pytest.mark.slow
    def test_monte_carlo_smoke(self):
        """Smoke test for Monte Carlo."""
        from econirl.replication.rust1987 import run_monte_carlo

        results = run_monte_carlo(
            n_simulations=3,
            n_individuals=50,
            n_periods=20,
            estimators=["Hotz-Miller"],
            seed=42,
        )

        assert len(results) == 3
        assert results['converged'].all()

    @pytest.mark.slow
    def test_export_pipeline(self):
        """Test export functionality."""
        import tempfile
        import os
        from econirl.replication.rust1987 import save_all_tables

        with tempfile.TemporaryDirectory() as tmpdir:
            save_all_tables(output_dir=tmpdir, original=False, groups=[1])

            # Verify output files exist
            assert os.path.exists(os.path.join(tmpdir, "table_ii.csv"))
            assert os.path.exists(os.path.join(tmpdir, "table_ii.tex"))
            assert os.path.exists(os.path.join(tmpdir, "table_iv.csv"))
            assert os.path.exists(os.path.join(tmpdir, "table_iv.tex"))
            assert os.path.exists(os.path.join(tmpdir, "table_v.csv"))
            assert os.path.exists(os.path.join(tmpdir, "table_v.tex"))
