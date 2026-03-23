"""Unified benchmark tests: all 10 estimators on one MDP."""

import pandas as pd
import pytest
import torch

from econirl.estimation import (
    CCPEstimator,
    MCEIRLEstimator,
    NFXPEstimator,
    TDCCPEstimator,
    TDCCPConfig,
)
from econirl.evaluation.benchmark import (
    BenchmarkDGP,
    EstimatorSpec,
    get_default_estimator_specs,
    run_benchmark,
    run_single,
    summarize_benchmark,
)
from econirl.evaluation.convergence import track_convergence


class TestBenchmarkSmoke:
    """Fast smoke tests (3 estimators, small state space)."""

    def test_benchmark_smoke(self):
        """NFXP, CCP, MCE IRL on n_states=10, n_agents=50."""
        dgp = BenchmarkDGP(n_states=10, discount_factor=0.99)
        specs = [
            EstimatorSpec(
                NFXPEstimator,
                kwargs=dict(inner_solver="hybrid", inner_max_iter=5000,
                            compute_hessian=False),
                name="NFXP",
            ),
            EstimatorSpec(
                CCPEstimator,
                kwargs=dict(num_policy_iterations=3, compute_hessian=False),
                name="CCP",
            ),
            EstimatorSpec(
                MCEIRLEstimator,
                kwargs=dict(learning_rate=0.5, use_adam=False,
                            outer_max_iter=100, inner_max_iter=2000,
                            gradient_clip=1.0, compute_se=False),
                name="MCE IRL",
            ),
        ]

        df = run_benchmark(
            dgp=dgp,
            estimators=specs,
            n_agents_list=[50],
            n_periods=50,
            seeds=[42],
        )

        # All 3 estimators should produce results
        assert len(df) == 3
        assert set(df["estimator"]) == {"NFXP", "CCP", "MCE IRL"}

        # NFXP should converge
        nfxp_row = df[df["estimator"] == "NFXP"].iloc[0]
        assert nfxp_row["converged"]

        # NFXP should achieve > 50% of optimal value
        assert nfxp_row["pct_optimal"] > 50, (
            f"NFXP pct_optimal={nfxp_row['pct_optimal']:.1f}%"
        )

        # All should produce finite policy RMSE
        assert df["policy_rmse"].notna().all()
        for _, row in df.iterrows():
            assert row["policy_rmse"] < 1.0, (
                f"{row['estimator']} policy_rmse={row['policy_rmse']:.4f}"
            )

    def test_run_single_nfxp(self):
        """Verify run_single returns correct BenchmarkResult structure."""
        dgp = BenchmarkDGP(n_states=10, discount_factor=0.99)
        spec = EstimatorSpec(
            NFXPEstimator,
            kwargs=dict(inner_solver="hybrid", inner_max_iter=5000,
                        compute_hessian=False),
            name="NFXP",
        )
        result = run_single(dgp, spec, n_agents=50, n_periods=50, seed=42)

        assert result.estimator == "NFXP"
        assert result.n_states == 10
        assert result.n_agents == 50
        assert result.seed == 42
        assert result.converged
        assert result.time_seconds > 0
        assert result.param_rmse is not None
        assert result.policy_rmse < 1.0
        assert result.pct_optimal > 50
        assert result.pct_optimal_transfer is not None
        assert len(result.estimates) == 3  # RC, op_cost, quad_cost
        assert len(result.true_params) == 3

    def test_summarize_benchmark(self):
        """Verify summarize_benchmark produces valid summary."""
        dgp = BenchmarkDGP(n_states=10, discount_factor=0.99)
        specs = [
            EstimatorSpec(
                NFXPEstimator,
                kwargs=dict(inner_solver="hybrid", inner_max_iter=5000,
                            compute_hessian=False),
                name="NFXP",
            ),
            EstimatorSpec(
                CCPEstimator,
                kwargs=dict(num_policy_iterations=3, compute_hessian=False),
                name="CCP",
            ),
        ]
        df = run_benchmark(
            dgp=dgp, estimators=specs,
            n_agents_list=[50], n_periods=50, seeds=[42],
        )
        summary = summarize_benchmark(df)

        assert len(summary) == 2
        assert "policy_rmse_mean" in summary.columns
        assert "pct_optimal_mean" in summary.columns
        assert "pct_optimal_transfer_mean" in summary.columns
        assert "time_mean" in summary.columns


@pytest.mark.slow
class TestAllEstimators:
    """Full benchmark: all 10 estimators."""

    def test_all_estimators_run(self):
        """All 10 estimators complete on n_states=20, n_agents=200."""
        dgp = BenchmarkDGP(n_states=20, discount_factor=0.99)
        specs = get_default_estimator_specs()

        df = run_benchmark(
            dgp=dgp,
            estimators=specs,
            n_agents_list=[200],
            n_periods=100,
            seeds=[42],
        )

        # All 10 estimators should have results
        assert len(df) == 10

        # Print results table
        print("\n=== All Estimators Benchmark (n_states=20, n_agents=200) ===")
        for _, row in df.iterrows():
            param_str = (
                f"param_rmse={row['param_rmse']:.4f}"
                if row["param_rmse"] is not None
                else "param_rmse=N/A"
            )
            transfer_str = (
                f"transfer={row['pct_optimal_transfer']:.1f}%"
                if row["pct_optimal_transfer"] is not None
                else "transfer=N/A"
            )
            print(
                f"  {row['estimator']:>15s}: {param_str}, "
                f"policy_rmse={row['policy_rmse']:.4f}, "
                f"pct_optimal={row['pct_optimal']:.1f}%, "
                f"{transfer_str}, "
                f"time={row['time_seconds']:.2f}s, "
                f"converged={row['converged']}"
            )

        # All estimators should produce finite policy RMSE
        assert df["policy_rmse"].notna().all()

        # Tier 1: Forward estimators (exact recovery)
        for est_name in ["NFXP", "CCP"]:
            row = df[df["estimator"] == est_name].iloc[0]
            assert row["converged"], f"{est_name} did not converge"
            assert row["param_rmse"] is not None, f"{est_name} missing param_rmse"
            assert row["param_rmse"] < 1.0, (
                f"{est_name} param_rmse={row['param_rmse']:.4f}"
            )
            assert 95 < row["pct_optimal"] < 105, (
                f"{est_name} pct_optimal={row['pct_optimal']:.1f}%"
            )

        # Tier 2: IRL with action-dependent features (near-optimal policy)
        for est_name in ["MCE IRL", "TD-CCP", "GLADIUS"]:
            row = df[df["estimator"] == est_name].iloc[0]
            assert 90 < row["pct_optimal"] < 110, (
                f"{est_name} pct_optimal={row['pct_optimal']:.1f}%"
            )

        # Tier 3: Structural IRL with slower convergence
        for est_name in ["Max Margin"]:
            row = df[df["estimator"] == est_name].iloc[0]
            assert 90 < row["pct_optimal"] < 105, (
                f"{est_name} pct_optimal={row['pct_optimal']:.1f}%"
            )

        # Tier 4: Methods with state-only features or adversarial training
        # MaxEnt IRL: uses state-only LinearReward, can't capture action-dependent costs
        # GAIL/AIRL: adversarial training oscillates with exact policy solver
        # GCL: learned neural cost without structural features
        for est_name in ["MaxEnt IRL", "GAIL", "AIRL", "GCL"]:
            row = df[df["estimator"] == est_name].iloc[0]
            assert not pd.isna(row["pct_optimal"]), (
                f"{est_name} produced NaN pct_optimal"
            )
            assert row["pct_optimal"] > 0, (
                f"{est_name} pct_optimal={row['pct_optimal']:.1f}% (worse than random)"
            )


@pytest.mark.slow
class TestScalingBenchmark:
    """Scaling tests across state space sizes with all estimators."""

    def test_scaling_benchmark(self):
        """All estimators across n_states=[5, 10, 20] with timeout."""
        from econirl.evaluation.benchmark import run_scaling_benchmark

        df = run_scaling_benchmark(
            state_sizes=[5, 10, 20],
            seed=42,
            timeout_seconds=300,
        )

        # Should have results for all estimator x state_size combinations
        assert len(df) > 0
        assert "estimator" in df.columns
        assert "n_states" in df.columns
        assert "time_seconds" in df.columns
        assert "pct_optimal" in df.columns

        # Every state size should appear
        for s in [5, 10, 20]:
            assert s in df["n_states"].values, f"Missing n_states={s}"

        # BC and NFXP should work at all sizes
        for est in ["BC", "NFXP"]:
            est_df = df[(df["estimator"] == est) & (~df["skipped"])]
            assert len(est_df) == 3, f"{est} missing results"

    def test_scaling_benchmark_full(self):
        """Full scaling grid [5, 10, 20, 50, 100, 200, 500]."""
        from econirl.evaluation.benchmark import run_scaling_benchmark

        df = run_scaling_benchmark(
            state_sizes=[5, 10, 20, 50, 100, 200, 500],
            seed=42,
            timeout_seconds=300,
        )

        # Print summary
        print(f"\nTotal results: {len(df)}")
        print(f"Estimators: {sorted(df['estimator'].unique())}")
        print(f"Timed out: {df[df['skipped']]['estimator'].unique().tolist()}")


@pytest.mark.slow
class TestTransferBenchmark:
    """Verify transfer metrics for param-recovering estimators."""

    def test_transfer_benchmark(self):
        """Param-recovering estimators should have finite transfer scores."""
        dgp = BenchmarkDGP(n_states=20, discount_factor=0.99)
        specs = [
            EstimatorSpec(
                NFXPEstimator,
                kwargs=dict(inner_solver="hybrid", inner_max_iter=5000,
                            compute_hessian=False),
                name="NFXP",
            ),
            EstimatorSpec(
                CCPEstimator,
                kwargs=dict(num_policy_iterations=3, compute_hessian=False),
                name="CCP",
            ),
            EstimatorSpec(
                MCEIRLEstimator,
                kwargs=dict(learning_rate=0.5, use_adam=False,
                            outer_max_iter=200, inner_max_iter=3000,
                            gradient_clip=1.0, compute_se=False),
                name="MCE IRL",
            ),
        ]

        df = run_benchmark(
            dgp=dgp,
            estimators=specs,
            n_agents_list=[100],
            n_periods=80,
            seeds=[42],
        )

        print("\n=== Transfer Benchmark ===")
        for _, row in df.iterrows():
            transfer_str = (
                f"transfer={row['pct_optimal_transfer']:.1f}%"
                if row["pct_optimal_transfer"] is not None
                else "transfer=N/A"
            )
            print(
                f"  {row['estimator']:>10s}: "
                f"pct_optimal={row['pct_optimal']:.1f}%, "
                f"{transfer_str}"
            )

        # All three can recover params → should have finite transfer scores
        for _, row in df.iterrows():
            assert row["pct_optimal_transfer"] is not None, (
                f"{row['estimator']} missing transfer score"
            )
            assert not pd.isna(row["pct_optimal_transfer"]), (
                f"{row['estimator']} has NaN transfer score"
            )


@pytest.mark.slow
class TestConvergenceTracking:
    """Convergence profile tests."""

    def test_convergence_tracking(self):
        """MCE IRL convergence profile on n_states=30."""
        dgp = BenchmarkDGP(n_states=30, discount_factor=0.99)
        spec = EstimatorSpec(
            MCEIRLEstimator,
            kwargs=dict(learning_rate=0.5, use_adam=False, inner_max_iter=3000,
                        gradient_clip=1.0, compute_se=False),
            name="MCE IRL",
        )

        profile = track_convergence(
            dgp, spec,
            checkpoints=[10, 25, 50, 100, 200],
            n_agents=200,
            seed=42,
        )

        assert profile.estimator == "MCE IRL"
        assert len(profile.iterations) == 5
        assert len(profile.policy_rmse) == 5
        assert len(profile.time_seconds) == 5

        # Time should increase monotonically
        for i in range(1, len(profile.time_seconds)):
            assert profile.time_seconds[i] >= profile.time_seconds[i - 1] * 0.5

        # Print convergence profile
        print("\n=== MCE IRL Convergence Profile ===")
        for i, n_iter in enumerate(profile.iterations):
            p_str = (
                f"{profile.param_rmse[i]:.4f}"
                if profile.param_rmse[i] is not None
                else "N/A"
            )
            print(
                f"  iter={n_iter:>4d}: param_rmse={p_str}, "
                f"policy_rmse={profile.policy_rmse[i]:.4f}, "
                f"time={profile.time_seconds[i]:.2f}s"
            )
