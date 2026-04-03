"""Tests for demand function analysis in the counterfactual module.

The demand function sweeps a parameter (e.g. replacement cost) and
computes the equilibrium replacement rate at each grid point.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation import simulate_panel
from econirl.simulation.counterfactual import demand_function, compute_stationary_distribution


@pytest.fixture
def estimated_result():
    """Quick NFXP estimation on small environment for demand function tests."""
    env = RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        seed=42,
    )
    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=123)
    utility = LinearUtility.from_environment(env)
    estimator = NFXPEstimator(
        optimizer="BHHH", inner_solver="hybrid", verbose=False,
    )
    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )
    return result, utility, env


class TestDemandFunction:
    """Tests for demand_function()."""

    def test_output_shapes(self, estimated_result):
        result, utility, env = estimated_result
        grid = jnp.linspace(0.5, 5.0, 10)

        df = demand_function(
            result=result,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
            parameter_name="replacement_cost",
            grid=grid,
        )

        assert df["demand"].shape == (10,)
        assert df["stationary_distributions"].shape == (10, 20)
        assert df["policies"].shape == (10, 20, 2)
        assert df["value_functions"].shape == (10, 20)
        assert df["parameter_name"] == "replacement_cost"

    def test_demand_decreases_with_rc(self, estimated_result):
        """Higher replacement cost should reduce replacement demand."""
        result, utility, env = estimated_result
        grid = jnp.linspace(1.0, 5.0, 5)

        df = demand_function(
            result=result,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
            parameter_name="replacement_cost",
            grid=grid,
        )

        # Demand should be monotonically decreasing with RC
        for i in range(len(grid) - 1):
            assert df["demand"][i] >= df["demand"][i + 1], (
                f"Demand should decrease with RC: "
                f"demand[{i}]={df['demand'][i]:.6f} < demand[{i+1}]={df['demand'][i+1]:.6f}"
            )

    def test_baseline_demand_matches_result(self, estimated_result):
        """Baseline demand should match the replacement rate from estimation."""
        result, utility, env = estimated_result
        grid = jnp.array([float(result.parameters[-1])])  # RC at estimated value

        df = demand_function(
            result=result,
            utility=utility,
            problem=env.problem_spec,
            transitions=env.transition_matrices,
            parameter_name="replacement_cost",
            grid=grid,
        )

        # Demand at estimated RC should match baseline_demand
        np.testing.assert_allclose(
            df["demand"][0], df["baseline_demand"], atol=0.01,
        )

    def test_scaling_with_num_buses(self, estimated_result):
        """Demand should scale linearly with num_buses."""
        result, utility, env = estimated_result
        grid = jnp.linspace(1.0, 4.0, 3)

        df1 = demand_function(
            result=result, utility=utility, problem=env.problem_spec,
            transitions=env.transition_matrices,
            parameter_name="replacement_cost", grid=grid,
            num_buses=1,
        )
        df10 = demand_function(
            result=result, utility=utility, problem=env.problem_spec,
            transitions=env.transition_matrices,
            parameter_name="replacement_cost", grid=grid,
            num_buses=10,
        )

        np.testing.assert_allclose(df10["demand"], 10 * df1["demand"], atol=1e-10)

    def test_stationary_distributions_sum_to_one(self, estimated_result):
        """Each stationary distribution should sum to 1."""
        result, utility, env = estimated_result
        grid = jnp.linspace(1.0, 4.0, 5)

        df = demand_function(
            result=result, utility=utility, problem=env.problem_spec,
            transitions=env.transition_matrices,
            parameter_name="replacement_cost", grid=grid,
        )

        for i in range(len(grid)):
            np.testing.assert_allclose(
                df["stationary_distributions"][i].sum(), 1.0, atol=1e-6,
            )
