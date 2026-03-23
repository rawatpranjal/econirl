"""Shared fixtures for benchmark tests."""

import pytest
import torch

from econirl.environments import (
    RustBusEnvironment,
    MultiComponentBusEnvironment,
    GridworldEnvironment,
)
from econirl.evaluation.benchmark import BenchmarkDGP
from econirl.evaluation.utils import compute_policy
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Full-size environments (for slow parameter-recovery tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def rust_bus_env():
    """Rust (1987) bus environment with canonical parameters."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        discount_factor=0.9999,
    )


@pytest.fixture
def multi_component_bus_env_k2():
    """Multi-component bus environment with K=2 components."""
    return MultiComponentBusEnvironment(K=2, M=10, discount_factor=0.99)


@pytest.fixture
def gridworld_env():
    """5x5 gridworld environment."""
    return GridworldEnvironment(grid_size=5, discount_factor=0.99)


@pytest.fixture
def rust_bus_panel(rust_bus_env):
    """Large panel from Rust bus environment (500 agents x 100 periods)."""
    return simulate_panel(rust_bus_env, n_individuals=500, n_periods=100, seed=42)


@pytest.fixture
def multi_component_bus_panel_k2(multi_component_bus_env_k2):
    """Panel from multi-component bus (K=2) environment."""
    return simulate_panel(
        multi_component_bus_env_k2, n_individuals=500, n_periods=100, seed=42
    )


@pytest.fixture
def gridworld_panel(gridworld_env):
    """Panel from gridworld environment."""
    return simulate_panel(gridworld_env, n_individuals=500, n_periods=100, seed=42)


# ---------------------------------------------------------------------------
# Small / quick environments (for fast non-slow tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def small_rust_bus_env():
    """Smaller Rust bus environment for quick tests (lower discount)."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        discount_factor=0.99,
    )


@pytest.fixture
def small_rust_bus_panel(small_rust_bus_env):
    """Small panel for quick tests (50 agents x 50 periods)."""
    return simulate_panel(small_rust_bus_env, n_individuals=50, n_periods=50, seed=42)


@pytest.fixture
def small_gridworld_env():
    """3x3 gridworld for quick tests."""
    return GridworldEnvironment(grid_size=3, discount_factor=0.95)


@pytest.fixture
def small_gridworld_panel(small_gridworld_env):
    """Small gridworld panel for quick tests."""
    return simulate_panel(
        small_gridworld_env, n_individuals=50, n_periods=50, seed=42
    )


# ---------------------------------------------------------------------------
# Unified benchmark fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def benchmark_dgp():
    """Default DGP for unified benchmarks."""
    return BenchmarkDGP(
        n_states=20,
        replacement_cost=2.0,
        operating_cost=1.0,
        quadratic_cost=0.5,
        discount_factor=0.99,
    )


@pytest.fixture
def benchmark_env(benchmark_dgp):
    """K=1 multi-component bus environment for unified benchmarks."""
    return MultiComponentBusEnvironment(
        K=1,
        M=benchmark_dgp.n_states,
        replacement_cost=benchmark_dgp.replacement_cost,
        operating_cost=benchmark_dgp.operating_cost,
        quadratic_cost=benchmark_dgp.quadratic_cost,
        discount_factor=benchmark_dgp.discount_factor,
    )


@pytest.fixture
def benchmark_panel(benchmark_env):
    """Panel data for unified benchmarks (200 agents x 100 periods)."""
    return simulate_panel(benchmark_env, n_individuals=200, n_periods=100, seed=42)


@pytest.fixture
def true_policy(benchmark_env):
    """True optimal policy from the benchmark environment."""
    env = benchmark_env
    true_params = env.get_true_parameter_vector()
    return compute_policy(
        true_params, env.problem_spec, env.transition_matrices, env.feature_matrix
    )
