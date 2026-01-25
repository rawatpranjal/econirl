"""Pytest fixtures for econirl tests.

This module provides reusable fixtures for testing:
- Rust bus environment with known parameters
- Simulated panel data
- Utility specifications
- Pre-computed solutions
"""

import pytest
import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ============================================================================
# Environment Fixtures
# ============================================================================

@pytest.fixture
def rust_env() -> RustBusEnvironment:
    """Standard Rust bus environment with default parameters."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=90,
        discount_factor=0.9999,
        scale_parameter=1.0,
        seed=42,
    )


@pytest.fixture
def rust_env_small() -> RustBusEnvironment:
    """Small Rust bus environment for faster tests."""
    return RustBusEnvironment(
        operating_cost=0.01,
        replacement_cost=2.0,
        num_mileage_bins=20,
        discount_factor=0.99,
        scale_parameter=1.0,
        seed=42,
    )


@pytest.fixture
def problem_spec(rust_env: RustBusEnvironment) -> DDCProblem:
    """DDCProblem from standard environment."""
    return rust_env.problem_spec


@pytest.fixture
def problem_spec_small(rust_env_small: RustBusEnvironment) -> DDCProblem:
    """DDCProblem from small environment."""
    return rust_env_small.problem_spec


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture
def small_panel(rust_env_small: RustBusEnvironment) -> Panel:
    """Small panel for quick tests."""
    return simulate_panel(
        rust_env_small,
        n_individuals=50,
        n_periods=50,
        seed=123,
    )


@pytest.fixture
def medium_panel(rust_env: RustBusEnvironment) -> Panel:
    """Medium panel for estimation tests."""
    return simulate_panel(
        rust_env,
        n_individuals=100,
        n_periods=100,
        seed=456,
    )


@pytest.fixture
def large_panel(rust_env: RustBusEnvironment) -> Panel:
    """Large panel for accuracy tests."""
    return simulate_panel(
        rust_env,
        n_individuals=500,
        n_periods=100,
        seed=789,
    )


@pytest.fixture
def single_trajectory() -> Trajectory:
    """Single trajectory for unit tests."""
    return Trajectory(
        states=torch.tensor([0, 1, 2, 3, 4, 5]),
        actions=torch.tensor([0, 0, 0, 0, 1, 0]),
        next_states=torch.tensor([1, 2, 3, 4, 0, 1]),
        individual_id=0,
    )


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def utility(rust_env: RustBusEnvironment) -> LinearUtility:
    """LinearUtility from standard environment."""
    return LinearUtility.from_environment(rust_env)


@pytest.fixture
def utility_small(rust_env_small: RustBusEnvironment) -> LinearUtility:
    """LinearUtility from small environment."""
    return LinearUtility.from_environment(rust_env_small)


# ============================================================================
# Transition Fixtures
# ============================================================================

@pytest.fixture
def transitions(rust_env: RustBusEnvironment) -> torch.Tensor:
    """Transition matrices from standard environment."""
    return rust_env.transition_matrices


@pytest.fixture
def transitions_small(rust_env_small: RustBusEnvironment) -> torch.Tensor:
    """Transition matrices from small environment."""
    return rust_env_small.transition_matrices


# ============================================================================
# Solution Fixtures
# ============================================================================

@pytest.fixture
def bellman_operator(
    problem_spec: DDCProblem, transitions: torch.Tensor
) -> SoftBellmanOperator:
    """Bellman operator for standard environment."""
    return SoftBellmanOperator(problem_spec, transitions)


@pytest.fixture
def optimal_policy(
    rust_env: RustBusEnvironment,
    bellman_operator: SoftBellmanOperator,
) -> torch.Tensor:
    """Optimal policy from true parameters."""
    utility_matrix = rust_env.compute_utility_matrix()
    result = value_iteration(bellman_operator, utility_matrix)
    return result.policy


@pytest.fixture
def optimal_value(
    rust_env: RustBusEnvironment,
    bellman_operator: SoftBellmanOperator,
) -> torch.Tensor:
    """Optimal value function from true parameters."""
    utility_matrix = rust_env.compute_utility_matrix()
    result = value_iteration(bellman_operator, utility_matrix)
    return result.V


# ============================================================================
# Parameter Fixtures
# ============================================================================

@pytest.fixture
def true_params(rust_env: RustBusEnvironment) -> torch.Tensor:
    """True parameters as tensor."""
    return rust_env.get_true_parameter_vector()


@pytest.fixture
def true_params_small(rust_env_small: RustBusEnvironment) -> torch.Tensor:
    """True parameters from small environment."""
    return rust_env_small.get_true_parameter_vector()


@pytest.fixture
def perturbed_params(true_params: torch.Tensor) -> torch.Tensor:
    """Parameters perturbed from true values (for testing)."""
    return true_params * 1.5


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture
def seed() -> int:
    """Standard seed for reproducibility."""
    return 42


@pytest.fixture
def tolerance() -> float:
    """Standard numerical tolerance."""
    return 1e-6


@pytest.fixture
def estimation_tolerance() -> float:
    """Tolerance for estimation accuracy (looser)."""
    return 0.5  # Within 0.5 of true value


# ============================================================================
# Helper Functions as Fixtures
# ============================================================================

@pytest.fixture
def assert_valid_policy():
    """Fixture providing a policy validation function."""
    def _assert_valid_policy(policy: torch.Tensor):
        """Assert that policy is valid (rows sum to 1, all non-negative)."""
        assert (policy >= 0).all(), "Policy has negative probabilities"
        row_sums = policy.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "Policy rows don't sum to 1"
    return _assert_valid_policy


@pytest.fixture
def assert_valid_value_function():
    """Fixture providing a value function validation function."""
    def _assert_valid_value(V: torch.Tensor, problem: DDCProblem):
        """Assert that value function has correct shape and is finite."""
        assert V.shape == (problem.num_states,), \
            f"Value function has wrong shape: {V.shape}"
        assert torch.isfinite(V).all(), "Value function has non-finite values"
    return _assert_valid_value


# ============================================================================
# MCE IRL Test Fixtures
# ============================================================================

@pytest.fixture
def mce_irl_seed():
    """Fixture for reproducible random state in MCE IRL tests.

    This fixture properly manages numpy random state for test isolation.
    """
    old_state = np.random.get_state()
    np.random.seed(42)
    yield 42
    np.random.set_state(old_state)


@pytest.fixture
def simple_problem():
    """Create a simple 10-state MDP with known structure.

    This fixture is shared across MCE IRL test classes for:
    - TestMCEIRLConvergence
    - TestFeatureMatching
    """
    n_states = 10
    problem = DDCProblem(
        num_states=n_states,
        num_actions=2,
        discount_factor=0.95,
    )

    # Deterministic transitions: keep -> next state, replace -> state 0
    transitions = torch.zeros((2, n_states, n_states))
    for s in range(n_states):
        transitions[0, s, min(s + 1, n_states - 1)] = 1.0  # keep
        transitions[1, s, 0] = 1.0  # replace

    return problem, transitions


@pytest.fixture
def synthetic_panel(simple_problem, mce_irl_seed):
    """Generate synthetic data from a known policy.

    This fixture is shared across MCE IRL test classes for:
    - TestMCEIRLConvergence
    - TestFeatureMatching

    Uses mce_irl_seed fixture for proper random state management.
    """
    problem, transitions = simple_problem
    n_states = problem.num_states

    trajectories = []

    for i in range(20):
        states, actions, next_states = [], [], []
        s = 0
        for t in range(50):
            states.append(s)
            # Replace with higher prob at high states
            p_replace = 0.05 + 0.15 * s / n_states
            a = 1 if np.random.random() < p_replace else 0
            actions.append(a)
            # Compute next state based on action
            if a == 1:
                next_s = 0
            else:
                next_s = min(s + 1, n_states - 1)
            next_states.append(next_s)
            s = next_s

        traj = Trajectory(
            states=torch.tensor(states, dtype=torch.long),
            actions=torch.tensor(actions, dtype=torch.long),
            next_states=torch.tensor(next_states, dtype=torch.long),
            individual_id=i,
        )
        trajectories.append(traj)

    return Panel(trajectories=trajectories)
