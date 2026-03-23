"""Inner-loop convergence tests.

Verifies that:
1. Value iteration converges for the Rust bus environment.
2. The soft Bellman operator is a contraction.
3. MCE IRL feature matching holds at convergence.

These tests use small environments and run quickly (not marked slow).
"""

import pytest
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.environments import RustBusEnvironment, GridworldEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_rust_env():
    """Small Rust bus environment for quick convergence tests."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=20,
        discount_factor=0.99,
    )


@pytest.fixture
def small_grid_env():
    """Small gridworld for quick convergence tests."""
    return GridworldEnvironment(grid_size=3, discount_factor=0.95)


# ---------------------------------------------------------------------------
# Value iteration convergence
# ---------------------------------------------------------------------------

def test_vi_converges_rust_bus(small_rust_env):
    """Value iteration should converge for the Rust bus environment."""
    env = small_rust_env
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()
    flow_utility = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)

    assert result.converged, (
        f"VI did not converge after {result.num_iterations} iterations "
        f"(final error={result.final_error:.2e})"
    )
    assert result.final_error < 1e-10


def test_vi_converges_gridworld(small_grid_env):
    """Value iteration should converge for the gridworld environment."""
    env = small_grid_env
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()
    flow_utility = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)

    assert result.converged, (
        f"VI did not converge after {result.num_iterations} iterations "
        f"(final error={result.final_error:.2e})"
    )


# ---------------------------------------------------------------------------
# Soft Bellman operator is a contraction
# ---------------------------------------------------------------------------

def test_soft_bellman_is_contraction(small_rust_env):
    """Applying the soft Bellman operator should strictly reduce the distance
    between two distinct value function iterates (contraction property).

    The operator T satisfies: ||T(V1) - T(V2)||_inf <= beta * ||V1 - V2||_inf
    """
    env = small_rust_env
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()
    flow_utility = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)

    # Two different starting value functions
    V1 = torch.zeros(problem.num_states)
    V2 = torch.randn(problem.num_states) * 5.0

    dist_before = torch.abs(V1 - V2).max().item()

    # Apply operator once to each
    result1 = operator.apply(flow_utility, V1)
    result2 = operator.apply(flow_utility, V2)

    dist_after = torch.abs(result1.V - result2.V).max().item()

    # Contraction: distance must decrease by at least factor beta
    beta = problem.discount_factor
    assert dist_after <= beta * dist_before + 1e-8, (
        f"Bellman operator is not a contraction: "
        f"dist_after={dist_after:.6f} > beta*dist_before={beta * dist_before:.6f}"
    )


def test_contraction_multiple_steps(small_rust_env):
    """Repeated applications of the Bellman operator should monotonically
    decrease the distance between two value functions."""
    env = small_rust_env
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()
    flow_utility = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)

    V1 = torch.zeros(problem.num_states)
    V2 = torch.randn(problem.num_states) * 10.0

    prev_dist = torch.abs(V1 - V2).max().item()

    for step in range(20):
        r1 = operator.apply(flow_utility, V1)
        r2 = operator.apply(flow_utility, V2)
        V1, V2 = r1.V, r2.V
        dist = torch.abs(V1 - V2).max().item()
        assert dist <= prev_dist + 1e-8, (
            f"Distance increased at step {step}: {dist:.6f} > {prev_dist:.6f}"
        )
        prev_dist = dist


# ---------------------------------------------------------------------------
# Policy is a valid probability distribution after VI
# ---------------------------------------------------------------------------

def test_policy_valid_distribution(small_rust_env):
    """Policy from value iteration should be a valid probability distribution."""
    env = small_rust_env
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()
    flow_utility = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)

    policy = result.policy

    # All entries non-negative
    assert (policy >= 0).all(), "Policy contains negative probabilities"

    # Rows sum to 1
    row_sums = policy.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
        f"Policy rows do not sum to 1: max deviation = {(row_sums - 1).abs().max():.8f}"
    )


# ---------------------------------------------------------------------------
# MCE IRL feature matching at convergence
# ---------------------------------------------------------------------------

def test_mce_feature_matching_at_convergence(small_rust_env):
    """After MCE IRL converges with true parameters, the expected features
    under the learned policy should approximately match the empirical features."""
    env = small_rust_env
    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=42)
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()

    # Compute empirical feature expectations
    feature_matrix = utility.feature_matrix  # (S, A, K)
    n_obs = 0
    empirical_features = torch.zeros(feature_matrix.shape[2])
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            empirical_features += feature_matrix[s, a, :]
            n_obs += 1
    empirical_features /= n_obs

    # Compute policy at true parameters
    flow_utility = utility.compute(true_params)
    operator = SoftBellmanOperator(problem, transitions)
    result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
    policy = result.policy  # (S, A)

    # Compute expected features under policy, iterating over empirical states
    expected_features = torch.zeros(feature_matrix.shape[2])
    n_obs2 = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            for a_idx in range(problem.num_actions):
                expected_features += policy[s, a_idx] * feature_matrix[s, a_idx, :]
            n_obs2 += 1
    expected_features /= n_obs2

    # Features should approximately match
    diff = torch.abs(empirical_features - expected_features)
    max_diff = diff.max().item()
    assert max_diff < 0.5, (
        f"Feature matching at true params failed: max |empirical - expected| = {max_diff:.4f}"
    )
