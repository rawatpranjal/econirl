"""Identification diagnostics tests.

Verifies that:
1. The Hessian has full rank for well-specified models.
2. Eigenvalues of the negative Hessian are positive.
3. Under-identification is correctly detected with collinear features.

These tests use small environments and run quickly (not marked slow).
"""

import pytest
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.environments import RustBusEnvironment
from econirl.inference.identification import check_identification
from econirl.inference.standard_errors import compute_numerical_hessian
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def id_env():
    """Small Rust bus environment for identification tests."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=20,
        discount_factor=0.99,
    )


@pytest.fixture
def id_setup(id_env):
    """Return (panel, utility, problem, transitions, true_params, hessian)."""
    panel = simulate_panel(id_env, n_individuals=100, n_periods=50, seed=42)
    utility = LinearUtility.from_environment(id_env)
    problem = id_env.problem_spec
    transitions = id_env.transition_matrices
    true_params = id_env.get_true_parameter_vector()

    operator = SoftBellmanOperator(problem, transitions)

    def _ll(params):
        flow_utility = utility.compute(params)
        result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
        log_probs = operator.compute_log_choice_probabilities(flow_utility, result.V)
        ll_val = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ll_val += log_probs[s, a].item()
        return torch.tensor(ll_val)

    hessian = compute_numerical_hessian(true_params, _ll)
    return panel, utility, problem, transitions, true_params, hessian


# ---------------------------------------------------------------------------
# Hessian rank test
# ---------------------------------------------------------------------------

def test_hessian_full_rank(id_setup):
    """Hessian at true parameters should have full rank (well-identified model)."""
    _, utility, _, _, _, hessian = id_setup
    diag = check_identification(hessian, utility.parameter_names)
    n_params = len(utility.parameter_names)
    assert diag.rank == n_params, (
        f"Hessian rank {diag.rank} < {n_params}: model appears under-identified. "
        f"Status: {diag.status}"
    )


# ---------------------------------------------------------------------------
# Eigenvalue positivity of negative Hessian
# ---------------------------------------------------------------------------

def test_negative_hessian_positive_eigenvalues(id_setup):
    """Eigenvalues of -H should all be positive at the true parameters."""
    _, _, _, _, _, hessian = id_setup
    neg_hessian = -hessian
    eigenvalues = torch.linalg.eigvalsh(neg_hessian)

    # All eigenvalues should be positive (negative Hessian is positive definite)
    assert eigenvalues.min().item() > -1e-4, (
        f"Negative Hessian has non-positive eigenvalue: min = {eigenvalues.min():.6f}"
    )


def test_identification_status_well_identified(id_setup):
    """check_identification should report well-identified for Rust bus."""
    _, utility, _, _, _, hessian = id_setup
    diag = check_identification(hessian, utility.parameter_names)
    assert diag.is_positive_definite, (
        f"Model not positive definite: status = {diag.status}"
    )
    # Status should not indicate under-identification
    assert "Under-identified" not in diag.status, (
        f"Model incorrectly flagged: {diag.status}"
    )


# ---------------------------------------------------------------------------
# Under-identification detection with collinear features
# ---------------------------------------------------------------------------

def test_collinear_features_detected(id_env):
    """With collinear features, the Hessian should be rank-deficient."""
    env = id_env
    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=42)
    problem = env.problem_spec
    transitions = env.transition_matrices

    # Build a feature matrix with a collinear third column (exact copy of col0)
    original_features = env.feature_matrix  # (S, A, 2)
    S, A, _ = original_features.shape
    collinear_col = original_features[:, :, 0:1].clone()  # exact copy
    bad_features = torch.cat([original_features, collinear_col], dim=2)  # (S, A, 3)

    utility = LinearUtility(
        feature_matrix=bad_features,
        parameter_names=["operating_cost", "replacement_cost", "collinear"],
    )

    # Use arbitrary parameters for the collinear model
    params = torch.tensor([0.001, 3.0, 0.0005], dtype=torch.float32)
    operator = SoftBellmanOperator(problem, transitions)

    def _ll(p):
        flow_utility = utility.compute(p)
        result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
        log_probs = operator.compute_log_choice_probabilities(flow_utility, result.V)
        ll_val = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ll_val += log_probs[s, a].item()
        return torch.tensor(ll_val)

    hessian = compute_numerical_hessian(params, _ll)
    diag = check_identification(hessian, utility.parameter_names)

    # The Hessian should have high condition number or be under-identified
    # Exact collinearity means the last eigenvalue should be near zero
    eigenvalues = torch.linalg.eigvalsh(-hessian)
    min_eigenvalue = eigenvalues.min().item()
    max_eigenvalue = eigenvalues.max().item()
    ratio = min_eigenvalue / max(max_eigenvalue, 1e-10)

    assert ratio < 0.01 or diag.rank < 3 or diag.hessian_condition_number > 1e4, (
        f"Collinear features not detected: ratio={ratio:.6f}, rank={diag.rank}, "
        f"condition={diag.hessian_condition_number:.2e}"
    )


# ---------------------------------------------------------------------------
# Condition number sanity check
# ---------------------------------------------------------------------------

def test_condition_number_finite(id_setup):
    """Condition number of -H should be finite for well-specified model."""
    _, utility, _, _, _, hessian = id_setup
    diag = check_identification(hessian, utility.parameter_names)
    assert diag.hessian_condition_number < float("inf"), (
        "Condition number is infinite (singular Hessian)"
    )
    # Condition number for the Rust bus should be reasonable
    assert diag.hessian_condition_number < 1e8, (
        f"Condition number is very large: {diag.hessian_condition_number:.2e}"
    )
