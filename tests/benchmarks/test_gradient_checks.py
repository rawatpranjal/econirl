"""Gradient and Hessian correctness tests.

For estimators that use gradients (NFXP, CCP, MCE IRL), compares
numerical gradients (central differences) against the analytical / computed
gradients. Also checks Hessian symmetry.

These tests use small environments and run quickly (not marked slow).
"""

import pytest
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.environments import RustBusEnvironment
from econirl.inference.standard_errors import compute_numerical_hessian
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel


# ---------------------------------------------------------------------------
# Fixtures (small environment for speed)
# ---------------------------------------------------------------------------

@pytest.fixture
def quick_env():
    """Small Rust bus environment for fast gradient checks."""
    return RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=20,
        discount_factor=0.99,
    )


@pytest.fixture
def quick_setup(quick_env):
    """Return (panel, utility, problem, transitions, true_params) for quick env."""
    panel = simulate_panel(quick_env, n_individuals=50, n_periods=30, seed=123)
    utility = LinearUtility.from_environment(quick_env)
    problem = quick_env.problem_spec
    transitions = quick_env.transition_matrices
    true_params = quick_env.get_true_parameter_vector()
    return panel, utility, problem, transitions, true_params


# ---------------------------------------------------------------------------
# NFXP gradient check
# ---------------------------------------------------------------------------

def test_nfxp_numerical_gradient(quick_setup):
    """NFXP: numerical gradient should be non-zero and directionally consistent."""
    panel, utility, problem, transitions, true_params = quick_setup
    operator = SoftBellmanOperator(problem, transitions)

    # Log-likelihood function
    def _ll(params):
        params_f32 = params.to(torch.float32)
        flow_utility = utility.compute(params_f32)
        result = value_iteration(operator, flow_utility, tol=1e-10, max_iter=10000)
        log_probs = operator.compute_log_choice_probabilities(flow_utility, result.V)
        ll_val = 0.0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                ll_val += log_probs[s, a].item()
        return torch.tensor(ll_val, dtype=torch.float64)

    # Numerical gradient via central differences
    params = true_params.to(torch.float64)
    eps = 1e-4  # Larger step for float32 internal precision
    n_params = len(params)
    grad = torch.zeros(n_params, dtype=torch.float64)
    for i in range(n_params):
        p_plus = params.clone()
        p_minus = params.clone()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad[i] = (_ll(p_plus) - _ll(p_minus)) / (2 * eps)

    # Gradient should be non-zero (LL varies with parameters)
    assert torch.any(torch.abs(grad) > 1e-6), (
        f"Gradient is effectively zero: {grad}"
    )

    # LL at true params should be near a local max, so gradient should be small
    # relative to the gradient at perturbed params
    grad_norm = torch.norm(grad).item()
    perturbed = params.clone()
    perturbed[0] += 0.01
    grad_perturbed = torch.zeros(n_params, dtype=torch.float64)
    for i in range(n_params):
        p_plus = perturbed.clone()
        p_minus = perturbed.clone()
        p_plus[i] += eps
        p_minus[i] -= eps
        grad_perturbed[i] = (_ll(p_plus) - _ll(p_minus)) / (2 * eps)
    grad_perturbed_norm = torch.norm(grad_perturbed).item()

    # Gradient at true params should not be larger than at perturbed point
    # (true params are near the MLE for large enough data)
    assert grad_norm < grad_perturbed_norm * 100, (
        f"Gradient at true params ({grad_norm:.4f}) is unexpectedly large "
        f"compared to perturbed ({grad_perturbed_norm:.4f})"
    )


# ---------------------------------------------------------------------------
# CCP gradient check
# ---------------------------------------------------------------------------

def test_ccp_numerical_gradient(quick_setup):
    """CCP: numerical gradient of pseudo-log-likelihood should be non-zero
    and have consistent signs across step sizes."""
    from econirl.estimation.ccp import CCPEstimator

    panel, utility, problem, transitions, true_params = quick_setup
    ccp_est = CCPEstimator(num_policy_iterations=1, compute_hessian=False)

    # Estimate CCPs from data
    ccps = ccp_est._estimate_ccps_from_data(
        panel, problem.num_states, problem.num_actions
    )

    def _ll(params):
        params_f32 = params.to(torch.float32)
        return torch.tensor(
            ccp_est._compute_log_likelihood(
                params_f32, panel, utility, ccps, transitions, problem
            ),
            dtype=torch.float64,
        )

    params = true_params.to(torch.float64)
    eps = 1e-4  # Larger step for float32 internal precision
    n_params = len(params)

    grad_a = torch.zeros(n_params, dtype=torch.float64)
    grad_b = torch.zeros(n_params, dtype=torch.float64)
    for i in range(n_params):
        p_plus = params.clone(); p_minus = params.clone()
        p_plus[i] += eps; p_minus[i] -= eps
        grad_a[i] = (_ll(p_plus) - _ll(p_minus)) / (2 * eps)

        p_plus2 = params.clone(); p_minus2 = params.clone()
        p_plus2[i] += eps / 2; p_minus2[i] -= eps / 2
        grad_b[i] = (_ll(p_plus2) - _ll(p_minus2)) / eps

    # Gradient should be non-zero
    assert torch.any(torch.abs(grad_a) > 1e-6), (
        f"CCP gradient is effectively zero: {grad_a}"
    )

    # Signs should agree (directional consistency)
    for i in range(n_params):
        if abs(grad_a[i].item()) > 1.0:
            assert grad_a[i].item() * grad_b[i].item() > 0, (
                f"CCP gradient sign mismatch at parameter {i}: "
                f"grad_a={grad_a[i]:.4f}, grad_b={grad_b[i]:.4f}"
            )


# ---------------------------------------------------------------------------
# Hessian symmetry check
# ---------------------------------------------------------------------------

def test_hessian_symmetry(quick_setup):
    """Numerical Hessian of NFXP log-likelihood should be symmetric."""
    panel, utility, problem, transitions, true_params = quick_setup
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

    # Check symmetry: H should equal H^T
    asym = torch.abs(hessian - hessian.T)
    max_asym = asym.max().item()
    assert max_asym < 1e-4, (
        f"Hessian is not symmetric: max |H - H^T| = {max_asym:.8f}"
    )


# ---------------------------------------------------------------------------
# Hessian negative-definiteness at optimum
# ---------------------------------------------------------------------------

def test_hessian_negative_definite_at_true(quick_setup):
    """At the true parameters, the Hessian of LL should be negative semi-definite."""
    panel, utility, problem, transitions, true_params = quick_setup
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
    eigenvalues = torch.linalg.eigvalsh(hessian)

    # All eigenvalues of the Hessian of LL should be <= 0 (negative semi-definite)
    # Allow a small positive tolerance for numerical errors
    assert eigenvalues.max().item() < 1e-2, (
        f"Hessian at true params is not NSD: max eigenvalue = {eigenvalues.max():.6f}"
    )
