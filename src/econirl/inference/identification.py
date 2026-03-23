"""Identification diagnostics for structural estimation.

This module provides tools for checking whether parameters are identified
in dynamic discrete choice models. Identification issues are common in
structural estimation and can lead to:
- Unreliable standard errors
- Non-convergence of optimization
- Sensitivity to starting values

Key diagnostics:
1. Hessian rank and condition number
2. Eigenvalue analysis
3. Parameter sensitivity analysis
4. Information matrix tests
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import torch

from econirl.inference.results import IdentificationDiagnostics


def check_identification(
    hessian: torch.Tensor,
    parameter_names: list[str] | None = None,
    tol: float = 1e-6,
) -> IdentificationDiagnostics:
    """Check identification of parameters from Hessian.

    Analyzes the Hessian (or information) matrix to determine whether
    all parameters are locally identified. Key checks:

    1. Rank: Hessian should have full rank
    2. Eigenvalues: All should be negative (for maximization)
    3. Condition number: Low is better (high indicates near-singularity)

    Args:
        hessian: Hessian matrix at optimum, shape (n_params, n_params)
        parameter_names: Names of parameters (for reporting)
        tol: Tolerance for considering eigenvalues as zero

    Returns:
        IdentificationDiagnostics with detailed analysis
    """
    n_params = hessian.shape[0]

    # Compute eigenvalues of negative Hessian (should be positive for well-identified)
    neg_hessian = -hessian
    eigenvalues = torch.linalg.eigvalsh(neg_hessian)

    # Sort eigenvalues
    eigenvalues_sorted = torch.sort(eigenvalues).values

    min_eigenvalue = eigenvalues_sorted[0].item()
    max_eigenvalue = eigenvalues_sorted[-1].item()

    # Numerical rank
    rank = (eigenvalues.abs() > tol).sum().item()

    # Condition number (ratio of largest to smallest eigenvalue)
    if min_eigenvalue > tol:
        condition_number = max_eigenvalue / min_eigenvalue
    else:
        condition_number = float("inf")

    # Positive definiteness check
    is_positive_definite = bool(min_eigenvalue > tol)

    # Status determination
    if rank < n_params:
        status = f"Under-identified (rank {rank} < {n_params})"
    elif not is_positive_definite:
        status = "Saddle point or local minimum (not a maximum)"
    elif condition_number > 1e6:
        status = "Weakly identified (near-singular Hessian)"
    elif condition_number > 1e4:
        status = "Potentially weakly identified"
    else:
        status = "Well-identified"

    return IdentificationDiagnostics(
        hessian_condition_number=condition_number,
        min_eigenvalue=min_eigenvalue,
        max_eigenvalue=max_eigenvalue,
        rank=rank,
        is_positive_definite=is_positive_definite,
        status=status,
    )


@dataclass
class SensitivityAnalysis:
    """Results of parameter sensitivity analysis.

    Attributes:
        parameter_name: Name of the parameter analyzed
        elasticities: Elasticity of other parameters w.r.t. this one
        influence_score: Overall measure of parameter influence
    """

    parameter_name: str
    elasticities: dict[str, float]
    influence_score: float


def analyze_parameter_sensitivity(
    hessian: torch.Tensor,
    parameters: torch.Tensor,
    parameter_names: list[str],
) -> list[SensitivityAnalysis]:
    """Analyze sensitivity of estimates to each parameter.

    Computes how much other parameter estimates would change if
    one parameter were fixed at a slightly different value.

    This can reveal:
    - Which parameters are most influential
    - Potential collinearity issues
    - Parameters that are hard to separately identify

    Args:
        hessian: Hessian matrix at optimum
        parameters: Estimated parameter values
        parameter_names: Names of parameters

    Returns:
        List of SensitivityAnalysis for each parameter
    """
    n_params = len(parameters)
    results = []

    # Compute variance-covariance matrix
    try:
        var_cov = torch.linalg.inv(-hessian)
    except RuntimeError:
        # Hessian not invertible
        return [
            SensitivityAnalysis(
                parameter_name=name,
                elasticities={},
                influence_score=float("nan"),
            )
            for name in parameter_names
        ]

    # Correlation matrix
    std_devs = torch.sqrt(torch.diag(var_cov))
    corr_matrix = var_cov / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1))

    for i, name_i in enumerate(parameter_names):
        elasticities = {}

        for j, name_j in enumerate(parameter_names):
            if i != j:
                # Elasticity: how much j changes per unit change in i
                # Use correlation as a proxy
                elasticities[name_j] = corr_matrix[i, j].item()

        # Influence score: average absolute correlation with other parameters
        other_corrs = [abs(corr_matrix[i, j].item()) for j in range(n_params) if j != i]
        influence_score = np.mean(other_corrs) if other_corrs else 0.0

        results.append(
            SensitivityAnalysis(
                parameter_name=name_i,
                elasticities=elasticities,
                influence_score=influence_score,
            )
        )

    return results


def check_local_identification(
    log_likelihood_fn: Callable[[torch.Tensor], torch.Tensor],
    parameters: torch.Tensor,
    parameter_names: list[str],
    eps: float = 1e-5,
    n_directions: int = 100,
    seed: int | None = None,
) -> dict:
    """Check local identification by exploring likelihood surface.

    Samples random directions in parameter space and checks whether
    the likelihood decreases in all directions (indicating a local maximum).

    This can detect:
    - Flat regions in the likelihood
    - Ridge-like structures
    - Multiple nearby optima

    Args:
        log_likelihood_fn: Function mapping parameters to log-likelihood
        parameters: Estimated parameters (should be at optimum)
        parameter_names: Names of parameters
        eps: Step size for exploration
        n_directions: Number of random directions to test
        seed: Random seed

    Returns:
        Dictionary with identification diagnostics
    """
    rng = np.random.default_rng(seed)
    n_params = len(parameters)

    ll_at_opt = log_likelihood_fn(parameters).item()

    # Sample random directions on unit sphere
    directions = torch.tensor(
        rng.standard_normal((n_directions, n_params)),
        dtype=parameters.dtype,
    )
    directions = directions / torch.norm(directions, dim=1, keepdim=True)

    # Check likelihood in each direction
    ll_increases = 0
    flat_directions = 0
    decrease_magnitudes = []

    for d in directions:
        ll_plus = log_likelihood_fn(parameters + eps * d).item()
        ll_minus = log_likelihood_fn(parameters - eps * d).item()

        # Check if likelihood increases in either direction
        if ll_plus > ll_at_opt + 1e-10 or ll_minus > ll_at_opt + 1e-10:
            ll_increases += 1

        # Check for flat directions
        if abs(ll_plus - ll_at_opt) < 1e-10 and abs(ll_minus - ll_at_opt) < 1e-10:
            flat_directions += 1

        # Record decrease magnitude
        decrease = min(ll_at_opt - ll_plus, ll_at_opt - ll_minus)
        decrease_magnitudes.append(decrease)

    # Summary statistics
    avg_decrease = np.mean(decrease_magnitudes)
    min_decrease = np.min(decrease_magnitudes)

    if ll_increases > 0:
        status = "Not at local maximum"
    elif flat_directions > n_directions * 0.1:
        status = "Potentially flat likelihood surface"
    elif min_decrease < 1e-8:
        status = "Near-flat directions detected"
    else:
        status = "Local maximum confirmed"

    return {
        "status": status,
        "ll_at_optimum": ll_at_opt,
        "directions_checked": n_directions,
        "directions_increasing": ll_increases,
        "flat_directions": flat_directions,
        "avg_decrease": avg_decrease,
        "min_decrease": min_decrease,
    }


def compute_information_matrix_equality_test(
    hessian: torch.Tensor,
    outer_product_gradient: torch.Tensor,
    n_observations: int,
) -> dict:
    """Test equality of information matrices (White's test).

    Under correct specification, the negative Hessian and the outer
    product of gradients should be asymptotically equal. Significant
    differences indicate misspecification.

    H0: Model is correctly specified
    H1: Model is misspecified

    Args:
        hessian: Hessian matrix at optimum
        outer_product_gradient: Σ_i g_i g_i' (outer product of score)
        n_observations: Number of observations

    Returns:
        Dictionary with test statistic and p-value
    """
    from scipy import stats

    n_params = hessian.shape[0]

    # Under H0, -H ≈ OPG
    diff = -hessian - outer_product_gradient

    # Vectorize the difference matrix (lower triangle)
    diff_vec = []
    for i in range(n_params):
        for j in range(i + 1):
            diff_vec.append(diff[i, j].item())

    diff_vec = np.array(diff_vec)

    # Test statistic (simplified version)
    # Full implementation would require estimating variance of the difference
    test_stat = n_observations * np.sum(diff_vec**2)

    # Degrees of freedom: number of unique elements in symmetric matrix
    df = n_params * (n_params + 1) // 2

    p_value = 1 - stats.chi2.cdf(test_stat, df)

    return {
        "test_statistic": test_stat,
        "df": df,
        "p_value": p_value,
        "reject_at_05": p_value < 0.05,
    }


def diagnose_identification_issues(
    hessian: torch.Tensor,
    parameter_names: list[str],
    threshold: float = 0.01,
) -> list[str]:
    """Diagnose specific identification issues.

    Analyzes the Hessian structure to identify which parameters
    may be causing identification problems.

    Args:
        hessian: Hessian matrix at optimum
        parameter_names: Names of parameters
        threshold: Threshold for considering an eigenvalue as zero

    Returns:
        List of diagnostic messages
    """
    messages = []
    n_params = len(parameter_names)

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(-hessian)

    # Check for near-zero eigenvalues
    near_zero_mask = eigenvalues.abs() < threshold
    n_near_zero = near_zero_mask.sum().item()

    if n_near_zero > 0:
        messages.append(
            f"Found {n_near_zero} near-zero eigenvalue(s), indicating identification issues."
        )

        # Find which parameters are involved
        for i, is_near_zero in enumerate(near_zero_mask):
            if is_near_zero:
                # The eigenvector shows which parameters are in the null space
                ev = eigenvectors[:, i].abs()
                top_params = torch.argsort(ev, descending=True)[:3]

                involved = [parameter_names[j] for j in top_params]
                messages.append(
                    f"  Near-zero eigenvalue {i+1} involves: {', '.join(involved)}"
                )

    # Check diagonal elements (individual parameter curvature)
    diag = torch.diag(-hessian)
    for i, name in enumerate(parameter_names):
        if diag[i] < threshold:
            messages.append(
                f"Parameter '{name}' has near-zero curvature (flat likelihood)."
            )

    # Check for high correlations
    try:
        var_cov = torch.linalg.inv(-hessian)
        std_devs = torch.sqrt(torch.diag(var_cov))
        corr_matrix = var_cov / (std_devs.unsqueeze(0) * std_devs.unsqueeze(1))

        for i in range(n_params):
            for j in range(i + 1, n_params):
                if abs(corr_matrix[i, j]) > 0.95:
                    messages.append(
                        f"High correlation ({corr_matrix[i, j]:.3f}) between "
                        f"'{parameter_names[i]}' and '{parameter_names[j]}'."
                    )
    except RuntimeError:
        messages.append("Could not compute correlations (Hessian not invertible).")

    if not messages:
        messages.append("No identification issues detected.")

    return messages
