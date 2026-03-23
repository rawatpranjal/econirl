"""Inference metrics for parameter recovery evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InferenceMetrics:
    """Metrics for evaluating parameter recovery (inference).

    Attributes:
        mse: Mean squared error between theta-hat and theta-star
        rmse: Root mean squared error
        mae: Mean absolute error
        bias: Per-parameter bias (theta-hat - theta-star)
        correlation: Pearson correlation coefficient
        cosine_similarity: Cosine similarity (direction match)
        relative_error: Per-parameter |theta-hat - theta-star| / |theta-star|
        coverage_90: Fraction of parameters where 90% CI contains theta-star
        coverage_95: Fraction of parameters where 95% CI contains theta-star
    """

    mse: float
    rmse: float
    mae: float
    bias: torch.Tensor
    correlation: float
    cosine_similarity: float
    relative_error: torch.Tensor
    coverage_90: float | None = None
    coverage_95: float | None = None


def inference_metrics(
    theta_true: torch.Tensor,
    theta_hat: torch.Tensor,
    standard_errors: torch.Tensor | None = None,
    mask: list[bool] | None = None,
    normalize: bool = False,
) -> InferenceMetrics:
    """Compute parameter recovery metrics.

    Args:
        theta_true: Ground truth parameters
        theta_hat: Estimated parameters
        standard_errors: Standard errors (for coverage computation)
        mask: Which parameters to include (True = include)
        normalize: Normalize both vectors to unit norm before comparison

    Returns:
        InferenceMetrics with all computed values
    """
    # Ensure tensors
    theta_true = torch.as_tensor(theta_true, dtype=torch.float32)
    theta_hat = torch.as_tensor(theta_hat, dtype=torch.float32)

    # Apply mask if provided
    if mask is not None:
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        theta_true = theta_true[mask_tensor]
        theta_hat = theta_hat[mask_tensor]
        if standard_errors is not None:
            standard_errors = standard_errors[mask_tensor]

    # Optionally normalize
    if normalize:
        theta_true = theta_true / theta_true.norm()
        theta_hat = theta_hat / theta_hat.norm()

    # Bias (per-parameter)
    bias = theta_hat - theta_true

    # MSE, RMSE, MAE
    mse = (bias**2).mean().item()
    rmse = mse**0.5
    mae = bias.abs().mean().item()

    # Relative error (handle zeros in theta_true)
    relative_error = torch.where(
        theta_true.abs() > 1e-10,
        (theta_hat - theta_true).abs() / theta_true.abs(),
        torch.zeros_like(theta_true),
    )

    # Correlation (Pearson)
    # Use unbiased=False to be consistent with mean() which divides by N
    if len(theta_true) > 1:
        mean_true = theta_true.mean()
        mean_hat = theta_hat.mean()
        cov = ((theta_true - mean_true) * (theta_hat - mean_hat)).mean()
        std_true = theta_true.std(unbiased=False)
        std_hat = theta_hat.std(unbiased=False)
        if std_true > 1e-10 and std_hat > 1e-10:
            correlation = (cov / (std_true * std_hat)).item()
        else:
            correlation = 1.0 if torch.allclose(theta_true, theta_hat) else 0.0
    else:
        correlation = 1.0 if torch.allclose(theta_true, theta_hat) else 0.0

    # Cosine similarity
    norm_true = theta_true.norm()
    norm_hat = theta_hat.norm()
    if norm_true > 1e-10 and norm_hat > 1e-10:
        cosine_similarity = (
            torch.dot(theta_true, theta_hat) / (norm_true * norm_hat)
        ).item()
    else:
        cosine_similarity = 1.0 if torch.allclose(theta_true, theta_hat) else 0.0

    # Coverage (if standard errors provided)
    coverage_90 = None
    coverage_95 = None
    if standard_errors is not None:
        standard_errors = torch.as_tensor(standard_errors, dtype=torch.float32)
        # z-scores for 90% and 95% CI
        z_90 = 1.645
        z_95 = 1.96

        # Check if true value is within CI
        lower_90 = theta_hat - z_90 * standard_errors
        upper_90 = theta_hat + z_90 * standard_errors
        covered_90 = (theta_true >= lower_90) & (theta_true <= upper_90)
        coverage_90 = covered_90.float().mean().item()

        lower_95 = theta_hat - z_95 * standard_errors
        upper_95 = theta_hat + z_95 * standard_errors
        covered_95 = (theta_true >= lower_95) & (theta_true <= upper_95)
        coverage_95 = covered_95.float().mean().item()

    return InferenceMetrics(
        mse=mse,
        rmse=rmse,
        mae=mae,
        bias=bias,
        correlation=correlation,
        cosine_similarity=cosine_similarity,
        relative_error=relative_error,
        coverage_90=coverage_90,
        coverage_95=coverage_95,
    )
