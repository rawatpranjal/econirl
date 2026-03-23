"""Tests for inference (parameter recovery) metrics."""

import pytest
import torch

from econirl.evaluation.inference import InferenceMetrics, inference_metrics


class TestInferenceMetrics:
    """Test inference_metrics function."""

    def test_perfect_recovery(self):
        """Perfect parameter recovery should give zero MSE."""
        theta_true = torch.tensor([1.0, 2.0, 3.0])
        theta_hat = torch.tensor([1.0, 2.0, 3.0])

        result = inference_metrics(theta_true, theta_hat)

        assert isinstance(result, InferenceMetrics)
        assert result.mse == pytest.approx(0.0)
        assert result.rmse == pytest.approx(0.0)
        assert result.mae == pytest.approx(0.0)
        assert result.correlation == pytest.approx(1.0)
        assert result.cosine_similarity == pytest.approx(1.0)

    def test_known_mse(self):
        """Test MSE computation with known values."""
        theta_true = torch.tensor([1.0, 2.0])
        theta_hat = torch.tensor([1.0, 3.0])  # Error of 1 on second param

        result = inference_metrics(theta_true, theta_hat)

        # MSE = (0^2 + 1^2) / 2 = 0.5
        assert result.mse == pytest.approx(0.5)
        assert result.rmse == pytest.approx(0.5**0.5)
        assert result.mae == pytest.approx(0.5)  # (0 + 1) / 2

    def test_bias_computation(self):
        """Test per-parameter bias."""
        theta_true = torch.tensor([1.0, 2.0, 3.0])
        theta_hat = torch.tensor([1.5, 1.5, 3.5])

        result = inference_metrics(theta_true, theta_hat)

        expected_bias = torch.tensor([0.5, -0.5, 0.5])
        assert torch.allclose(result.bias, expected_bias)

    def test_normalize_option(self):
        """Test that normalize=True normalizes before comparison."""
        theta_true = torch.tensor([1.0, 0.0])
        theta_hat = torch.tensor([2.0, 0.0])  # Same direction, different scale

        result = inference_metrics(theta_true, theta_hat, normalize=True)

        # After normalization, both are [1, 0], so cosine similarity = 1
        assert result.cosine_similarity == pytest.approx(1.0)

    def test_mask_option(self):
        """Test that mask excludes parameters from metrics."""
        theta_true = torch.tensor([1.0, 2.0, 100.0])  # Third param is way off
        theta_hat = torch.tensor([1.0, 2.0, 0.0])

        # Without mask: huge MSE
        result_no_mask = inference_metrics(theta_true, theta_hat)
        assert result_no_mask.mse > 1000

        # With mask: exclude third param
        result_masked = inference_metrics(
            theta_true, theta_hat, mask=[True, True, False]
        )
        assert result_masked.mse == pytest.approx(0.0)

    def test_coverage_with_standard_errors(self):
        """Test confidence interval coverage computation."""
        theta_true = torch.tensor([1.0, 2.0])
        theta_hat = torch.tensor([1.1, 2.1])
        se = torch.tensor([0.5, 0.5])  # Large SEs should contain true values

        result = inference_metrics(theta_true, theta_hat, standard_errors=se)

        # With SE=0.5, 95% CI is approximately +/- 1 (1.96 * 0.5)
        # True values are within 0.1, so should be covered
        assert result.coverage_95 == 1.0  # Both covered
