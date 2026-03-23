"""Evaluation metrics for IRL and DDC estimation.

This module provides metrics for three distinct evaluation dimensions:
- Inference: Parameter recovery (comparing theta-hat to theta-star)
- Prediction: In-environment policy accuracy
- Generalization: Cross-environment policy accuracy
"""

from econirl.evaluation.inference import InferenceMetrics, inference_metrics
from econirl.evaluation.adapters import build_utility_for_estimator, project_state_features
from econirl.evaluation.benchmark import (
    BenchmarkDGP,
    BenchmarkResult,
    EstimatorSpec,
    get_default_estimator_specs,
    run_benchmark,
    run_single,
    summarize_benchmark,
)
from econirl.evaluation.convergence import ConvergenceProfile, track_convergence

__all__ = [
    "InferenceMetrics",
    "inference_metrics",
    # Adapters
    "build_utility_for_estimator",
    "project_state_features",
    # Benchmark
    "BenchmarkDGP",
    "BenchmarkResult",
    "EstimatorSpec",
    "get_default_estimator_specs",
    "run_benchmark",
    "run_single",
    "summarize_benchmark",
    # Convergence
    "ConvergenceProfile",
    "track_convergence",
]
