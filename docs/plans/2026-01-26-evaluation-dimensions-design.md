# Evaluation Dimensions Design

**Date**: 2026-01-26
**Status**: Draft

## Overview

Separate evaluation into three conceptually distinct dimensions:

1. **Inference** — How well do we recover the true reward parameters θ?
2. **Prediction** — How well does the learned policy match behavior in the *same* environment?
3. **Generalization** — How well does the learned policy match behavior in a *different* environment?

This distinction matters because:
- Imitation learning can achieve good prediction but fails at generalization
- IRL methods claim to learn transferable rewards — generalization tests this claim
- Different use cases prioritize different dimensions

## Module Structure

```
src/econirl/evaluation/
├── __init__.py          # Re-exports main functions
├── inference.py         # Parameter recovery metrics
├── prediction.py        # In-environment policy metrics
├── generalization.py    # Cross-environment policy metrics
└── utils.py             # Shared helpers (policy computation, etc.)
```

## API Design

### Inference Metrics

```python
@dataclass
class InferenceMetrics:
    mse: float                    # Mean squared error
    rmse: float                   # Root MSE
    mae: float                    # Mean absolute error
    bias: torch.Tensor            # Per-parameter bias (θ̂ - θ*)
    correlation: float            # Pearson correlation
    cosine_similarity: float      # Direction similarity (scale-invariant)
    relative_error: torch.Tensor  # Per-parameter |θ̂ - θ*| / |θ*|
    coverage_90: float | None     # 90% CI coverage (if SEs provided)
    coverage_95: float | None     # 95% CI coverage (if SEs provided)


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
        mask: Which parameters to include in metrics
        normalize: Normalize both vectors to unit norm before comparison

    Returns:
        InferenceMetrics with all computed values
    """
```

### Prediction Metrics

```python
@dataclass
class PredictionMetrics:
    accuracy: float              # Fraction of actions correctly predicted (modal)
    log_likelihood: float        # Sum of log P(observed action | state)
    avg_log_likelihood: float    # Per-observation log-likelihood
    policy_kl_divergence: float | None  # KL(π* || π̂) if true policy provided
    action_probabilities: torch.Tensor  # P(a_observed | s) for each observation


def prediction_metrics(
    theta: torch.Tensor,
    panel: Panel,
    problem: DDCProblem,
    transitions: torch.Tensor,
    utility: UtilityFunction,
    true_policy: torch.Tensor | None = None,
) -> PredictionMetrics:
    """Compute in-environment prediction metrics.

    Args:
        theta: Learned parameters
        panel: Held-out test trajectories (same environment as training)
        problem: DDC problem specification
        transitions: Transition matrix (same as training)
        utility: Utility function specification
        true_policy: Ground truth policy (for KL divergence)

    Returns:
        PredictionMetrics with all computed values
    """
```

### Generalization Metrics

```python
@dataclass
class GeneralizationMetrics:
    accuracy: float              # Fraction correct under new dynamics
    log_likelihood: float        # Log P(observed | state) under new env
    avg_log_likelihood: float    # Per-observation
    policy_kl_divergence: float | None  # KL(π*_new || π̂_new)

    # Comparison to prediction (performance drop)
    accuracy_delta: float | None
    log_likelihood_delta: float | None


def generalization_metrics(
    theta: torch.Tensor,
    panel_new_env: Panel,
    problem: DDCProblem,
    transitions_new: torch.Tensor,
    utility: UtilityFunction,
    true_policy_new: torch.Tensor | None = None,
    prediction_baseline: PredictionMetrics | None = None,
) -> GeneralizationMetrics:
    """Compute cross-environment generalization metrics.

    Args:
        theta: Learned parameters (trained on different environment)
        panel_new_env: Expert demonstrations under new transitions
        problem: DDC problem specification
        transitions_new: New transition matrix (different from training)
        utility: Utility function specification
        true_policy_new: Ground truth policy under new transitions
        prediction_baseline: Prediction metrics for delta computation

    Returns:
        GeneralizationMetrics with all computed values
    """
```

## Panel.train_test_split

Add to `Panel` class:

```python
def train_test_split(
    self,
    test_frac: float = 0.2,
    by: Literal["trajectory", "time"] = "trajectory",
    seed: int | None = None,
) -> tuple[Panel, Panel]:
    """Split panel into train and test sets.

    Args:
        test_frac: Fraction of data for test set
        by: Split strategy
            - "trajectory": Random trajectories go to test (default)
            - "time": Last N% of each trajectory goes to test
        seed: Random seed for reproducibility

    Returns:
        (train_panel, test_panel)
    """
```

## Notebook: evaluation_dimensions_nfxp.ipynb

Demonstrates all three dimensions using NFXP on Rust bus:

1. **Introduction** — Explain the three dimensions
2. **Setup** — Define θ* and two environments (P₁, P₂ with faster deterioration)
3. **Generate Data** — Expert demos under both environments
4. **Estimation** — Fit NFXP on P₁ training data
5. **Inference Evaluation** — Compare θ̂ to θ*, visualize recovery
6. **Prediction Evaluation** — Accuracy on held-out P₁ data
7. **Generalization Evaluation** — Accuracy on P₂ data using θ̂
8. **Summary** — Compare all three dimensions

## Test Structure

```
tests/
├── test_evaluation_inference.py      # Parameter recovery metrics
├── test_evaluation_prediction.py     # In-sample policy metrics
├── test_evaluation_generalization.py # Cross-environment metrics
├── test_panel_split.py               # train_test_split functionality
└── integration/
    └── test_evaluation_nfxp.py       # End-to-end with NFXP
```

## Implementation Order

1. `src/econirl/evaluation/utils.py` — Policy computation helper
2. `src/econirl/evaluation/inference.py` — Simplest, no policy needed
3. `Panel.train_test_split()` — Needed for prediction testing
4. `src/econirl/evaluation/prediction.py` — Uses utils
5. `src/econirl/evaluation/generalization.py` — Builds on prediction
6. Tests for each module
7. Notebook demonstrating all three

## Design Decisions

- **Stateless functions** over methods on estimators (works with any method)
- **Dataclass returns** for named, typed access to metrics
- **Optional ground truth** — Inference requires θ*, but prediction/generalization can work without true policy
- **Generalization reuses prediction** — Same computation, different transitions
- **Two split strategies** — By trajectory (standard) or by time (temporal holdout)
