# SEES

**Reference PDF:** [SEES reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees.pdf)

SEES, sieve estimation of economic structural models, estimates structural
dynamic discrete choice models by approximating the value function with a
deterministic basis and penalizing Bellman residuals. The optimizer solves
jointly for reward parameters and sieve coefficients, so there is no nested
Bellman fixed point inside each likelihood evaluation.

Use SEES when the reward is parametric, transitions are known or first-stage
estimated, and the state space is smooth enough for a deterministic sieve to
approximate the value function. It is a structural estimator, not a behavioral
cloning shortcut.

## Current Validation Status

The reference PDF documents the full known-truth synthetic DGP validation. No
real data are used. The canonical validation cell is `canonical_low_action`,
with 2,000 simulated individuals, 80 periods per individual, 21 states, 3
actions, known transitions, an exit-action normalization, and action-dependent
reward features.

The validated run uses a B-spline basis with `basis_dim=21`, penalty weight
`100.0`, L-BFGS-B, and Schur-complement standard errors. The optimizer's
absolute-gradient convergence flag is reported honestly in the PDF; the hard
SEES gate is the structural Bellman residual plus known-truth recovery.

| Metric | Value | Gate |
| --- | ---: | ---: |
| Bellman violation | 0.044579 | <= 0.05 |
| Standard errors finite | true | true |
| Parameter cosine similarity | 0.995485 | >= 0.99 |
| Parameter relative RMSE | 0.124955 | <= 0.15 |
| Reward RMSE | 0.013579 | <= 0.03 |
| Policy total variation | 0.008884 | <= 0.02 |
| Value RMSE | 0.080710 | <= 0.10 |
| Q RMSE | 0.080546 | <= 0.10 |

The same run also solves Type A, Type B, and Type C counterfactuals against
known oracle objects and recovers all three with regret below `0.01`.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python -m experiments.known_truth \
    --estimator SEES \
    --cell-id canonical_low_action \
    --output-dir outputs/known_truth \
    --show-progress \
    --verbose
```

The full PDF contains the model math, identification checks, DGP design,
estimator settings, recovery tables, counterfactual results, and practical
debugging notes.

## Implementation

- Estimator: `src/econirl/estimation/sees.py`
- Known-truth harness: `experiments/known_truth.py`
- Fast validation tests: `tests/test_known_truth.py`
