# CCP

**Reference PDF:** [CCP reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/ccp/ccp.pdf)

CCP, the conditional choice probability estimator, estimates structural dynamic
discrete choice models by replacing repeated NFXP Bellman solves with
Hotz-Miller policy inversion. Empirical CCPs initialize the estimator, and NPL
iterations move the pseudo-likelihood mapping toward the MLE fixed point.

Use CCP when the state-action space is tabular, transitions are known or
first-stage estimated, action support is strong, and you want a structural
estimator that avoids an inner dynamic-programming solve at every likelihood
evaluation.

## Current Validation Status

The reference PDF documents the full known-truth synthetic DGP validation. No
real data are used. The canonical validation cell is `canonical_low_action`,
with 2,000 simulated individuals, 80 periods per individual, 21 states, 3
actions, known transitions, an exit-action normalization, and action-dependent
reward features.

| Metric | Value | Gate |
| --- | ---: | ---: |
| Parameter cosine similarity | 0.998867 | >= 0.98 |
| Parameter relative RMSE | 0.065372 | <= 0.15 |
| Policy total variation | 0.005697 | <= 0.03 |
| Value RMSE | 0.019438 | <= 0.10 |
| Q RMSE | 0.022432 | <= 0.10 |

The same run also solves Type A, Type B, and Type C counterfactuals against
known oracle objects and recovers all three with small policy error and regret.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python -m experiments.known_truth \
    --estimator CCP \
    --cell-id canonical_low_action \
    --output-dir outputs/known_truth \
    --show-progress \
    --verbose
```

The full PDF contains the model math, identification checks, DGP design,
estimator settings, recovery tables, counterfactual results, and practical
debugging notes.

## Implementation

- Estimator: `src/econirl/estimation/ccp.py`
- Known-truth harness: `experiments/known_truth.py`
- Fast validation tests: `tests/test_known_truth.py`
