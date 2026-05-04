# MPEC

**Reference PDF:** [MPEC reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/mpec/mpec.pdf)

MPEC, mathematical programming with equilibrium constraints, estimates the
same structural dynamic discrete choice likelihood as NFXP but treats the
Bellman fixed point as an explicit equality constraint. The optimizer solves
jointly for reward parameters and the value function.

Use MPEC when the state-action space is tabular, transitions are known or
first-stage estimated, the Bellman constraint dimension is moderate, and you
want a constrained-optimization check on the NFXP likelihood.

## Current Validation Status

The reference PDF documents the full known-truth synthetic DGP validation. No
real data are used. The canonical validation cell is `canonical_low_action`,
with 2,000 simulated individuals, 80 periods per individual, 21 states, 3
actions, known transitions, an exit-action normalization, and action-dependent
reward features.

| Metric | Value | Gate |
| --- | ---: | ---: |
| Bellman constraint violation | 7.72e-12 | <= 1e-6 |
| Parameter cosine similarity | 0.998867 | >= 0.98 |
| Parameter relative RMSE | 0.065378 | <= 0.15 |
| Policy total variation | 0.005697 | <= 0.03 |
| Value RMSE | 0.019445 | <= 0.10 |
| Q RMSE | 0.022437 | <= 0.10 |

The same run also solves Type A, Type B, and Type C counterfactuals against
known oracle objects and recovers all three with small policy error and regret.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python -m experiments.known_truth \
    --estimator MPEC \
    --cell-id canonical_low_action \
    --output-dir outputs/known_truth \
    --show-progress \
    --verbose
```

The full PDF contains the model math, identification checks, DGP design,
estimator settings, recovery tables, counterfactual results, and practical
debugging notes.

## Implementation

- Estimator: `src/econirl/estimation/mpec.py`
- Known-truth harness: `experiments/known_truth.py`
- Fast validation tests: `tests/test_known_truth.py`
