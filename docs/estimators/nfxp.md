# NFXP

**Reference PDF:** [NFXP reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nfxp/nfxp.pdf)

NFXP, the nested fixed point estimator, is the reference estimator for
structural dynamic discrete choice. It estimates primitive reward parameters by
solving the agent's dynamic programming problem inside the likelihood.

Use NFXP when the state space is tabular, transitions are known or first-stage
estimated, the reward is parameterized by a low-dimensional vector, and exact
dynamic programming is feasible.

## Current Validation Status

The reference PDF documents the full known-truth synthetic DGP validation. No
real data are used. The canonical validation cell is `canonical_low_action`,
with 2,000 simulated individuals, 80 periods per individual, 21 states, 3
actions, known transitions, an exit-action normalization, and action-dependent
reward features.

| Metric | Value | Gate |
| --- | ---: | ---: |
| Parameter cosine similarity | 0.998867 | >= 0.98 |
| Parameter relative RMSE | 0.065378 | <= 0.15 |
| Policy total variation | 0.005697 | <= 0.03 |
| Value RMSE | 0.019445 | <= 0.10 |

The same run also solves Type A, Type B, and Type C counterfactuals against
known oracle objects and recovers all three with small policy error and regret.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python -m experiments.known_truth \
    --estimator NFXP \
    --cell-id canonical_low_action \
    --output-dir outputs/known_truth \
    --show-progress \
    --verbose
```

The full PDF contains the model math, identification checks, DGP design,
estimator settings, recovery tables, counterfactual results, and practical
debugging notes.

## Implementation

- Estimator: `src/econirl/estimation/nfxp.py`
- Known-truth harness: `experiments/known_truth.py`
- Fast validation tests: `tests/test_known_truth.py`
