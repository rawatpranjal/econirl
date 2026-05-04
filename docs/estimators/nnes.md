# NNES

**Reference PDF:** [NNES reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes.pdf)

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. The method
targets the same structural objects as NFXP, CCP, MPEC, and SEES: reward
parameters, policy, value function, and counterfactual policies.

## Current Validation Status

NNES is validated in the shared known-truth harness. The canonical run uses
`canonical_low_action`, with 2,000
simulated individuals, 80 periods per individual, 21 states, 3 actions, known
transitions, an exit-action normalization, and action-dependent reward
features.

The estimator trains the NNES value network and optimizes the structural
parameters through the profiled one-step NPL objective. The generated PDF runs
with enforced hard gates.

| Metric | Value | Gate |
| --- | ---: | ---: |
| Final V-network loss | 0.000399 | <= 0.05 |
| Parameter cosine similarity | 0.998240 | >= 0.95 |
| Parameter relative RMSE | 0.065179 | <= 0.30 |
| Reward RMSE | 0.010210 | <= 0.08 |
| Policy total variation | 0.005646 | <= 0.03 |
| Value RMSE | 0.019845 | <= 0.20 |
| Q RMSE | 0.023370 | <= 0.20 |

NNES also passes the high-dimensional action-dependent preset
`canonical_high_action` after the high-dimensional reward basis is conditioned
for structural recovery. That run uses 16-dimensional state features, 32
reward features, and 160,000 observations.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/nnes/nnes_run.py
```

The generator writes `nnes_results.tex`, `nnes_results.json`, and a full JSON
copy under `/tmp/econirl_nnes_primer_known_truth`.

## Implementation

- Estimator: `src/econirl/estimation/nnes.py`
- Known-truth harness: `experiments/known_truth.py`
- Fast validation tests: `tests/test_known_truth.py`
