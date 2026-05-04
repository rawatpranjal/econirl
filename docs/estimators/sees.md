# SEES

**Reference PDF:** [SEES reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees.pdf)

SEES estimates structural dynamic discrete choice models by approximating the
value function with a deterministic sieve and penalizing Bellman residuals.
It solves jointly for reward parameters and sieve coefficients, so there is no
nested Bellman fixed point inside each likelihood evaluation.

Use SEES when the reward is parametric, transitions are known or first-stage
estimated, and the value function is smooth enough for a deterministic basis.
It is a structural estimator, not a behavioral cloning shortcut.

## Status

SEES now passes the enforced known-truth gates in both validation settings:

- an easy low-dimensional structural DGP using the historical state-index
  B-spline basis;
- a high-dimensional structural DGP with 16 encoded state features and 32
  reward features, using the encoded-state basis.

Both runs use 2,000 simulated individuals and 80 periods per individual, for
160,000 observations. Both use known transitions, 3 actions, homogeneous
rewards, finite Schur-complement standard errors, and Type A/B/C
counterfactual checks.

| Validation cell | What it checks | Basis | Result |
| --- | --- | --- | ---: |
| `canonical_low_action` | Small structural model with 21 states and 4 reward parameters | state index, `K=21` | 11 / 11 gates pass |
| `canonical_high_action` | High-dimensional model with 81 states and 32 reward parameters | encoded state, `K=81` | 11 / 11 gates pass |

The high-dimensional run is the primary validation. It confirms that SEES uses
`problem.state_encoder` rather than treating the 81 state labels as a
one-dimensional index.

## Main Numbers

| Metric | Gate | Low-dimensional | High-dimensional |
| --- | ---: | ---: | ---: |
| Bellman violation | at most 0.05 | 0.000058 | 0.000003 |
| Parameter cosine similarity | at least 0.99 | 0.999146 | 0.999955 |
| Parameter relative RMSE | at most 0.15 | 0.059671 | 0.009528 |
| Reward RMSE | at most 0.03 | 0.008988 | 0.004432 |
| Policy total variation | at most 0.02 | 0.005179 | 0.002117 |
| Value RMSE | at most 0.10 | 0.017591 | 0.037836 |
| Q RMSE | at most 0.10 | 0.020514 | 0.031480 |

The high-dimensional run uses a full-rank encoded-state RBF/SVD basis with
`basis_dim=81` and Bellman penalty `10000.0`. The low-dimensional sanity run
keeps the B-spline index basis with `basis_dim=21` and penalty `100.0`.

## Counterfactuals

SEES is also tested on three counterfactuals. The gate is regret below `0.01`.

| Counterfactual | Low-dimensional regret | High-dimensional regret | Interpretation |
| --- | ---: | ---: | --- |
| Type A | 0.000179 | 0.000113 | Recovers behavior after a reward or state shift |
| Type B | 0.000299 | 0.000183 | Recovers behavior after a transition change |
| Type C | 0.000071 | 0.000014 | Recovers behavior after restricting an action |

All three counterfactuals pass in both validation settings.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/sees/sees_run.py --quiet-progress
PYTHONPATH=src:. python -m experiments.known_truth --cell-id canonical_high_action --estimator SEES --output-dir /tmp/econirl_sees_status_high
PYTHONPATH=src:. pytest tests/test_sees_known_truth_components.py tests/test_known_truth.py -v
```

The primer generator writes `sees_results.tex`, `sees_results.json`, and a full
JSON copy under `/tmp/econirl_sees_primer_known_truth`. The tables above are
rounded for readability; the JSON file keeps the exact values.

## Code Pointers

- Estimator: `src/econirl/estimation/sees.py`
- Known-truth harness: `experiments/known_truth.py`
- Component tests: `tests/test_sees_known_truth_components.py`
- Shared known-truth tests: `tests/test_known_truth.py`
