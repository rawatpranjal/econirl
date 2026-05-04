# NNES

**Reference PDF:** [NNES reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes.pdf)

NNES estimates structural dynamic discrete choice models with a neural
value-function approximation inside an NPL-style policy iteration. In this
package, NNES is used when the model is structural, transitions are known or
estimated up front, and the state space is large enough that repeatedly solving
the exact dynamic program is unattractive.

The important question is not only whether the likelihood improves. We test
whether NNES recovers the true reward, policy, value function, Q function, and
counterfactual behavior in a synthetic DGP where the truth is known exactly.

## Status

NNES now passes the enforced known-truth gates in both validation settings:

- an easy low-dimensional structural DGP;
- a high-dimensional structural DGP with 16 state features and 32 reward
  features.

Both runs use 2,000 simulated individuals and 80 periods per individual, for
160,000 observations. Both use known transitions, 3 actions, homogeneous
rewards, and the same Type A/B/C counterfactual checks.

| Validation cell | What it checks | Result |
| --- | --- | ---: |
| `canonical_low_action` | Small structural model with 21 states and 4 reward parameters | 11 / 11 gates pass |
| `canonical_high_action` | High-dimensional model with 81 states and 32 reward parameters | 11 / 11 gates pass |

The easy case is very tight. The high-dimensional case is harder and has larger
value and Q errors, but it stays inside every hard gate.

## Main Numbers

These are the numbers to look at first.

| Metric | Gate | Easy case | High-dimensional case |
| --- | ---: | ---: | ---: |
| Parameter cosine similarity | at least 0.95 | 0.9982 | 0.9912 |
| Parameter relative RMSE | at most 0.30 | 0.0652 | 0.1351 |
| Reward RMSE | at most 0.08 | 0.0102 | 0.0640 |
| Policy total variation | at most 0.03 | 0.0056 | 0.0238 |
| Value RMSE | at most 0.20 | 0.0198 | 0.1156 |
| Q RMSE | at most 0.20 | 0.0234 | 0.1371 |

The high-dimensional run is the more informative stress test. It has full
feature rank, a well-conditioned reward basis, all 81 states observed, and
state-action coverage of 0.959.

## Counterfactuals

NNES is also tested on three counterfactuals. The gate is regret below 0.05.

| Counterfactual | Easy case regret | High-dimensional regret | Interpretation |
| --- | ---: | ---: | --- |
| Type A | 0.0002 | 0.0049 | Recovers behavior after a reward or state shift |
| Type B | 0.0003 | 0.0056 | Recovers behavior after a transition change |
| Type C | 0.0001 | 0.0013 | Recovers behavior after restricting an action |

All three counterfactuals pass in both validation settings.

## What Is Being Validated

The known-truth harness compares the NNES output to the exact DGP solution.
The checks cover:

- reward parameters;
- reward values over state-action pairs;
- policy probabilities;
- value function and Q function;
- counterfactual regret and counterfactual policy distance.

There are also targeted component tests for the NNES algebra: policy
evaluation, the profiled value fixed point, the profiled Q/policy fixed point,
the theta-dependent continuation term, anchor normalization, and the
high-dimensional fixed point.

## Reproduce

From the repository root:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/nnes/nnes_run.py --quiet-progress
PYTHONPATH=src:. python experiments/known_truth.py --cell-id canonical_high_action --estimator NNES --output-dir /tmp/econirl_nnes_status_high
PYTHONPATH=src:. pytest tests/test_nnes_known_truth_components.py tests/test_known_truth.py -v
```

The primer generator writes `nnes_results.tex`, `nnes_results.json`, and a full
JSON copy under `/tmp/econirl_nnes_primer_known_truth`. The tables above are
rounded for readability; the JSON file keeps the exact values.

## Code Pointers

- Estimator: `src/econirl/estimation/nnes.py`
- Known-truth harness: `experiments/known_truth.py`
- Component tests: `tests/test_nnes_known_truth_components.py`
- Shared known-truth tests: `tests/test_known_truth.py`
