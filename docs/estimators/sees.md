# SEES

## Overview

SEES estimates structural dynamic discrete choice models without solving a
nested fixed point at every likelihood evaluation. It approximates the value
function with a sieve and estimates the reward parameters and value-function
approximation together.

In practice, SEES is useful when you want a structural estimator like NFXP, but
the repeated dynamic-programming solve is too expensive for the model you want
to fit.

## When to Use SEES

Use SEES when:

- choices are discrete and forward-looking;
- rewards are parametric;
- transitions are known or can be estimated in a first stage;
- the state space is moderate or can be represented with usable state features;
- you want structural parameters, value functions, and counterfactual policies.

Avoid SEES when the reward is not parameterized, the transition process is not
credible, the state representation is too rough for a smooth value-function
approximation, or you only need a predictive policy model rather than a
structural economic estimate.

## Basic Usage

The high-level wrapper accepts a pandas DataFrame with state, action, and unit
identifier columns:

```python
import pandas as pd

from econirl.estimators import SEES

data = pd.read_csv("zurcher_bus.csv")

model = SEES(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
)
model.fit(data, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

For custom reward features or lower-level control over the dynamic discrete
choice problem, use `econirl.estimation.SEESEstimator` directly.

## Validation Status

SEES passes the package known-truth checks on both the low-dimensional and
high-dimensional synthetic DGPs. The high-dimensional validation uses encoded
state features, so it tests the state-feature path rather than only a tabular
state index.

The reference PDF contains the validation tables, enforced gates,
counterfactual checks, estimator settings, and generated JSON details.

## Further Reading

- Reference PDF: [SEES reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees.pdf)
- Estimator source: [`src/econirl/estimation/sees.py`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/sees.py)
- Result generator: [`papers/econirl_package/primers/sees/sees_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees_run.py)
- Generated JSON: [`papers/econirl_package/primers/sees/sees_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/sees/sees_results.json)

From the repository root, regenerate the SEES validation artifacts with:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/sees/sees_run.py --quiet-progress
```
