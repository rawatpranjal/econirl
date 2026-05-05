# TD-CCP

## Overview

TD-CCP estimates dynamic discrete choice models by combining the CCP likelihood
with temporal-difference estimates of the recursive value terms. It does not
estimate or use transition densities for structural parameter estimation. The
implementation learns \(h(a,x)\) and \(g(a,x)\) from observed
state-action-next-state tuples, then optimizes the CCP pseudo-likelihood.

Known transitions are only needed after estimation if you want final policy,
value, or counterfactual evaluation.

## When to Use TD-CCP

Use TD-CCP when:

- choices are discrete and forward-looking;
- rewards are finite-dimensional and linear in known features;
- state features may be flexible, including neural state encodings;
- transition-density modeling is the bottleneck;
- you have panel trajectories with current and next state-action pairs;
- you want structural parameters from a CCP-style estimator.

Avoid TD-CCP when the observed policy has very sparse action support, the reward
features are weakly identified, or you need raw nonparametric neural reward
recovery from choices alone.

## Basic Usage

```python
import pandas as pd

from econirl.estimators import TDCCP

data = pd.read_csv("zurcher_bus.csv")

model = TDCCP(
    n_states=90,
    n_actions=2,
    discount=0.9999,
    utility="linear_cost",
    method="semigradient",
    basis_type="encoded",
)
model.fit(data, state="mileage_bin", action="replaced", id="bus_id")

print(model.params_)
print(model.summary())
```

For custom reward features or lower-level control over the dynamic discrete
choice problem, use `econirl.estimation.TDCCPEstimator` directly.

## Validation Status

TD-CCP passes the package known-truth gates on `canonical_low_action`.

TD-CCP also passes the paper-faithful hard flexible DGP
`shapeshifter_linear_reward_neural_features`: stochastic shapeshifter
transitions, frozen neural state features, and a finite linear structural
reward with an action-0 normalization. This is the hard-case showcase that
matches the TD-CCP paper's finite-\(\theta\) setup.

The `canonical_high_action` encoded-state stress cell currently runs but fails
the same structural recovery gates. Treat the high-dimensional path as
diagnostic, not fully migrated.

The raw `shapeshifter_neural_neural` diagnostic is retained as a failure
artifact. It has a frozen neural reward matrix and no finite true
\(\theta\), so it should not be described as raw reward recovery by TD-CCP.

The reference PDF contains the validation tables, estimator settings, gate
audit, counterfactual checks, and generated JSON details.

## Further Reading

- Reference PDF: [TD-CCP reference and known-truth validation tutorial](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp.pdf)
- Estimator source: [`src/econirl/estimation/td_ccp.py`](https://github.com/rawatpranjal/EconIRL/blob/main/src/econirl/estimation/td_ccp.py)
- Result generator: [`papers/econirl_package/primers/tdccp/tdccp_run.py`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_run.py)
- Generated JSON: [`papers/econirl_package/primers/tdccp/tdccp_results.json`](https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/tdccp/tdccp_results.json)

Regenerate the TD-CCP validation artifacts from the repository root with:

```bash
PYTHONPATH=src:. python papers/econirl_package/primers/tdccp/tdccp_run.py --quiet-progress
```
