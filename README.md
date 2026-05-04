# econirl

Structural dynamic discrete choice and inverse reinforcement learning tools,
with estimator validation built around synthetic data-generating processes where
the truth is known exactly.

The current development focus is a shared **known-truth validation harness**.
For each estimator, the harness generates a DGP with known rewards,
transitions, policies, values, Q functions, and Type A/B/C counterfactual
oracles. An estimator is treated as migrated only after it recovers the relevant
truth objects and passes hard gates.

**Docs:** https://econirl.readthedocs.io/

## Install

```bash
pip install econirl
```

## Try It

Load a bundled dataset and fit one structural estimator.

```python
from econirl.datasets import load_rust_bus
from econirl import CCP

df = load_rust_bus()
model = CCP(n_states=90, discount=0.9999)
model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
print(model.params_)
print(model.summary())
```

## What Is Validated Now

The structural-estimator pages below have been migrated to the shared
known-truth workflow. Each page is a short front door to a generated PDF and
the exact run artifacts.

| Estimator | Role | Current validation status |
| --- | --- | --- |
| [NFXP](https://econirl.readthedocs.io/en/latest/estimators/nfxp.html) | Exact nested fixed-point MLE | Low-dimensional structural reference |
| [CCP / NPL](https://econirl.readthedocs.io/en/latest/estimators/ccp.html) | Hotz-Miller inversion plus NPL updates | Low-dimensional structural recovery |
| [MPEC](https://econirl.readthedocs.io/en/latest/estimators/mpec.html) | Constrained likelihood formulation | Low-dimensional structural recovery |
| [SEES](https://econirl.readthedocs.io/en/latest/estimators/sees.html) | Sieve value approximation | Low-dimensional structural recovery |
| [NNES](https://econirl.readthedocs.io/en/latest/estimators/nnes.html) | Neural NPL value approximation | Low-dimensional sanity check plus high-dimensional primary validation |

NNES is the first migrated page where the high-dimensional DGP is central to
the story: it passes both the easy low-dimensional cell and the 81-state,
32-reward-parameter high-dimensional cell.

## Known-Truth Harness

The harness lives in `experiments/known_truth.py`. It checks more than
in-sample choice fit:

- structural parameter recovery when the estimator has structural parameters;
- reward, value, Q, and policy recovery;
- Type A reward/state-shift counterfactuals;
- Type B transition-change counterfactuals;
- Type C action-restriction counterfactuals;
- compatibility with low-dimensional, high-dimensional, and latent-segment
  synthetic DGPs where appropriate.

Useful commands:

```bash
PYTHONPATH=src:. pytest tests/test_known_truth.py -v
PYTHONPATH=src:. python papers/econirl_package/primers/nnes/nnes_run.py --quiet-progress
```

## Estimators In The Repo

These are the main estimator families currently exposed or wired into the
known-truth migration plan.

| Family | Estimators |
| --- | --- |
| Structural econometrics | NFXP, CCP / NPL, MPEC, SEES, NNES, TD-CCP |
| Entropy and feature-matching IRL | MCE-IRL, neural MCE-IRL, MaxEnt IRL, Deep MaxEnt IRL, Bayesian IRL |
| Margin and distribution matching | Max Margin IRL, Max Margin Planning, f-IRL |
| Neural / Q-based methods | GLADIUS, Neural GLADIUS, IQ-Learn |
| Adversarial IRL | AIRL, Neural AIRL, AIRL-Het, GAIL, GCL |
| Baselines and utilities | Behavioral cloning, transition estimation, Rust bus replication tools |

The migration status is intentionally not the same as "code exists." Some
estimators are already validated in the shared known-truth framework; others
are present in the package and are being brought into that framework
estimator-by-estimator.

## Package Surface

The recommended public API is sklearn-style:

```python
from econirl import (
    NFXP,
    CCP,
    SEES,
    NNES,
    TDCCP,
    MCEIRL,
    MaxEntIRL,
    MaxMarginIRL,
    GLADIUS,
    NeuralGLADIUS,
    AIRL,
    NeuralAIRL,
    IQLearn,
    MCEIRLNeural,
)
```

Lower-level estimator implementations remain available under
`econirl.estimation` and `econirl.contrib` for research workflows that need
direct access to configuration objects or diagnostics. For example, the MPEC
implementation is currently available as `econirl.estimation.mpec.MPECEstimator`
while its public docs page is already part of the known-truth validation set.
The lower-level namespace also includes f-IRL and behavioral cloning, and
`econirl.contrib` keeps the older MaxEnt, Deep MaxEnt, max-margin, GAIL, GCL,
and Bayesian IRL implementations available for comparison work.

## Reference Pages

- Estimator index: https://econirl.readthedocs.io/en/latest/estimators.html
- NNES page: https://econirl.readthedocs.io/en/latest/estimators/nnes.html
- NNES PDF: https://github.com/rawatpranjal/EconIRL/blob/main/papers/econirl_package/primers/nnes/nnes.pdf

## License

MIT
