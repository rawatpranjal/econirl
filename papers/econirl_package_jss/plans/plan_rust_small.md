# Plan: rust-small validation run

## Headline claim

On the canonical Rust 1987 bus panel, the classical structural
estimators (NFXP, CCP, MPEC) recover the operating cost and
replacement cost parameters within Monte Carlo standard error of the
data-generating process, and their asymptotic standard errors
achieve close to the nominal 95 percent coverage rate over R = 100
replications. The approximate estimators (NNES, SEES, TD-CCP) recover
the same parameters with documented bias and slower convergence. The
IRL estimators (MCE-IRL, AIRL, IQ-Learn, f-IRL) recover an
identification-anchored reward whose policy KL against the optimal
policy is small. Behavioral cloning achieves the lowest policy KL
in-sample but fails on counterfactual replacement-cost shifts, which
is the precise sense in which it is a baseline rather than a
structural estimator.

## Estimators in scope

All twelve production estimators run on this dataset: NFXP, CCP,
MPEC, MCE-IRL, NNES, SEES, TD-CCP, GLADIUS, AIRL, IQ-Learn, f-IRL,
and BC. This is the only plan in which all twelve appear together
because rust-small is the only panel small enough that even the
slowest estimator fits in a few minutes per replication.

## Inference validity classification

Per the cross-cutting convention. Reported alongside every standard
error in the artifact CSV.

| Estimator | SE method | Validity |
| --- | --- | --- |
| NFXP     | inverse Hessian (asymptotic)              | valid asymptotic under regularity |
| CCP      | inverse Hessian on NPL pseudo-likelihood  | valid asymptotic under regularity |
| MPEC     | inverse augmented Hessian                 | valid asymptotic under regularity |
| MCE-IRL  | inverse Hessian on dual objective         | valid under tabular linear |
| NNES     | inverse Hessian on penalized likelihood   | bootstrap-supported |
| SEES     | inverse Hessian on penalized likelihood   | bootstrap-supported |
| TD-CCP   | cross-fitting variance estimator          | bootstrap-supported |
| GLADIUS  | influence-function variance               | heuristic |
| AIRL     | bootstrap                                  | bootstrap-supported |
| IQ-Learn | none                                       | unavailable |
| f-IRL    | none                                       | unavailable |
| BC       | inverse Fisher information                | valid for the imitation likelihood |

## Metrics reported per fit

For every replication (one row per fit per estimator):

- Point estimate per parameter (theta_1 and RC).
- Standard error per parameter, plus the SE method label.
- t-statistic and 95 percent confidence interval, where SE is available.
- In-sample log-likelihood.
- Out-of-sample log-likelihood from 5-fold cross-validation over individuals.
- Policy KL: KL divergence between the estimated logit choice probabilities and the optimal logit choice probabilities under the true parameters, averaged over visited states.
- Replacement-probability RMSE: root mean square error of the estimated replace probability against the truth, computed at every visited mileage state.
- Value loss: absolute error between the estimated and true value functions at the modal initial state (mileage bin 0).
- Counterfactual error: absolute error in the predicted replacement boundary (defined as the mileage at which replace probability crosses 0.5) under a 25 percent reduction in RC. The reference is the boundary computed from the true parameters.
- Wall-clock time and converged flag.
- Relative parameter error per parameter (estimated minus true, divided by absolute true). Replaces cosine similarity as the headline accuracy metric.

## Monte Carlo extension

R = 100 replications under the bundled synthetic ground truth (theta_1
= 0.001, RC = 3.0). Each replication uses seed 42 + r and resamples
the synthetic panel from the same DGP. The Monte Carlo summary CSV
adds:

- Bias per parameter (mean across replications minus true value).
- RMSE per parameter (square root of mean squared error across replications).
- 95 percent CI coverage per parameter (fraction of replications where the asymptotic CI contains the true value). Targets 0.95 with a tolerance window of [0.90, 0.99].
- Convergence rate (fraction of replications that returned converged = True).

## Compute spec

CPU. 12 cells, R = 100 each, 1200 fits total. Median fit time around
60 seconds, AIRL is the long pole at 600 seconds per fit. Total
single-pod wall-clock around 12 hours; at 8-way RunPod parallelism
the wall-clock floors at the AIRL cell at roughly 17 hours. Cost
around 6 USD on RunPod CPU pods.

For a first iteration we also support R = 20, total 240 fits, total
wall-clock 4 hours sequential or 30 minutes at 8-way parallelism, cost
around 1 USD. Use R = 20 to validate the pipeline; rerun at R = 100
once the pipeline is solid.

## Acceptance criteria

The plan succeeds when each of the following holds.

- NFXP, CCP, and MPEC each return converged = True on at least 95 percent of replications.
- NFXP, CCP, and MPEC each report relative parameter error below 5 percent on both theta_1 and RC.
- NFXP and CCP each achieve 95 percent CI coverage on theta_1 in [0.90, 0.99] over R = 100 replications.
- MCE-IRL relative parameter error matches NFXP within 5 percent on both parameters.
- NNES, SEES, TD-CCP each report relative parameter error below 10 percent on both parameters.
- AIRL converges on at least 80 percent of replications (allowing for the documented adversarial instability) and achieves policy KL below 0.05.
- IQ-Learn returns finite parameters on every replication and achieves policy KL below 0.10.
- f-IRL is allowed to fail; its row in the artifact carries a documented failure mode.
- BC achieves the lowest in-sample log-likelihood; the artifact reports its out-of-sample log-likelihood and counterfactual error so the failure mode under counterfactual is visible.

## Failure modes documented in the artifact

The artifact CSV has a `failure_mode` column. Allowed values:

- `none`: estimator converged and the headline metrics are within thresholds.
- `did_not_converge`: optimizer did not return converged = True.
- `parameter_drift`: relative parameter error exceeds the threshold even though the optimizer converged.
- `policy_drift`: policy KL exceeds the threshold even though parameters look reasonable.
- `inference_unsupported`: estimator does not support the requested SE method.
- `runtime_exceeded`: wall-clock exceeded the per-cell ceiling.

## What the paper says now

Until this plan fires, Section 4.1 of the paper is a single
placeholder paragraph: "Results for the canonical Rust 1987 panel
are forthcoming. The methodology, hardware, and acceptance criteria
are documented in plans/plan_rust_small.md." No tables. No figures.
A `\todo[inline]{Run plan_rust_small.md before drafting}` marker.
Section 4.4 (the cross-estimator equivalence table) is similarly
placeheld and will be regenerated from the per-replication CSV by
`papers/econirl_package_jss/code_snippets/table3_benchmark_all.py`.

After the plan fires the prose is written from the artifact and the
table is generated by the script. No values are typed by hand.

## Artifact paths

- `experiments/jss_deep_run/results/plan_rust_small/<cell_id>.csv` per cell, R rows each.
- `experiments/jss_deep_run/results/plan_rust_small_summary.csv` one row per estimator, Monte Carlo summary.
- `experiments/jss_deep_run/results/plan_rust_small_failures.csv` one row per (estimator, replication) where failure_mode is not `none`.
- `papers/econirl_package_jss/figures/table3_rust_small_benchmark.tex` the LaTeX table consumed by Section 4.4.

## Run command

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 2 --image econirl-deep-run:v1 --max-parallel 8
```

For the local R = 20 dry run:

```
python -m experiments.jss_deep_run.dispatch_runpod --local --tier 2
```
