# Plan: lsw-synthetic heterogeneity run

## Headline claim

On the semi-synthetic mirror of the Lee-Sudhir-Wang 2026 serialized
fiction reading panel, AIRL-Het with two latent types recovers the
mixture weights within 5 percentage points of the data-generating
process weights of (0.4, 0.6) on at least 80 percent of replications,
and the per-type reward parameters with relative error below 15
percent. Homogeneous AIRL on the same panel produces a
population-average reward whose per-type log-likelihood is at least
0.30 worse per observation than AIRL-Het, demonstrating that
modeling unobserved heterogeneity recovers signal that the
homogeneous variant cannot. MCE-IRL fit without latent types
performs no better than homogeneous AIRL on per-type predictions.
Behavioral cloning achieves the lowest in-sample log-likelihood but
fails to predict the type-conditional choice patterns on held-out
users.

## Estimators in scope

Four estimators: AIRL-Het (with `num_segments=2`), homogeneous AIRL,
MCE-IRL, and BC. The plan exists to demonstrate that heterogeneity
matters when the DGP has it. The contrast against homogeneous
estimators is the headline.

## Inference validity classification

| Estimator | SE method | Validity |
| --- | --- | --- |
| AIRL-Het | bootstrap over users; EM standard errors via the observed-information matrix at the converged E-step | bootstrap-supported |
| AIRL     | bootstrap over users                              | bootstrap-supported |
| MCE-IRL  | inverse Hessian on the dual objective             | valid under tabular linear features |
| BC       | inverse Fisher                                    | valid for the imitation likelihood |

## Metrics reported per fit

For every replication of every estimator:

- Recovered mixture weights pi_hat (length equal to num_segments). Trivially length one for the homogeneous estimators.
- Recovered per-type reward parameter vectors theta_hat. Length two for AIRL-Het, length one for the others.
- L2 error of the recovered mixture weights against the DGP weights (0.4, 0.6).
- L2 error of the recovered per-type reward parameters against the DGP per-type alpha_pay and alpha_wait, after the standard IRL affine normalization within each type.
- Per-type policy KL: KL divergence between the type-conditional policy under the recovered reward and the type-conditional policy under the DGP, averaged over the visited state distribution.
- Type assignment accuracy: fraction of users whose Maximum-A-Posteriori type label from the AIRL-Het E-step matches the true latent type. Reported only for AIRL-Het.
- In-sample log-likelihood and out-of-sample log-likelihood from a 5-fold CV over users.
- Per-type out-of-sample log-likelihood difference: log-likelihood under AIRL-Het minus log-likelihood under the homogeneous estimator on the same held-out users, decomposed by true type.
- Wall-clock time and EM iteration count for AIRL-Het.

## Monte Carlo extension

R = 20 per cell. 80 fits total. Each replication regenerates the
synthetic panel under the same DGP with seed 42 + r so the Monte
Carlo isolates the estimator's variance.

The Monte Carlo summary adds:

- Mean and standard deviation of the mixture-weight L2 error for AIRL-Het.
- Mean and standard deviation of the type assignment accuracy for AIRL-Het.
- Mean and standard deviation of per-type policy KL per estimator.
- Mean and standard deviation of per-type log-likelihood difference for AIRL-Het versus the homogeneous baselines.
- Convergence rate per estimator.
- Median wall-clock per estimator.

## Compute spec

GPU. 4 cells, R = 20 each, 80 fits total. Expected per-fit time on
GPU: AIRL-Het 600 seconds (the EM outer loop is the long pole), AIRL
300 seconds, MCE-IRL 120 seconds, BC 10 seconds. Total single-pod
wall-clock around 5.7 hours; at 4-way RunPod parallelism the
wall-clock floors at the AIRL-Het cell around 3.3 hours. Cost around
6 USD on RunPod A100 pods.

For a first iteration we also support R = 3, which cuts the wall-clock
to under 1 hour and costs around 1 USD. Recommended for pipeline
validation.

## Acceptance criteria

The plan succeeds when each of the following holds.

- AIRL-Het converges on at least 80 percent of replications.
- AIRL-Het reports mixture-weight L2 error below 0.10 (5 percentage points per weight) on at least 80 percent of replications.
- AIRL-Het reports per-type reward parameter relative error below 15 percent on at least 70 percent of replications.
- AIRL-Het reports type assignment accuracy above 0.75 on at least 80 percent of replications.
- Homogeneous AIRL and MCE-IRL converge on at least 80 percent of replications. Their per-type log-likelihood is at least 0.30 worse per observation than AIRL-Het on average.
- BC achieves the lowest in-sample log-likelihood. Its per-type out-of-sample log-likelihood is at least 0.50 worse per observation than AIRL-Het, demonstrating that BC does not model the latent structure.

## Failure modes documented in the artifact

- `none`: as in plan_rust_small.
- `did_not_converge`: outer EM loop did not reach the convergence tolerance within the iteration cap.
- `degenerate_mixture`: recovered mixture weights collapsed to a single type. AIRL-Het returned a homogeneous solution.
- `type_swap`: recovered types match the DGP after permutation; flagged but not counted as failure (the artifact records the permutation).
- `reward_drift`: per-type reward parameter relative error exceeds the threshold.
- `runtime_exceeded`: wall-clock exceeded the per-cell ceiling. Expected occasionally for AIRL-Het because of EM outer-loop variability.

## What the paper says now

Until this plan fires, Section 4.4 (the heterogeneity centerpiece)
is a placeholder paragraph: "Results for the semi-synthetic mirror
of the Lee-Sudhir-Wang panel are forthcoming. The methodology,
hardware, and acceptance criteria are documented in
plans/plan_lsw_synthetic.md." No tables. No figures. No claim that
AIRL-Het recovers the mixture until it has been measured.

## Artifact paths

- `experiments/jss_deep_run/results/plan_lsw_synthetic/<cell_id>.csv` per cell.
- `experiments/jss_deep_run/results/plan_lsw_synthetic_summary.csv` Monte Carlo summary.
- `experiments/jss_deep_run/results/plan_lsw_synthetic_failures.csv` documented failure rows.
- `papers/econirl_package_jss/figures/fig_heterogeneity.tex` the LaTeX table for Section 4.4.

## Run command

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 3c --image econirl-deep-run:v1 --max-parallel 4
```

For the local R = 3 dry run (CPU; expect AIRL-Het to take 30+ minutes
per replication and possibly hit the per-cell ceiling):

```
python -m experiments.jss_deep_run.dispatch_runpod --local --tier 3c
```
