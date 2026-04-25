# Plan: rust-big failure-and-recovery run

## Headline claim

When the Rust bus panel is augmented with 30 dummy state variables of
cardinality 20 per the Kang-Yoganarasimhan-Jain 2025 high-dimensional
extension, tabular methods (NFXP and CCP) fail cleanly through the
package's convergence flag and identification diagnostics, and
neural-approximation methods (GLADIUS, NNES, TD-CCP) recover the
parameters on the genuine mileage dimension within 10 percent of the
rust-small reference. The dummies, which are independent of the
action by construction, do not leak into the recovered reward.

## Estimators in scope

Five estimators: NFXP, CCP, GLADIUS, NNES, TD-CCP. The first two are
expected to fail; the last three are expected to recover. The plan
documents both behaviors and treats the failure of NFXP and CCP as a
positive result rather than a bug.

## Inference validity classification

| Estimator | SE method | Validity |
| --- | --- | --- |
| NFXP     | inverse Hessian | valid asymptotic when fit converges; SE not reported when convergence fails |
| CCP      | inverse Hessian on NPL | valid asymptotic when fit converges; SE not reported when convergence fails |
| GLADIUS  | influence-function variance on the recovered Q-function | heuristic |
| NNES     | inverse Hessian on penalized likelihood | bootstrap-supported |
| TD-CCP   | cross-fitting variance estimator | bootstrap-supported |

## Metrics reported per fit

For every replication of every estimator:

- Convergence flag and convergence reason string. The reason
  string distinguishes Bellman-residual divergence, optimizer
  iteration cap, out-of-memory, and successful convergence.
- Memory peak in megabytes.
- Recovered theta_1 and RC on the mileage dimension. Recovered
  coefficients on the dummy dimensions, reported as a vector. The
  dummy coefficients should hover near zero by construction.
- Relative parameter error of theta_1 and RC against the rust-small
  reference (0.001, 3.0).
- Out-of-sample log-likelihood from 5-fold CV over individuals.
- Policy RMSE on the genuine mileage state dimension, integrated
  over the dummies under their empirical marginal.
- Wall-clock time.

The dummy-coefficient vector is summarized into one scalar per fit:
the L2 norm of the dummy coefficients. The headline check is that
the L2 norm of the dummies stays below 5 percent of the L2 norm of
the genuine-mileage coefficient.

## Monte Carlo extension

R = 20 per cell. 100 fits total. The dummies are redrawn per
replication so each replication tests both genuine recovery and dummy
robustness against a new noise realization.

The Monte Carlo summary adds:

- Convergence rate per estimator. NFXP and CCP are expected at
  most 0.10; the headline is that they converge rarely. GLADIUS,
  NNES, TD-CCP are expected above 0.90.
- Median memory peak per estimator.
- Median wall-clock per estimator.
- Bias and RMSE on theta_1 and RC for the three recovering
  estimators. Not reported for NFXP and CCP because their convergence
  rate is too low for a meaningful summary.
- Median dummy L2 norm per recovering estimator.

## Compute spec

GPU. 5 cells, R = 20 each, 100 fits total. Expected per-fit time:
NFXP and CCP timeout at the 30-minute per-cell ceiling on at least 75
percent of replications, GLADIUS at 120 seconds, NNES at 180 seconds,
TD-CCP at 300 seconds. Total single-pod wall-clock around 40 hours
because of the timeout overhead on the failing cells. At 5-way RunPod
parallelism (one pod per cell) the wall-clock floors at the timeout
ceiling, around 10 hours of wall-clock for the failing cells running
in parallel. Cost around 12 USD on RunPod GPU pods.

For a first iteration we also support R = 5, which cuts the timeout
overhead down to 2 hours and costs around 3 USD. Recommended for
pipeline validation.

## Acceptance criteria

The plan succeeds when each of the following holds.

- NFXP and CCP each report converged = True on at most 25 percent of replications. The headline is that they fail; if they converge often something is wrong with the augmentation.
- GLADIUS converges on at least 90 percent of replications and reports relative parameter error on theta_1 below 10 percent and on RC below 10 percent against the rust-small reference.
- NNES and TD-CCP each converge on at least 80 percent of replications with the same parameter-error tolerance.
- For all three recovering estimators the dummy L2 norm stays below 5 percent of the genuine-coefficient L2 norm. The dummies do not leak into the recovered reward.
- Memory peak for NFXP exceeds 4 GB on at least one replication, demonstrating the tabular memory wall.

## Failure modes documented in the artifact

- `none`: as in plan_rust_small.
- `out_of_memory`: estimator allocated more than the per-cell memory ceiling and was killed. Expected for NFXP and CCP.
- `bellman_divergence`: inner Bellman solve did not contract. Expected for NFXP and CCP at this state-space size.
- `dummy_leakage`: dummy L2 norm exceeded the threshold. Genuine mileage parameters may still be correct but the recovered reward picked up noise.
- `did_not_converge`: optimizer did not return converged = True for reasons other than memory or Bellman divergence.

## What the paper says now

Until this plan fires, Section 4.3 (the failure-and-recovery
subsection) is a placeholder paragraph: "Results for the
high-dimensional Rust extension are forthcoming. The methodology,
hardware, and acceptance criteria are documented in
plans/plan_rust_big.md." No tables. No figures. No claims about
NFXP failing or GLADIUS succeeding until both have been measured.

## Artifact paths

- `experiments/jss_deep_run/results/plan_rust_big/<cell_id>.csv` per cell.
- `experiments/jss_deep_run/results/plan_rust_big_summary.csv` Monte Carlo summary.
- `experiments/jss_deep_run/results/plan_rust_big_failures.csv` documented failure rows.
- `papers/econirl_package_jss/figures/fig_failure_recovery.tex` the LaTeX table for Section 4.3.

## Run command

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 3a --image econirl-deep-run:v1 --max-parallel 5
```

For the local R = 5 dry run (CPU only, expect NFXP and CCP to time out):

```
python -m experiments.jss_deep_run.dispatch_runpod --local --tier 3a
```
