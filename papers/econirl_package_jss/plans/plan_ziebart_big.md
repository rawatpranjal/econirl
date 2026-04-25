# Plan: ziebart-big IRL scalability run

## Headline claim

On a 50-by-50 stochastic gridworld with eight actions, slip
probability 0.1, and a 16-dimensional radial-basis-function reward,
deep MCE-IRL with a neural reward parametrization recovers the true
RBF coefficients to L2 error below 0.10 and produces a policy whose
percent optimal value is at least 0.90 of the optimal-policy value.
Tabular MCE-IRL on the same panel converges to the same recovered
reward but takes at least 5 times longer wall-clock on CPU. AIRL on
this panel produces a discriminator-based reward whose policy KL
against the optimal policy is below 0.10 at the cost of two orders
of magnitude more wall-clock than the deep MCE-IRL variant.

## Estimators in scope

Three estimators: deep MCE-IRL (with `reward_type='neural'`), tabular
MCE-IRL (with the default linear reward over the RBF basis), and
AIRL. The plan does not include IQ-Learn because the soft-Q tabular
representation does not scale to 2500 states with eight actions in a
useful comparison.

## Inference validity classification

| Estimator | SE method | Validity |
| --- | --- | --- |
| Deep MCE-IRL    | bootstrap over trajectories                | bootstrap-supported |
| Tabular MCE-IRL | inverse Hessian on the dual objective       | valid under linear features in the basis |
| AIRL            | bootstrap over trajectories                | bootstrap-supported |

## Metrics reported per fit

For every replication of every estimator:

- Recovered RBF coefficient vector (length 16).
- L2 error of the recovered RBF coefficients against the true RBF coefficients (declared in the dataset metadata JSON), after the standard IRL affine normalization.
- Policy KL averaged over the visited state distribution.
- Percent optimal value on a 5000-trajectory rollout under the recovered policy and the true reward.
- Out-of-sample log-likelihood from 5-fold CV over expert trajectories.
- Wall-clock time.
- GPU memory peak.
- Inner-loop iteration count for tabular MCE-IRL (the bottleneck).

## Monte Carlo extension

R = 20 per cell. 60 fits total. Each replication regenerates the
expert trajectories under the same true reward with seed 42 + r.

The Monte Carlo summary adds:

- Mean and standard deviation of L2 reward error per estimator.
- Mean and standard deviation of policy KL per estimator.
- Mean and standard deviation of percent optimal value per estimator.
- Median wall-clock per estimator.
- Speedup ratio: median CPU wall-clock divided by median GPU wall-clock for deep MCE-IRL.
- Convergence rate per estimator.

## Compute spec

GPU. 3 cells, R = 20 each, 60 fits total. Expected per-fit time on
GPU: deep MCE-IRL 300 seconds, tabular MCE-IRL 1200 seconds, AIRL
1200 seconds. Total single-pod wall-clock around 15 hours; at 3-way
RunPod parallelism the wall-clock floors at the longer cells around
6.7 hours. Cost around 8 USD on RunPod A100 pods.

For a first iteration we also support R = 5, which cuts the wall-clock
to under 2 hours and costs around 2 USD. Recommended for pipeline
validation.

## Acceptance criteria

The plan succeeds when each of the following holds.

- Deep MCE-IRL converges on at least 90 percent of replications, reports L2 reward error below 0.10, and reports percent optimal value at least 0.90.
- Tabular MCE-IRL converges on at least 90 percent of replications and reports L2 reward error and percent optimal value within 5 percent of the deep variant.
- AIRL converges on at least 70 percent of replications and reports policy KL below 0.10.
- The CPU-versus-GPU speedup ratio for deep MCE-IRL is at least 3.
- The wall-clock ratio between tabular MCE-IRL and deep MCE-IRL on GPU is at least 3 (the deep variant is meaningfully faster).

## Failure modes documented in the artifact

- `none`: as in plan_rust_small.
- `did_not_converge`: optimizer did not return converged.
- `reward_drift`: L2 reward error exceeds the threshold.
- `policy_drift`: policy KL exceeds the threshold.
- `oom`: GPU memory peak exceeded the available memory; the cell was killed and is reported separately.
- `runtime_exceeded`: wall-clock exceeded the per-cell ceiling.

## What the paper says now

Until this plan fires, Section 4.3 IRL-scaling subsection is a
placeholder paragraph: "Results for the large stochastic gridworld
are forthcoming. The methodology, hardware, and acceptance criteria
are documented in plans/plan_ziebart_big.md." No tables. No
figures. No claim that the deep variant beats the tabular variant
until it has been measured.

## Artifact paths

- `experiments/jss_deep_run/results/plan_ziebart_big/<cell_id>.csv` per cell.
- `experiments/jss_deep_run/results/plan_ziebart_big_summary.csv` Monte Carlo summary.
- `experiments/jss_deep_run/results/plan_ziebart_big_failures.csv` documented failure rows.
- `papers/econirl_package_jss/figures/fig_irl_scaling.tex` the LaTeX table for the IRL-scaling subsection.

## Run command

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 3b --image econirl-deep-run:v1 --max-parallel 3
```

For the local R = 5 dry run (CPU; expect tabular MCE-IRL to time out
under the 30-minute ceiling):

```
python -m experiments.jss_deep_run.dispatch_runpod --local --tier 3b
```
