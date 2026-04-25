# Plan: ziebart-small IRL canonical run

## Headline claim

On Ziebart's 10-by-10 gridworld with five actions, a known step cost
of -0.1, and a known terminal reward of +10, MCE-IRL recovers the
true reward weights to relative error below 5 percent and produces a
policy whose KL divergence against the soft-optimal policy under the
true reward is below 0.01. AIRL recovers the same reward up to an
additive constant, with the constant absorbed by the
identification-anchored normalization. IQ-Learn recovers an
approximate reward through the inverse Bellman operator with policy
KL below 0.05. Behavioral cloning matches the expert state-action
distribution but fails on transfer to perturbed grid layouts, which
is the documented failure mode of imitation without dynamics.

## Estimators in scope

Four estimators: MCE-IRL, AIRL, IQ-Learn, BC. This is the only place
in the paper where the IRL family is benchmarked against itself
rather than against the structural family. The structural estimators
do not appear here because the dataset has no structural utility
parameters to recover; the only quantity worth measuring is the
recovered reward.

## Inference validity classification

| Estimator | SE method | Validity |
| --- | --- | --- |
| MCE-IRL  | inverse Hessian on dual objective | valid under tabular linear features |
| AIRL     | bootstrap over individuals        | bootstrap-supported |
| IQ-Learn | none                               | unavailable |
| BC       | inverse Fisher                    | valid for the imitation likelihood |

## Metrics reported per fit

For every replication of every estimator:

- Recovered reward weights theta_hat (length equal to the feature dimension).
- Reward identification residual: the L2 norm of theta_hat minus its closest affine transformation of the true theta. Replaces cosine similarity as the headline reward-recovery metric. A residual of zero means the recovered reward equals the true reward up to scale and additive constant, which is the only thing that can be claimed under the standard IRL identification result.
- Reward angular error: the angle between theta_hat and theta_true after subtracting the best constant. In radians.
- Policy KL: KL divergence between the soft-optimal policy under theta_hat and the soft-optimal policy under theta_true, averaged over the visited state distribution.
- Percent optimal value: the value of the recovered policy as a fraction of the value of the true optimal policy under the true reward, on a 1000-trajectory rollout from the same initial-state distribution.
- Out-of-sample log-likelihood from 5-fold CV over expert trajectories.
- Wall-clock time.

## Monte Carlo extension

R = 50 per cell. 200 fits total. Each replication regenerates the
expert trajectories under the true reward with seed 42 + r so the
Monte Carlo isolates the estimator's variance from the data.

The Monte Carlo summary adds:

- Mean and standard deviation of the reward identification residual per estimator.
- Mean and standard deviation of policy KL per estimator.
- Mean and standard deviation of percent optimal value per estimator.
- Convergence rate per estimator.
- Median wall-clock per estimator.

## Compute spec

CPU. 4 cells, R = 50 each, 200 fits total. Median fit time around 30
seconds. Total single-pod wall-clock around 1.7 hours, parallelizable
across 4 pods to under 30 minutes. Cost around 1 USD on RunPod CPU
pods.

This plan is the cheapest of the five and is recommended as the
second smoke test after plan_rust_small.

## Acceptance criteria

The plan succeeds when each of the following holds.

- MCE-IRL converges on at least 95 percent of replications and reports reward identification residual below 0.05 and policy KL below 0.01.
- AIRL converges on at least 80 percent of replications and reports reward identification residual below 0.10 and policy KL below 0.05.
- IQ-Learn converges on at least 95 percent of replications (no inner optimization to fail) and reports policy KL below 0.10.
- BC reports the lowest in-sample log-likelihood among the four estimators (it is fit directly to the action distribution) but fails on a transfer test to a perturbed grid (the obstacle layout is shifted by one cell). The transfer-test policy KL exceeds 0.30.

## Failure modes documented in the artifact

- `none`: as in plan_rust_small.
- `did_not_converge`: optimizer did not return converged.
- `reward_drift`: reward identification residual exceeds the threshold even though the optimizer converged.
- `policy_drift`: policy KL exceeds the threshold even though the reward looks reasonable.
- `transfer_failure`: BC transfer-test policy KL exceeds 0.30. This is the expected behavior for BC and is documented as a positive result, not a bug.

## What the paper says now

Until this plan fires, Section 4.2 (the IRL canonical subsection) is
a placeholder paragraph: "Results for Ziebart's gridworld are
forthcoming. The methodology, hardware, and acceptance criteria are
documented in plans/plan_ziebart_small.md." No tables. No figures.
No claims about MCE-IRL recovering the reward until it has been
measured.

## Artifact paths

- `experiments/jss_deep_run/results/plan_ziebart_small/<cell_id>.csv` per cell.
- `experiments/jss_deep_run/results/plan_ziebart_small_summary.csv` Monte Carlo summary.
- `experiments/jss_deep_run/results/plan_ziebart_small_failures.csv` documented failure rows.
- `papers/econirl_package_jss/figures/fig_irl_canonical.tex` the LaTeX table for Section 4.2.

## Run command

The plan does not currently have its own tier in the dispatcher
matrix; it maps to a subset of Tier 3 cells once they are wired up.
For now the local dry-run path is:

```
python -m experiments.jss_deep_run.dispatch_runpod --local --tier 3
```

with the cell registry filtering on `dataset == "ziebart-small"`.
