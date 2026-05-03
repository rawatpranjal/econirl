# NFXP

| Category | Source papers | Reward | Transitions | Standard errors | Best scale |
| --- | --- | --- | --- | --- | --- |
| Structural dynamic discrete choice | Rust (1987); Iskhakov, Rust, Schjerning (2016) | Parametric, action-dependent | Known or first-stage estimated | Analytical MLE/BHHH | Small to medium tabular state spaces |

NFXP, the nested fixed point estimator, is the reference estimator for
structural dynamic discrete choice. It estimates primitive reward parameters by
solving the agent's dynamic programming problem inside the likelihood.

This tutorial is built around a synthetic DGP with known truth. No real data are
used. Because the DGP is known, we can check whether NFXP recovers the true
reward parameters, value function, policy, and Type A/B/C counterfactuals.

:::{important}
This page is a validation tutorial, not a real-data demo. The point is to show
that the estimator recovers ground truth when its identification assumptions
hold, and to make failures interpretable when they do not.
:::

## When to Use NFXP

Use NFXP when you have a tabular dynamic discrete choice model and need
publication-grade structural estimates:

- the reward is parameterized as a low-dimensional vector;
- transitions are known or can be estimated in a first stage;
- the state space is small enough that exact dynamic programming is feasible;
- you need likelihood values, standard errors, and counterfactual policies from
  recovered primitives.

NFXP is usually the wrong tool when the state is high-dimensional or continuous.
For those cases, look at NNES, SEES, GLADIUS, or TD-CCP, depending on whether
you want structural parameters, value-function approximation, or transition-free
estimation.

## Model

The observed data are state-action trajectories:

$$
\{(s_{it}, a_{it}, s_{i,t+1}) : i = 1,\ldots,N,\ t = 1,\ldots,T\}.
$$

The model has flow utility

$$
u_\theta(s,a) = \phi(s,a)^\top \theta,
$$

known transition probabilities

$$
P_a(s,s') = \Pr(s_{t+1}=s' \mid s_t=s, a_t=a),
$$

discount factor $\beta$, and type-I extreme-value shocks with scale $\sigma$.
The soft Bellman fixed point is

$$
V_\theta(s)
= \sigma \log \sum_a
\exp\left(
  \frac{
    u_\theta(s,a)
    + \beta \sum_{s'} P_a(s,s') V_\theta(s')
  }{\sigma}
\right).
$$

The choice-specific value is

$$
Q_\theta(s,a)
=
u_\theta(s,a)
+ \beta \sum_{s'} P_a(s,s')V_\theta(s'),
$$

and the implied conditional choice probability is

$$
\pi_\theta(a \mid s)
=
\frac{\exp(Q_\theta(s,a)/\sigma)}
{\sum_b \exp(Q_\theta(s,b)/\sigma)}.
$$

NFXP maximizes the conditional log-likelihood

$$
\hat{\theta}
=
\arg\max_\theta
\sum_{i,t} \log \pi_\theta(a_{it} \mid s_{it}).
$$

In this package, the default outer optimizer is BHHH and the default inner
solver is the hybrid successive-approximation plus Newton-Kantorovich
polyalgorithm. Pure value iteration is available, but the hybrid solver is the
right default when $\beta$ is high.

## Identification

The DGP must be identified before NFXP is judged.

Rust's original setup handles this in three ways.

First, the dynamic problem is reduced to a tractable Markov state by the
conditional independence assumption. The observed state transition does not
depend on the current logit shock, and the expected value function can be solved
on the observed state space rather than on the full observed-plus-unobserved
state.

Second, the transition process is treated separately from the reward problem.
In Rust's bus model the transition parameters are estimated by a separate
partial likelihood step. In this synthetic tutorial the transition tensor is
known exactly, so the NFXP run focuses on reward recovery.

Third, the reward is normalized. The shock scale is fixed at $\sigma = 1$, and
the synthetic DGP uses an exit action and absorbing state as a zero-reward
anchor. Without a scale and location normalization, reward levels are not
uniquely pinned down.

For the NFXP validation cell, the most important practical requirement is
action-dependent reward variation. State-only features copied across actions
collapse the likelihood surface for a structural theta estimator.

## Known-Truth DGP

The canonical validation cell is `canonical_low_action`. It is deliberately
small enough to solve exactly, but large enough to expose support and
identification failures.

| Quantity | Value |
| --- | ---: |
| Regular states | 20 |
| Absorbing states | 1 |
| Total states | 21 |
| Actions | 3 |
| Exit action | 2 |
| Discount factor | 0.95 |
| Shock scale | 1.0 |
| Reward mode | action-dependent |
| Reward dimension | low |
| State mode | low-dimensional |
| Simulated individuals | 2,000 |
| Periods per individual | 80 |
| Observations | 160,000 |

The action-dependent reward features are

$$
\phi(s,a)
=
\begin{cases}
(1, x_s, 0, 0) & a = 0,\\
(0, 0, 1, x_s) & a = 1,\\
(0, 0, 0, 0) & a = 2.
\end{cases}
$$

The exit action has zero reward and sends the process to the absorbing state.
The true parameter vector is

| Parameter | Truth |
| --- | ---: |
| `action_0_intercept` | 0.10 |
| `action_0_progress` | 0.50 |
| `action_1_intercept` | 0.00 |
| `action_1_progress` | -0.20 |

This is not the real Rust bus dataset. It is a known-truth structural DGP with
the same nested fixed-point logic and exact oracle objects.

## Pre-Estimation Checks

Run diagnostics before fitting. If these checks fail, the problem is usually the
DGP or feature design, not the optimizer.

```python
from experiments.known_truth import (
    SimulationConfig,
    build_known_truth_dgp,
    get_cell,
    run_pre_estimation_diagnostics,
    simulate_known_truth_panel,
)

cell = get_cell("canonical_low_action")
dgp = build_known_truth_dgp(cell.dgp_config)
panel = simulate_known_truth_panel(
    dgp,
    SimulationConfig(n_individuals=2000, n_periods=80, seed=42),
)
diagnostics = run_pre_estimation_diagnostics(dgp, panel)

assert diagnostics.passed
assert diagnostics.feature_rank == diagnostics.num_features
assert diagnostics.is_action_dependent
```

The validation run produced:

| Check | Value | Status |
| --- | ---: | --- |
| Feature rank | 4 / 4 | pass |
| Feature condition number | 4.512 | pass |
| Transition row error | 2.42e-08 | pass |
| Observed states | 21 / 21 | pass |
| State-action coverage | 1.000 | pass |
| Action shares | 0.345, 0.330, 0.325 | pass |
| Minimum action share | 0.325 | pass |
| Exit/absorbing anchor | true | pass |

The action shares are not cosmetic. A previous DGP generated almost no support
for one action. That is an identification problem: the likelihood cannot pin
down a payoff for an action that the agent almost never takes.

## Run the Known-Truth Tutorial

From the repository root:

```bash
PYTHONPATH=src:. python -m experiments.known_truth \
    --estimator NFXP \
    --cell-id canonical_low_action \
    --output-dir outputs/known_truth \
    --show-progress \
    --verbose
```

The command does the full validation path:

1. build the DGP;
2. solve the true Bellman problem;
3. simulate the panel with `tqdm` progress;
4. run pre-estimation diagnostics;
5. fit NFXP;
6. evaluate reward, value, Q, and policy recovery;
7. solve Type A, Type B, and Type C counterfactual oracles;
8. write one `result.json` artifact;
9. raise if any non-smoke hard gate fails.

NFXP itself does not need a GPU on this canonical tabular cell. The same module
command is used by RunPod workers so the estimator pages share one validation
interface.

You can also call the estimator directly:

```python
from experiments.known_truth import (
    build_known_truth_dgp,
    get_cell,
    known_truth_initial_params,
    simulate_known_truth_panel,
)
from econirl.estimation.nfxp import NFXPEstimator

cell = get_cell("canonical_low_action")
dgp = build_known_truth_dgp(cell.dgp_config)
panel = simulate_known_truth_panel(dgp, cell.simulation_config)

estimator = NFXPEstimator(
    optimizer="BHHH",
    inner_solver="hybrid",
    inner_tol=1e-12,
    inner_max_iter=100_000,
    outer_max_iter=500,
    compute_hessian=True,
    verbose=True,
)

summary = estimator.estimate(
    panel=panel,
    utility=dgp.utility(),
    problem=dgp.problem,
    transitions=dgp.transitions,
    initial_params=known_truth_initial_params(dgp),
)
```

## Estimation Results

Medium-scale run: `canonical_low_action`, 2,000 individuals, 80 periods, 160,000
observations.

| Quantity | Value |
| --- | ---: |
| Converged | true |
| Outer iterations | 12 |
| Log-likelihood | -174875.7719 |
| Estimation time | 5.28 seconds |
| Function evaluations | 80 |
| Total inner iterations | 11259 |
| Final inner iterations | 139 |
| Inner solver | hybrid |
| Outer optimizer | BHHH |

Parameter recovery:

| Parameter | Truth | Estimate | SE | Error |
| --- | ---: | ---: | ---: | ---: |
| `action_0_intercept` | 0.100000 | 0.083894 | 0.029335 | -0.016106 |
| `action_0_progress` | 0.500000 | 0.528522 | 0.035889 | 0.028522 |
| `action_1_intercept` | 0.000000 | -0.014461 | 0.036733 | -0.014461 |
| `action_1_progress` | -0.200000 | -0.200511 | 0.052502 | -0.000511 |

Recovery metrics:

| Metric | Value |
| --- | ---: |
| Parameter RMSE | 0.017904 |
| Parameter relative RMSE | 0.065378 |
| Parameter cosine similarity | 0.998867 |
| Reward RMSE | 0.009694 |
| Value RMSE | 0.019445 |
| Q RMSE | 0.022438 |
| Policy KL | 9.21e-05 |
| Policy total variation | 0.005697 |
| Policy max state L1 | 0.018905 |

Hard gates:

| Gate | Threshold | Value | Status |
| --- | ---: | ---: | --- |
| Converged | true | true | pass |
| Parameter cosine | >= 0.98 | 0.998867 | pass |
| Parameter relative RMSE | <= 0.15 | 0.065378 | pass |
| Policy TV | <= 0.03 | 0.005697 | pass |
| Value RMSE | <= 0.10 | 0.019445 | pass |

The estimates are not exactly equal to truth because the panel is finite. The
validation target is recovery within strict tolerances, not equality in one
finite sample.

## Counterfactuals

The tutorial evaluates three post-estimation counterfactuals.

| Type | Intervention | Purpose |
| --- | --- | --- |
| Type A | Shift the reward surface and hold transitions fixed | Payoff counterfactual |
| Type B | Change transitions and hold reward fixed | State-dynamics counterfactual |
| Type C | Disable one non-anchor action with a large penalty | Action-set/design counterfactual |

For each intervention, the harness solves the true counterfactual oracle. It
then solves the same intervention using the reward recovered by NFXP and
evaluates the resulting policy in the true counterfactual environment.

| Counterfactual | Policy TV | Policy KL | Value RMSE | Regret |
| --- | ---: | ---: | ---: | ---: |
| Type A | 0.005109 | 7.56e-05 | 0.000238 | 0.000213 |
| Type B | 0.005457 | 8.20e-05 | 0.000363 | 0.000362 |
| Type C | 0.003548 | 3.56e-05 | 0.000114 | 0.000086 |

These regrets are small because the recovered reward is close enough to the
true reward that re-solving the intervened model produces almost the same policy
as the oracle.

## Debugging Gaps

If NFXP fails to recover truth, debug in this order.

1. **Feature rank.** If rank is below the number of reward parameters, theta is
   not identified.
2. **Action support.** If one action is rarely observed, its payoff is weakly
   identified.
3. **Normalization.** If the exit/absorbing anchor is invalid, reward levels can
   drift.
4. **State-only rewards.** The canonical NFXP theta validation needs
   action-dependent features.
5. **Transition tensor.** NFXP needs stochastic transition rows and the right
   action-state-next-state orientation.
6. **Inner tolerance.** Use a tight tolerance such as `1e-10` to `1e-12` for
   structural validation.
7. **High beta.** Use `inner_solver="hybrid"` near $\beta = 1`; pure value
   iteration is intentionally slow in that regime.
8. **Soft gates.** Do not accept a non-smoke run that only warns. Known-truth
   validation should pass hard gates or raise.

## Practical Defaults

Recommended non-smoke validation settings:

```python
NFXPEstimator(
    optimizer="BHHH",
    inner_solver="hybrid",
    inner_tol=1e-12,
    inner_max_iter=100_000,
    switch_tol=1e-3,
    outer_tol=1e-6,
    outer_max_iter=500,
    compute_hessian=True,
)
```

Use smoke settings only for code-path tests. A smoke test shows that the
estimator runs and returns finite objects; it is not evidence of recovery.

## Reproducibility Paths

- Estimator: `src/econirl/estimation/nfxp.py`
- Known-truth harness: `experiments/known_truth.py`
- Fast tests: `tests/test_known_truth.py`
- Paper alignment audit: `papers/econirl_package_jss/plans/alignment/02_nfxp.md`
- Rust paper text: `papers/foundational/1987_rust_optimal_replacement.md`
- NK comparison paper:
  `papers/foundational/iskhakov_rust_schjerning_2016_mpec_comment.md`

## References

- Rust, J. (1987). Optimal Replacement of GMC Bus Engines: An Empirical Model
  of Harold Zurcher. *Econometrica*, 55(5), 999-1033.
- Iskhakov, F., Lee, J., Rust, J., Schjerning, B., and Seo, K. (2016). Comment
  on "Constrained Optimization Approaches to Estimation of Structural Models."
  *Econometrica*, 84(1), 365-370.
