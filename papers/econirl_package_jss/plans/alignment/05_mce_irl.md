## Estimator: MCE-IRL (Maximum Causal Entropy IRL, tabular linear)
## Paper(s): Ziebart 2008 (MaxEnt IRL); Ziebart 2010 thesis (Maximum Causal Entropy IRL — the soft Bellman version that matches DDC's logit shocks). Paper at `papers/foundational/2008_maximum_entropy_inverse_reinforcement_learning.pdf`. The 2010 thesis is the more relevant reference but is not in the foundational/ directory.
## Code: `src/econirl/estimation/mce_irl.py`

### Loss / objective

- Paper formula (Ziebart 2010, Algorithm 1; the dual MCE objective): minimize over `theta`

  ```
  L_MCE(theta) = log Z(theta) - <theta, mu_E>
  ```

  where `mu_E` is the empirical feature expectation and `log Z(theta) = sigma * V(s_0; theta)` is the partition function under the soft Bellman fixed-point. The gradient is

  ```
  d_theta L = mu_pi(theta) - mu_E
  ```

  the difference between the policy-induced expected feature count and the empirical one. At the optimum, the moments match (Ziebart 2010, Proposition 1).

- Code implementation: `mce_irl.py:_optimize` constructs the dual loss and runs gradient descent (or L-BFGS) with the gradient computed via `_compute_expected_features`. Per the project root CLAUDE.md, the expected feature is computed using **occupancy measures** (state visitation frequencies under the current policy):

  ```
  E_pi[phi] = sum_s D_pi(s) sum_a pi(a|s) phi(s, a, k)
  ```

  This matches Ziebart 2010 Algorithm 1 and the imitation library's MCE-IRL implementation.

- Match: **yes** for the dual loss and gradient. The implementation follows imitation's tabular MCE-IRL where action-dependent features are passed in.

### Gradient

- Paper formula: `mu_pi - mu_E`.

- Code implementation: `mce_irl.py:_compute_expected_features`. For tabular linear, this is the closed-form occupancy times features; for the deep variant, it is a Monte Carlo estimate from rollout trajectories.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: per outer iteration, solve the soft Bellman fixed-point under the current `theta` to compute the policy `pi(a|s)`, then compute occupancy `D_pi`. Both are inner sub-problems; both have linear-system closed forms in the tabular case.

- Code algorithm: `mce_irl.py` inner loop calls a soft VI solver (matches `core/solvers.py:value_iteration` with the same scale parameter). The occupancy is computed from the policy via the `core/occupancy.py:solve_occupancy_measure` direct linear solve (borrowed from imitation per CLAUDE.md).

- Match: **yes**.

### Identification assumptions

- Paper conditions: features must be **action-dependent** (varying across actions for at least some states), per Ziebart 2010 Section 3.2. With state-only features, `phi(s, a) = phi(s)` for all a, and the moment-matching condition is satisfied trivially by any `theta` that integrates to `mu_E`. The likelihood surface collapses.

- Code enforcement: the underlying `MCEIRLEstimator` accepts an action-dependent feature matrix and the shared known-truth harness now validates that low-level path directly. The sklearn-style wrapper no longer silently treats `feature_matrix=None` as a validated multi-action structural default: `fit()` raises unless the user passes a `RewardSpec` or an explicit `feature_matrix`.

- Match: **yes for the validated low-level path; wrapper guard fixed**.

### Hyperparameter defaults vs paper defaults

- `MCEIRLConfig.learning_rate`: 0.02 (gradient/Adam path; not used by the
  gated root run).
- `MCEIRLConfig.outer_max_iter`: 200.
- `MCEIRLConfig.inner_max_iter`: 10000 (for the soft VI inner solver).
- Known-truth validation override: `optimizer="root"`, `outer_tol=1e-8`,
  `compute_se=False`.
- `MCEIRLConfig.occupancy_tol`: dual stopping criterion borrowed from imitation, controls the L-infinity threshold between demo and policy state visitation.

Match: **yes** for the configurable knobs. The defaults are reasonable for medium panels.

### Findings / fixes applied

- **Known-truth validation added.** `experiments/known_truth.py` now runs
  `MCEIRLEstimator` with `MCEIRLConfig(optimizer="root", compute_se=False)` on
  action-dependent known-truth features. The non-smoke gates are convergence,
  feature residual, occupancy moment residual, normalized reward/value/Q RMSE,
  policy TV, and Type A/B/C counterfactual regret. Raw parameter cosine is not
  a gate.

- **Moment bug fixed.** Empirical MCE feature moments now use discounted
  state-action occupancy, matching the infinite-horizon expected occupancy
  side. The old unweighted panel average could match moments while recovering
  the wrong reward and policy.

- **Wrapper-default guard applied.** The wrapper keeps constructor
  compatibility but raises at `fit()` for multi-action models without an
  explicit reward specification or feature matrix. It also accepts 3D
  state-action feature matrices.

- VALIDATION_LOG.md status: **Pass**. The generated artifact
  `papers/econirl_package/primers/mce_irl/mce_irl_results.json` passes 20/20
  gates across the canonical sanity cell and the primary
  `mce_low_high_reward` cell.
