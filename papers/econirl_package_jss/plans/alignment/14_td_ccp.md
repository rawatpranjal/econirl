## Estimator: TD-CCP (Temporal-Difference CCP)
## Paper(s): Adusumilli & Eckardt 2025 "Temporal-Difference Estimation of Dynamic Discrete Choice Models" (working paper). Doclinged at `papers/foundational/adusumilli_eckardt_2025_td_ccp.md`.
## Code: `src/econirl/estimation/td_ccp.py`

### Known-truth migration status

- Status: **validated for the paper-faithful hard case**.
- RTD front door: `docs/estimators/tdccp.md`.
- Dedicated TeX/PDF: `papers/econirl_package/primers/tdccp/tdccp.tex`.
- Result generator: `papers/econirl_package/primers/tdccp/tdccp_run.py`.
- Shared DGP harness: `experiments/known_truth.py`.
- Focused tests: `tests/test_td_ccp.py`, `tests/test_tdccp_wrapper.py`, and `tests/test_shapeshifter_env.py`.

The validation target is deliberately narrow and paper-faithful: TD-CCP
recovers finite-dimensional structural reward parameters when utility is
linear in known features. Those features may be flexible functions of the
state, including neural state encodings. Raw neural rewards with no finite
true `theta` are retained only as a diagnostic stress test.

### Loss / objective

- Paper formula: Adusumilli and Eckardt start from the CCP pseudo-log-likelihood
  for a DDC model with instantaneous utility `z(a, x)' theta + epsilon`. The
  recursive nuisance terms `h(a, x)` and `g(a, x)` enter the choice-specific
  value index as

  ```
  h(a, x)' theta + g(a, x)
  ```

  where `h` is the discounted value of known reward features and `g` is the
  discounted value of the CCP entropy term. The paper's core computational
  move is to estimate those recursive terms directly from observed
  `(a, x, a', x')` transitions, without estimating a transition density.

- Code implementation: `td_ccp.py` estimates empirical CCPs, constructs the
  observed transition tuples, estimates `h` and `g`, and then maximizes the
  CCP pseudo-likelihood over the structural reward parameter vector. The
  objective is scaled as an empirical average so optimizer tolerances have a
  stable interpretation across sample sizes.

- Match: **yes** for the validated semi-gradient path.

### Gradient / TD nuisance estimation

- Paper formula: the linear semi-gradient method approximates each component
  of `h` with a basis `phi(a, x)' omega` and computes the projected TD fixed
  point

  ```
  omega = [E phi(a,x) {phi(a,x) - beta phi(a',x')}']^{-1}
          E[phi(a,x) z(a,x)]
  ```

  The same construction estimates `g` with a possibly different basis `r` and
  target `beta e(a', x'; eta)`.

- Code implementation: the low-level `TDCCPEstimator` implements the
  semi-gradient estimator with tabular and encoded bases. The paper-faithful
  hard case uses `method="semigradient"` and `basis_type="tabular"` so the
  finite-state value-term approximation is rich enough to recover the DGP.

- Match: **yes** after the hard-case fixes. The implementation now uses the
  paper initialization for AVI output shifts, stable empirical-average
  likelihood scaling, and an honest optimizer convergence flag. The hard-case
  showcase itself uses the linear semi-gradient path, where the paper theory
  is strongest.

### Bellman / inner loop

- Paper algorithm: no repeated structural Bellman solve is required for
  parameter estimation. TD learning estimates the recursive terms from sample
  successors; counterfactual analysis may still use a transition model after
  `theta` is estimated.

- Code algorithm: the estimator does not use the known transition matrix while
  fitting the hard case. Known transitions from `ShapeshifterEnvironment` are
  used only after fitting to evaluate policy, value, Q, and counterfactual
  truth.

- Match: **yes**.

### Identification assumptions

- Paper conditions: the reward is finite-dimensional and linear in known
  features `z(a, x)`. Identification is about `theta`, not about an arbitrary
  unrestricted neural reward surface. The linear semi-gradient theory also
  requires a well-conditioned basis for the value-term approximation and
  adequate empirical support over observed state-action-successor tuples.

- Code enforcement: the gated hard case constructs a shapeshifter DGP with
  `feature_type="neural"` and `reward_type="linear"`. The neural features are
  frozen known features, so the true reward still has a finite structural
  `theta`. The raw `reward_type="neural"`, `feature_type="neural"` case is
  evaluated separately and is not used to make a TD-CCP success claim.

- Match: **yes** for the paper-faithful target. The raw neural-reward case is
  outside the finite-theta identification claim and correctly fails the reward,
  value, and Q gates.

### Hyperparameter defaults vs paper defaults

Validated hard-case config:

- `method="semigradient"`.
- `basis_type="tabular"`.
- `cross_fitting=False`.
- `robust_se=False`.
- `compute_se=False`.
- `n_policy_iterations=1`.
- `outer_max_iter=1000`.
- `outer_tol=1e-8`.

The public estimator names and constructors are unchanged. The hard-case
artifact uses the low-level estimator directly so the showcase is not obscured
by wrapper defaults.

### Current gated artifact

Paper-faithful hard case:

- Cell: `shapeshifter_linear_reward_neural_features`.
- DGP: 32 states, 3 actions, 4 frozen neural state features, stochastic
  transitions, deterministic linear rewards, infinite horizon, beta = 0.95.
- Result: **Pass, 10/10 gates**.
- Parameter cosine: 0.999982.
- Parameter relative RMSE: 0.006953.
- Reward normalized RMSE: 0.006953.
- Policy TV: 0.001697.
- Value normalized RMSE: 0.002452.
- Q normalized RMSE: 0.002559.
- Type A/B/C regret: 0.000238, 0.000177, 0.000268.

Raw neural diagnostic:

- Cell: `shapeshifter_neural_neural`.
- Result: **Fail, 4/8 gates**.
- Failed gates: convergence, reward normalized RMSE, value normalized RMSE,
  and Q normalized RMSE.
- Interpretation: this is useful evidence about the boundary of TD-CCP, not a
  contradiction of the paper-faithful validation. The raw neural reward matrix
  has no finite true `theta`, so parameter recovery is not a meaningful gate.

Other canonical cells:

- `canonical_low_action` passes 10/10 gates and remains the simple sanity
  check.
- `canonical_high_action` is retained as a diagnostic and currently fails 0/10
  gates. It is not used as the TD-CCP showcase claim.

### Findings / fixes applied

- The previous audit incorrectly described the estimator as a joint
  pseudo-likelihood plus TD-residual loss. The paper estimates TD nuisance
  terms `h` and `g`, then estimates `theta` through the CCP criterion. The
  audit has been corrected to match that structure.
- The shapeshifter environment now uses a real state-action neural feature map
  when action-dependent neural features are requested.
- TD-CCP's hard-case runner now separates the paper-faithful finite-theta
  neural-feature showcase from the out-of-scope raw neural-reward diagnostic.
- RTD and primer prose now state the boundary explicitly: TD-CCP is for linear
  rewards in known features, while those known features can be flexible or
  neural encodings of state.

- VALIDATION_LOG.md status: **Pass for finite-theta neural-feature hard case;
  raw neural reward remains diagnostic and currently fails structural gates**.
