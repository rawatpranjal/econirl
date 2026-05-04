## Estimator: SEES (Sieve Estimation of Economic Structural Models)
## Paper(s): Luo & Sang 2024 "Efficient Estimation of Structural Models via Sieves". Doclinged at `papers/foundational/luo_sang_2024_sees.md`.
## Code: `src/econirl/estimation/sees.py`

### Known-truth migration status

- Status: **migrated**.
- RTD front door: `docs/estimators/sees.md`.
- Dedicated TeX/PDF: `papers/econirl_package/primers/sees/sees.tex`.
- Result generator: `papers/econirl_package/primers/sees/sees_run.py`.
- Shared DGP harness: `experiments/known_truth.py`.
- Fast test: `tests/test_known_truth.py::test_sees_smoke_fit_produces_known_truth_metrics_and_gates`.

The migrated validation uses only the synthetic known-truth DGP. No real data
or legacy examples are used on the RTD surface.

### Loss / objective

- Paper formula: replace the value function with a sieve approximation
  `V_K(s) = sum_j alpha_j b_j(s)` and maximize the choice likelihood minus an
  equilibrium penalty:

  ```
  L(theta, alpha) =
      sum_i sum_t log Pr(a_it | s_it; theta, V_K(alpha))
      - omega_n * ||V_K(alpha) - T_theta V_K(alpha)||^2
  ```

- Code implementation: `sees.py:_optimize` constructs a basis matrix `B`,
  parameterizes `V = B @ alpha`, and runs joint L-BFGS-B over `(theta, alpha)`.

- Match: **yes**. The implementation uses the Bellman residual penalty, not a
  placeholder regularizer.

### Gradient

- Paper formula: penalized-MLE gradient over structural and sieve parameters.
- Code implementation: JAX autodiff of the full penalized objective.
- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner Bellman fixed point. The Bellman penalty replaces
  the hard fixed-point solve.
- Code algorithm: no value-iteration loop inside the likelihood. The final
  Bellman residual is reported in `metadata["bellman_violation"]`.
- Match: **yes**.

### Standard errors

- Paper target: marginal structural inference after accounting for the sieve
  nuisance parameters.
- Code implementation: numerical Hessian of the joint penalized objective and
  Schur complement

  ```
  H_tt - H_ta @ solve(H_aa, H_at)
  ```

- Known-truth gate: `standard_errors_finite == true`.
- Match: **yes**, with finite-SE validation now enforced.

### Hyperparameter findings

The old primer used a Fourier basis with `K=8` on a legacy multi-component bus
comparison. That is archived material, not the current validation.

A live medium-scale run on `canonical_low_action` showed that the previous
harness default (`basis_dim=8`, penalty weight inherited from the estimator,
and no SE computation) did not recover the full known truth:

- Bellman violation was too large for structural validation.
- Parameter recovery missed the non-smoke gates.
- Standard errors were not computed.

The migrated harness uses the finite-state limiting logic from Luo and Sang:

- `basis_type="bspline"`
- `basis_dim=min(num_states, 21)` for the non-smoke canonical cell
- `penalty_weight=100.0`
- `max_iter=1000`
- `tol=1e-7`
- `compute_se=True`

On the canonical 21-state DGP this makes the sieve as rich as the finite value
vector. The Bellman penalty then enforces the dynamic structure while retaining
the SEES joint-optimization path.

### Hard gates

The non-smoke SEES gates in `experiments/known_truth.py` require:

- Bellman violation <= 0.05.
- finite standard errors.
- parameter cosine >= 0.99.
- parameter relative RMSE <= 0.15.
- reward RMSE <= 0.03.
- policy TV <= 0.02.
- value RMSE <= 0.10.
- Q RMSE <= 0.10.
- Type A/B/C counterfactual regret <= 0.01.

Current canonical run:

- Bellman violation: 0.044579.
- parameter cosine: 0.995485.
- parameter relative RMSE: 0.124955.
- reward RMSE: 0.013579.
- policy TV: 0.008884.
- value RMSE: 0.080710.
- Q RMSE: 0.080546.
- Type A/B/C regret: 0.000526, 0.001281, 0.000313.

All hard gates pass.

### Optimizer flag nuance

The L-BFGS-B wrapper reports the absolute-gradient convergence flag as
`summary.converged`. For the validated SEES run that flag is `false` even
though the structural Bellman residual, standard errors, reward/value/Q/policy
recovery, and counterfactual recovery pass. This is not hidden. The RTD page
and PDF report the flag and explain that the hard SEES validation gate is the
structural recovery bundle, not the sample-summed absolute-gradient flag.

### Remaining scope

No additional SEES files are needed for the current RTD migration. Future work
can add an adaptive penalty schedule following the paper's asymptotic
discussion, but that is not required for the current finite-state known-truth
validation.
