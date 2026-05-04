## Estimator: MPEC (Mathematical Programming with Equilibrium Constraints)
## Paper(s): Su & Judd 2012 "Constrained Optimization Approaches to Estimation of Structural Models," Econometrica. Also Iskhakov, Rust, Schjerning 2016 (the comparison paper at `papers/foundational/iskhakov_rust_schjerning_2016_mpec_comment.md`).
## Code: `src/econirl/estimation/mpec.py`

### Known-truth migration status

- Migrated to the shared synthetic DGP in `papers/econirl_package/primers/mpec/mpec_run.py`.
- RTD front door is `docs/estimators/mpec.md`; reference PDF source is `papers/econirl_package/primers/mpec/mpec.tex`.
- Main validation cell is `canonical_low_action`: 2,000 individuals, 80 periods, 21 states, 3 actions, known transitions, and action-dependent reward features.
- Hard gates are now active in `experiments/known_truth.py`: SLSQP convergence, Bellman constraint violation, finite standard errors, parameter recovery, policy/value/Q recovery, and Type A/B/C counterfactual regret.
- The run uses Su-Judd's constrained likelihood with scipy SLSQP and JAX objective/constraint derivatives. The Iskhakov-Rust-Schjerning caution remains: MPEC is not the large-state/high-discount default.

### Loss / objective

- Paper formula (Su-Judd 2012, eq. 2.4–2.6): instead of NFXP's nested approach, treat the value function `V` as a decision variable subject to the Bellman fixed-point as an equality constraint. Maximize the same conditional log-likelihood as NFXP

  ```
  L(theta, V) = sum_i sum_t log Pr(a_it | s_it; theta, V)
  ```

  subject to

  ```
  V(s) = sigma * log sum_a exp[(u(s, a; theta) + beta * sum_s' P(s'|s,a) V(s')) / sigma]   for all s
  ```

  Solver: SQP or augmented Lagrangian on the joint variable `(theta, V)`.

- Code implementation: `mpec.py` ships two solvers selectable via `MPECConfig.solver`:
  - `"sqp"`: scipy SLSQP with JAX-JIT'd objective and constraint Jacobians (the recommended path per the codebase comment block).
  - `"augmented_lagrangian"` / `"slsqp"`: legacy augmented Lagrangian aliases with manual penalty schedule.

  The pseudo-likelihood is the same conditional choice probability as NFXP; the constraint set is the per-state Bellman residual stacked into a single `S`-dimensional equality constraint vector.

- Match: **yes**, both solver paths implement the canonical Su-Judd formulation. The `"sqp"` path is the recommended default per the codebase's documented comparison ("Best MPEC: scipy SLSQP + JAX JIT gradients; full JAX-native SQP fails (nearly-square J)" from the project memory `mpec_slsqp_jax.md`).

### Gradient

- Paper formula: KKT conditions over `(theta, V, lambda)` where `lambda` is the constraint multiplier. The Jacobian of the constraint w.r.t. `V` is `I - beta * P_pi`, and the Jacobian w.r.t. `theta` is the gradient of the soft Bellman operator.

- Code implementation: JAX autodiff produces the constraint Jacobian; SLSQP receives both the objective gradient and the constraint Jacobian as JIT-compiled callables.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner loop. The Bellman equation appears as an equality constraint, not as a fixed-point computation.

- Code algorithm: matches. The constraint residual is computed once per SQP iteration and the solver moves `(theta, V)` jointly toward the KKT point.

- Match: **yes**.

### Identification assumptions

- Paper conditions: same as NFXP (full-rank feature matrix, known transitions). Additionally, the constraint set must satisfy LICQ at the solution; Su-Judd 2012 Theorem 2 gives sufficient conditions for this in the soft-max case.

- Code enforcement: not explicitly checked. The wrapper relies on SLSQP to fail loudly if the constraint Jacobian is singular at iteration. The Iskhakov-Rust-Schjerning 2016 comment paper warns that MPEC can be slower than NFXP at high beta when the constraint Jacobian becomes nearly singular, and the project memory `mpec_slsqp_jax.md` confirms this on the package's own benchmarks ("3x slower than NFXP at beta=0.99").

- Match: **yes** for theory; **runtime caveat** documented separately.

### Hyperparameter defaults vs paper defaults

- `solver`: `"sqp"` (SLSQP).
- `tol`: `1e-6` (SLSQP convergence tolerance).
- `max_iter`: `200`.

Match: **yes**, matches Su-Judd 2012's recommended SLSQP settings for medium-scale problems.

### Findings / fixes applied

- The estimator implementation already matched the Su-Judd constrained likelihood. The migration change was in the known-truth harness: MPEC now runs the full SLSQP path with robust standard errors and hard non-smoke recovery gates.
- The old Rust-bus primer has been replaced with the shared known-truth DGP tutorial. No real-data estimation is published on RTD.
