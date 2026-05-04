## Estimator: CCP (Conditional Choice Probability)
## Paper(s): Hotz & Miller 1993 (CCP inversion theorem); Aguirregabiria & Mira 2002 (NPL iterations); Arcidiacono & Miller 2011 (CCP with unobserved heterogeneity, used by AIRL-Het not directly here). All at `papers/foundational/hotz_miller_1993_ccp.md`, `papers/foundational/AguirregabiriaMira_ECMA2002.md`, `papers/foundational/AguirregabiriaMira_ECMA2002.pdf`, and `papers/foundational/arcidiacono_miller_2011_ccp_unobserved.md`.
## Code: `src/econirl/estimation/ccp.py`

### Known-truth migration status

- Migrated to the shared synthetic DGP in `papers/econirl_package/primers/ccp/ccp_run.py`.
- RTD front door is `docs/estimators/ccp.md`; reference PDF source is `papers/econirl_package/primers/ccp/ccp.tex`.
- Main validation cell is `canonical_low_action`: 2,000 individuals, 80 periods, 21 states, 3 actions, known transitions, and action-dependent reward features.
- Hard gates are now active in `experiments/known_truth.py` for K=10 NPL: finite standard errors, parameter recovery, policy/value/Q recovery, and Type A/B/C counterfactual regret.
- CCP reports values in the package's soft-Bellman convention by evaluating the recovered policy under the recovered reward. The Hotz-Miller Euler-constant value representation is used internally, not as the reported value object.

### Loss / objective

- Paper formula (Hotz-Miller 1993, Theorem 3.1; the inversion lemma): for any reference action `a_ref`,

  ```
  V(s) - U(s, a_ref; theta) - beta * E[V(s') | s, a_ref] = -log Pr(a_ref | s)
  ```

  so that `V` is recovered up to a state-independent constant from the observed CCPs and one additional Bellman evaluation. The pseudo-likelihood is

  ```
  L_PL(theta) = prod_i prod_t Pr(a_it | s_it; theta, hat_pi)
  ```

  where `hat_pi` is the empirical CCP from the data, and `Pr(a | s; theta, hat_pi)` uses the Hotz-Miller inversion to compute the conditional choice probability without solving for `V` from scratch.

- Aguirregabiria-Mira 2002, NPL iteration: alternate (a) optimize `L_PL` over theta given current CCPs, (b) update CCPs from the optimized theta, until both fixed. The pseudo-likelihood with `K` NPL iterations converges to the MLE as `K -> infinity` and equals MLE in `1` iteration only when the Hotz-Miller pseudo-likelihood has a unique mode (rare in practice).

- Code implementation: `ccp.py:_optimize` constructs the Hotz-Miller pseudo-likelihood and runs L-BFGS-B over theta. The NPL iteration count is the constructor argument `num_policy_iterations` (default `1`); it loops over (optimize theta, update CCP) for that many iterations.

- Match: **yes** for the formulas; **default knob is wrong** for the iteration count. NPL=1 is the original Hotz-Miller; Aguirregabiria-Mira showed it has multiple local modes and recommend NPL >= 5.

### Gradient

- Paper formula: same as NFXP under the inversion (the pseudo-likelihood is differentiable in theta given fixed `hat_pi`).

- Code implementation: JAX autodiff through the pseudo-likelihood objective; same `minimize_lbfgsb` used by NFXP.

- Match: **yes**.

### Bellman / inner loop

- Paper algorithm: no inner Bellman fixed-point. The Hotz-Miller inversion replaces the Bellman fixed-point with a single matrix solve given the CCPs.

- Code algorithm: matches. Per-iteration, `ccp.py` solves the linear system for V given CCPs (single linear solve, not an iterative VI loop).

- Match: **yes**.

### Identification assumptions

- Paper conditions: full-rank feature matrix; transitions are known; **CCPs are positive on the support** (no zero observed-frequencies). The Aguirregabiria-Mira 2002 NPL fix relies on the same conditions plus monotone improvement in the pseudo-likelihood.

- Code enforcement: the empirical CCPs from `panel.compute_choice_frequencies` use the same row-sum trick as the rest of the codebase (smoothing only when row is zero). For panels where every state-action pair is observed at least a few times, this is safe. The `pre-estimation diagnostics` check in CLAUDE.md flags single-action states as a structural problem; the wrapper does not abort on those, but the diagnostic prints the count.

- Match: **yes**, with the caveat that single-action states still produce degenerate CCPs.

### Hyperparameter defaults vs paper defaults

- `num_policy_iterations` default in code: **`1`** (classic Hotz-Miller).
- Aguirregabiria-Mira 2002 recommendation: **`>= 5`**, ideally `>= 10` until the pseudo-likelihood stops moving.
- Match: **disagrees**. This is the documented sign-flip trap from VALIDATION_LOG.md ("CCP on rust-small": with NPL=1 theta_c comes out -0.001289, with NPL=10 it comes out +0.001003 matching NFXP exactly).

### Findings / fixes applied

- **Default knob fix recommended but not applied here.** The trap is documented in VALIDATION_LOG.md and in CLAUDE.md; the JSS Tier 2 and Tier 4 cells now explicitly pass `num_policy_iterations=10` via `extra_per_estimator` in `experiments/jss_deep_run/matrix.py`. Changing the package default is a separate decision (it would affect all downstream users) and is left to the package maintainer; the alignment audit's role is to surface the discrepancy.

- The cell-level fix is in matrix.py and is safe to dispatch.

- VALIDATION_LOG.md status: **Pass with caveat (requires NPL >= 10)**.
