# Estimator validation log

A running record of estimator-by-estimator validation, ordered from
the most fundamental claim down to the more speculative ones. Each
entry says exactly what was tested, what the result was, and whether
the estimator is fit for the role assigned to it in the five plans.
A claim does not enter the paper until its row in this log says
"Pass" with the recovered numbers attached.

The principle: the deep run on RunPod costs nothing if the
estimators don't work in their intended conditions. Validate locally
on cheap single fits first; then expand to Monte Carlo; only then
spend compute.

## Status legend

- **Pass.** Estimator converges, recovers the target quantity within
  tolerance, behaves as documented.
- **Pass with caveat.** Estimator works but a documented condition
  needs to hold (e.g. requires bootstrap SEs because asymptotic
  Hessian is degenerate; requires careful initialization; only valid
  when transitions are known).
- **Fail.** Estimator does not converge, recovers the wrong answer,
  or only works under conditions we do not satisfy. The plan that
  depended on this estimator gets revised before any compute spend.
- **Pending.** Not yet run.

## Log

### NFXP on rust-small

- Source paper: Rust 1987 (canonical native environment).
- Target: recover theta_1 = 0.001, RC = 3.0 from the bundled
  synthetic panel (9410 obs, 90 mileage bins).
- Acceptance: relative error below 5 percent on both parameters.
- Status: **Pass.**
  - Wall-clock: 8.5 seconds on CPU (Apple M2 Pro).
  - Converged: True.
  - Log-likelihood: -1900.33.
  - theta_c estimate 0.001003 (truth 0.001), relative error 0.29 percent. Asymptotic SE 0.000395.
  - RC estimate 3.0722 (truth 3.0), relative error 2.4 percent. Asymptotic SE 0.0747.
  - Standard errors cover the truth at the 95 percent level.
- Reviewer cross-check. The current paper draft has two
  inconsistent NFXP-on-rust-small numbers: Listing 2 reports
  RC = 3.0722 and log-likelihood -1900.33 (matches this run), and
  the prose elsewhere reports RC = 3.0115 and log-likelihood
  -4263.20 (does not match this run). The second number must come
  from a different setting (different data subset, different
  transitions, different bus group). Tracking under failure mode
  "stale numbers in prose", to be resolved when paper Section 4 is
  rewritten from the artifacts.

### CCP on rust-small

- Source paper: Hotz-Miller 1993 with NPL of Aguirregabiria-Mira 2002.
- Target: recover the same parameters; match NFXP within 5 percent.
- Acceptance: relative error below 5 percent on both parameters.
- Status: **Fail (sign bug on theta_c).**
  - Wall-clock: 7.8 seconds on CPU.
  - Converged: True.
  - Log-likelihood: -1907.05.
  - RC estimate 2.9444 (truth 3.0), relative error 1.85 percent. **Correct.**
  - theta_c estimate **-0.001289** (truth +0.001000). **Wrong sign**, magnitude in the right ballpark (1.3x).
  - Asymptotic SE 0.0002329 on theta_c, but the sign is wrong so the SE is meaningless.
  - The feature matrix in `nfxp._create_utility` (which CCP inherits) encodes `U(s, a=0) = -theta_c * s`, so a positive theta_c means keeping becomes less attractive at higher mileage. NFXP recovers theta_c = +0.001 with this convention. CCP recovers it with the opposite sign even though it inherits the same `_create_utility`.
  - The bug is in the low-level `CCPEstimator._optimize` in `src/econirl/estimation/ccp.py`, not the sklearn wrapper. NPL pseudo-likelihood gradient sign or feature transposition is the most likely cause.
  - **Cross-check.** The existing `examples/rust-bus-engine/benchmark_results.csv` reports CCP recovering `operating_cost = +0.001230` (correct sign) on the *same* underlying CCPEstimator. So CCP *can* recover the right sign through some code path. The bug is therefore most likely in the sklearn-wrapper layer (`econirl.estimators.ccp.CCP.fit`), not in the low-level NPL optimizer itself. Most plausible causes:
    - Wrapper passes a transition tensor with axes swapped relative to what the low-level CCP expects (the package docs warn about `(A,S,S)` vs `(S,A,S)` convention).
    - Wrapper builds the feature matrix with the wrong sign on the keep-action mileage column.
    - Wrapper sets initial parameters with the wrong sign and the inner pseudo-likelihood is bimodal so it converges to a sign-flipped local mode.
  - **Action**: cannot use CCP in any plan until this is reproduced and fixed. Filed as a blocking issue. The fix should make a single fresh CCP call through `econirl.estimators.CCP` recover the same `+0.001` operating cost the benchmark CSV reports.

### Cross-cutting finding: log-likelihood inconsistency in the existing materials

- The current paper draft (Listing 2) reports NFXP log-likelihood **-1900.33** and `RC = 3.0722` on rust-small. Independently reproduced above.
- The current paper prose elsewhere reports NFXP log-likelihood **-4263.20** and `RC = 3.0115` on rust-small. The bundled `examples/rust-bus-engine/benchmark_results.csv` matches the second pair.
- The two numbers cannot both be right for the same problem. Either the benchmark uses a different bus group (or `original=True`) or different transitions or different starting values.
- The reviewer flagged this as a credibility issue. The fix is the same as for CCP: rewrite Section 4 of the paper from artifacts produced by a single frozen script. Until the artifact exists the paper says only what is in the validation log.

### MCE-IRL on rust-small

- Source paper: Ziebart 2008. Equivalence claim against NFXP
  under linear features and known transitions.
- Target: recover the same parameters as NFXP up to scale and
  additive constant, after the standard IRL identification
  normalization.
- Acceptance: reward identification residual below 0.05 against
  the NFXP point estimate.
- Status: **Pending**.

### MCE-IRL on ziebart-small

- Source paper: Ziebart 2008 (canonical native environment).
- Target: recover step penalty -0.1 and terminal reward 10.0 to
  high precision.
- Acceptance: reward identification residual below 0.05 against
  the true reward weights.
- Status: **Pending**.

### MPEC on rust-small

- Source paper: Su-Judd 2012.
- Target: recover NFXP estimates within 5 percent.
- Status: **Pending**.

### NNES on rust-small

- Source paper: Nguyen 2025. Native environment is Rust bus.
- Target: recover NFXP estimates within 10 percent.
- Status: **Pending**.

### SEES on rust-small

- Source paper: Luo-Sang 2024.
- Target: recover NFXP estimates within 10 percent.
- Status: **Pending**.

### TD-CCP on rust-small

- Source paper: Adusumilli et al. 2022.
- Target: recover NFXP estimates within 10 percent.
- Status: **Pending**.

### GLADIUS on rust-small

- Source paper: Kang et al. 2025.
- Target: recover NFXP estimates within 10 percent on the genuine
  mileage dimension.
- Status: **Pending**.

### AIRL on rust-small

- Source paper: Fu et al. 2018 (native environment is MuJoCo, but
  the package adapts it to discrete DDC panels).
- Target: recover NFXP point estimate up to additive constant.
- Status: **Pending**. Likely candidate for "fail in this
  environment" given the MuJoCo native setting; if it fails the
  paper says so.

### IQ-Learn on rust-small

- Source paper: Garg et al. 2021 (native environment is Atari and
  MuJoCo).
- Target: produce a finite tabular Q-function with policy KL below
  0.10 against NFXP's policy.
- Status: **Pending**.

### f-IRL on rust-small

- Source paper: Ni et al. 2021 (native environment is MuJoCo).
- Target: produce non-degenerate reward weights.
- Status: **Pending**. Already known to struggle on this setting
  per the README benchmark; if it fails the plan documents that.

### BC on rust-small

- Source paper: Pomerleau 1991.
- Target: maximum in-sample action log-likelihood; documented
  failure on counterfactual replacement-cost shifts.
- Status: **Pending**.

### AIRL-Het on lsw-synthetic

- Source paper: Lee, Sudhir, Wang 2026 (the one estimator built
  specifically for this dataset).
- Target: recover mixture weights (0.4, 0.6) within 5 percentage
  points; per-type reward parameters within 15 percent.
- Status: **Pending**.

### GLADIUS, NNES, TD-CCP on rust-big

- Tested against the rust-small reference once they pass on
  rust-small.
- Status: **Pending**.

### Deep MCE-IRL on ziebart-big

- Tested against the metadata-declared true RBF coefficients once
  the tabular MCE-IRL passes on ziebart-small.
- Status: **Pending**.
