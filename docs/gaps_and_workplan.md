# econirl Gap Analysis and Work Plan

Date: 2026-04-03.
Sources: internal codebase audit, external reviewer feedback on PyPI install and docs, and email thread with collaborators.

This document describes every known gap between the current state of econirl and where it needs to be. The gaps are organized into five independent chunks that can be worked on in parallel by separate agents or contributors. Each chunk is self-contained and does not depend on another chunk being completed first, though some chunks have natural ordering within their own steps.

---

## Chunk 1: Packaging, Install, and Backend Consolidation

This chunk addresses the most urgent problem: a stranger cannot pip install econirl and get a working import on a clean machine. The external reviewer confirmed this. The install surface is the first thing that breaks trust.

### 1.1 Backend ambiguity: PyTorch versus JAX

The core estimation layer (src/econirl/estimation/) is almost entirely JAX. 86 files import JAX. But 17 files still import PyTorch, concentrated in:

Files that still import torch:
- src/econirl/estimators/nfxp.py (sklearn wrapper)
- src/econirl/estimators/tdccp.py (sklearn wrapper)
- src/econirl/estimators/neural_gladius.py (sklearn wrapper)
- src/econirl/estimators/neural_airl.py (sklearn wrapper)
- src/econirl/estimators/mceirl_neural.py (sklearn wrapper)
- src/econirl/estimators/gcl.py (sklearn wrapper)
- src/econirl/estimators/max_margin_irl.py (sklearn wrapper)
- src/econirl/estimators/maxent_irl.py (sklearn wrapper)
- src/econirl/estimation/iq_learn.py (core estimator)
- src/econirl/contrib/bayesian_irl.py
- src/econirl/contrib/deep_maxent_irl.py
- src/econirl/contrib/gail.py
- src/econirl/contrib/gcl.py
- src/econirl/contrib/max_margin_irl.py
- src/econirl/contrib/max_margin_planning.py
- src/econirl/contrib/maxent_irl.py
- src/econirl/simulation/counterfactual.py

The estimation-layer files (estimation/nfxp.py, estimation/ccp.py, estimation/mce_irl.py, etc.) are already JAX. The sklearn wrappers in estimators/ still use torch for tensor ops and data handling. The contrib/ directory is entirely torch.

pyproject.toml declares JAX dependencies but does NOT declare torch. The installation docs (docs/installation.rst) say "PyTorch >= 2.0" is required. This is the core inconsistency.

**Decision needed:** The package should be JAX-only. Torch was the original backend and the migration is almost complete. The remaining torch usage is in wrappers and contrib.

**Work items:**
- Migrate all 8 sklearn wrappers (estimators/*.py) from torch to JAX/numpy. These files use torch mainly for tensor creation, type conversion, and softmax. The underlying estimation calls already return JAX arrays.
- Migrate estimation/iq_learn.py from torch to JAX. This is the only core estimator still on torch.
- Migrate simulation/counterfactual.py from torch to JAX.
- Migrate all 7 contrib/ files from torch to JAX. These are lower priority since contrib is experimental.
- Remove torch from installation docs.
- Do NOT add torch to pyproject.toml dependencies. The goal is to eliminate it.
- After migration, add a CI check that greps for "import torch" in src/econirl/ and fails if any are found (excluding a possible torch-compat shim for tests).

### 1.2 Silent import suppression in __init__.py

The top-level __init__.py wraps estimator imports in try/except blocks that silently swallow ImportError and SyntaxError:

```python
try:
    from econirl.estimators import NFXP, CCP, MaxEntIRL, MaxMarginIRL, MCEIRL, NNES, SEES, TDCCP
except (ImportError, SyntaxError):
    pass
```

This means if torch is missing (which it will be after a clean pip install since torch is not in dependencies), all sklearn-style estimators silently vanish. The user gets no error, just missing names. The external reviewer hit exactly this.

**Work items:**
- After the torch migration (1.1), all these imports will succeed because they will only need JAX (which is in dependencies). At that point, remove the try/except blocks and let imports fail loudly.
- If the migration is not yet complete, at minimum replace the silent pass with a warning: `warnings.warn("Could not import sklearn-style estimators: {e}. Install torch with: pip install torch")`.
- The comment "still on PyTorch, pending migration" in __init__.py should be removed once migration is done.

### 1.3 JAX float64 precision

The file src/econirl/_jax_config.py exists and enables float64:
```python
jax.config.update("jax_enable_x64", True)
```

But nothing imports this file. It is dead code. The external reviewer saw float64 truncation warnings.

Structural estimation requires float64 for inner loop tolerances of 1e-12 and accurate Hessian computation for standard errors. This is not optional.

**Work items:**
- Add `import econirl._jax_config` as the FIRST import in src/econirl/__init__.py, before any other econirl imports. This ensures x64 is enabled before any JAX computation.
- Alternatively, put the `jax.config.update("jax_enable_x64", True)` call directly at the top of __init__.py.
- Verify that individual modules that call `jax.config.update("jax_enable_x64", True)` locally (there are 16 such files) still work. After the __init__.py fix, the per-module calls become redundant but harmless.
- Add a test that verifies `jax.numpy.ones(1).dtype == jax.numpy.float64` after import.

### 1.4 Installation docs drift

docs/installation.rst has three factual errors:
1. Line 13: Lists "PyTorch >= 2.0" as a dependency. After JAX migration, this should be removed entirely.
2. Line 33: Clone URL is `https://github.com/econirl/econirl.git`. The actual repo is `https://github.com/rawatpranjal/econirl.git`.
3. Line 13: Does not mention JAX, equinox, optax, or any of the actual JAX dependencies.

**Work items:**
- Fix the clone URL to `https://github.com/rawatpranjal/econirl.git`.
- Replace the dependency list with the actual dependencies from pyproject.toml (JAX stack, numpy, scipy, pandas, matplotlib, gymnasium, tqdm).
- Remove PyTorch from the dependency list.
- Add a note about Python 3.10+ requirement.

### 1.5 README and PyPI description drift

The README "Try It" example uses `from econirl.estimation import BehavioralCloningEstimator`, which is the legacy API. The quickstart in docs uses `from econirl import NFXP, CCP`, which is the recommended API. These tell different stories about what the package is.

**Work items:**
- Update README "Try It" to use the recommended sklearn-style API: `from econirl import NFXP` with `load_rust_bus()`.
- Ensure the PyPI long description (rendered from README.md) matches the docs quickstart.

### 1.6 Clean-install CI smoke test

No GitHub Actions workflows exist (`.github/workflows/` is empty). The repo has no CI.

**Work items:**
- Create `.github/workflows/ci.yml` with:
  - Python 3.10, 3.11, 3.12 matrix
  - Linux and macOS
  - Step 1: `pip install .` (wheel install from source)
  - Step 2: `python -c "import econirl; print(econirl.__version__); from econirl import NFXP, CCP; from econirl.datasets import load_rust_bus; df = load_rust_bus(); print(len(df))"`
  - Step 3: `pip install .[dev] && python -m pytest tests/ -v -m "not slow" --timeout=120`
- Run on every push and PR to main.
- Add a nightly job that runs slow benchmarks.

### 1.7 tqdm dependency

pyproject.toml already lists `tqdm>=4.60` in dependencies (line 42). The external reviewer said tqdm was missing, which suggests the reviewer tested an older PyPI release (0.0.1 or 0.0.2) that did not have tqdm. This may already be fixed in 0.0.3, but should be verified.

**Work items:**
- Verify that `pip install econirl==0.0.3` in a clean environment provides tqdm.
- If not, check that the 0.0.3 wheel on PyPI was built from the current pyproject.toml.

---

## Chunk 2: API Consistency and Data Contracts

This chunk addresses the second most critical gap: the package has two APIs (legacy and sklearn-style) with no clear boundary, and input validation is weak.

### 2.1 Two API layers without clear boundary

The package has:
- Legacy API: `NFXPEstimator.estimate(panel, utility, problem, transitions)` with `Panel`, `LinearUtility`, `DDCProblem` types. Lives in src/econirl/estimation/.
- Sklearn-style API: `NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")` with DataFrames. Lives in src/econirl/estimators/.
- Both are exported from __init__.py.
- The docs quickstart uses sklearn-style. The README uses legacy. The benchmarks use legacy. The tests use both.

**Work items:**
- Decide on the primary public API. The docs already position sklearn-style as recommended and legacy as deprecated. Commit to this.
- Add deprecation warnings to legacy API imports: `warnings.warn("NFXPEstimator is deprecated. Use NFXP instead.", DeprecationWarning)`.
- In __init__.py, clearly separate the public surface. The __all__ list should put sklearn-style names first and legacy names last with a comment.
- Eventually (not now), remove legacy names from __init__.py entirely and only expose them via `from econirl.estimation import NFXPEstimator`.

### 2.2 DataFrame versus Panel mismatch

The external reviewer found that `TransitionEstimator.fit(data=df, ...)` raised `AttributeError: 'DataFrame' object has no attribute 'trajectories'`, meaning it expects a Panel object despite the sklearn-like signature.

**Work items:**
- Audit every sklearn-style estimator's `fit()` method. Each must accept a pandas DataFrame as the primary input.
- Internal conversion from DataFrame to Panel should happen inside fit(), not be the caller's responsibility.
- TransitionEstimator.fit() must accept DataFrames. If it currently requires a Panel, add DataFrame support.
- Add input validation at the top of every fit() method: check that the required columns (state, action, id) exist in the DataFrame. Raise a clear ValueError with the expected column names if they don't.
- Write one integration test that does the full quickstart flow with DataFrames only, never touching Panel directly.

### 2.3 Estimator protocol enforcement

The package defines an `EstimatorProtocol` in src/econirl/estimators/protocol.py. Every sklearn-style estimator should satisfy it.

**Work items:**
- Verify that every estimator in estimators/ implements the protocol (fit, params_, se_, conf_int, predict_proba, policy_, value_, summary).
- For estimators where some attributes don't apply (e.g., neural estimators can't provide se_), they should return None or raise NotImplementedError with a clear message, not silently omit the attribute.
- Add a parametrized test that instantiates every estimator and checks protocol conformance.

### 2.4 Fitted result attributes

The quickstart promises params_, se_, conf_int(), predict_proba(), policy_, value_, and summary(). Each estimator should either provide these or clearly say why it cannot.

**Work items:**
- Create a compatibility matrix (which estimator provides which attributes).
- For neural estimators that cannot provide se_: return None and document why in the docs page for that estimator.
- For IRL estimators that recover rewards but not structural parameters: params_ should contain the recovered reward parameters (even if they are in Q-space or reward-table space), not be empty.

### 2.5 API stability labels

The package has production estimators, contrib estimators, and some experimental ones. Users cannot tell which is which.

**Work items:**
- Add a `stability` attribute to each estimator class: "stable", "experimental", or "contrib".
- In the docs, label each estimator page with its stability level.
- In __init__.py, only export stable estimators at the top level. Experimental and contrib should be accessed via `from econirl.contrib import ...` or `from econirl.estimators import ...`.

---

## Chunk 3: Test Architecture and Parameter Recovery

This chunk addresses the core scientific credibility gap: most estimators lack rigorous correctness validation.

### 3.1 Missing parameter recovery benchmarks

The benchmark file tests/benchmarks/test_parameter_recovery.py tests 9 of 22 estimators. The following focus estimators are missing:

**Structural estimators (should recover theta to tight tolerance):**

a) MPEC: Should match NFXP to within RMSE < 0.15. Both optimize the same likelihood via different numerical paths. Add test_mpec_rust_bus() following the existing pattern with _simulate_and_prepare. Import MPECEstimator from econirl.estimation. Use discount_factor=0.9999 and n_individuals=500. Also assert Bellman constraint violation < 1e-4.

b) NNES: Should recover with RMSE < 0.3. Uses neural V approximation with NPL Bellman (Neyman orthogonal). Add test_nnes_rust_bus() with bellman="npl", hidden_dim=64. Use discount_factor=0.9999. The NPL variant should be more robust to V-approximation error than the NFXP variant.

c) SEES: Should recover with RMSE < 0.5. Uses sieve basis (Fourier/polynomial) for V approximation. Add test_sees_rust_bus() with basis_dim=8, penalty_lambda=0.01. Also check that metadata["alpha"] (sieve coefficients) is finite.

**IRL estimators (recover up to scale, test cosine similarity or ratio):**

d) IQ-Learn: Cannot recover structural theta directly. With q_type="linear" and ActionDependentReward features, can recover feature-space parameters up to scale. Add test_iq_learn_rust_bus() that checks: (1) policy is valid distribution, (2) cosine similarity between recovered and true parameter vectors exceeds 0.7, (3) Bellman consistency r = Q - gamma * E[V(s')]. Use discount_factor=0.99 and ActionDependentReward.from_rust_environment(env).

**Neural estimators (cannot recover theta, test policy quality):**

e) MCE-IRL Neural: Recovers neural reward, not theta. Add test_mceirl_neural_rust_bus() that checks: (1) policy is valid distribution, (2) policy predicts expert actions better than random (accuracy > 0.6), (3) learned reward matrix is finite and bounded. Use discount_factor=0.99.

### 3.2 Tighten existing weak benchmarks

Four existing benchmark tests are too weak to validate correctness:

a) test_maxent_irl_rust_bus (line 138): Currently just checks "produces a policy" and "should produce parameters". Does not assert anything about parameter quality. Add: the policy should have higher replacement probability at high mileage (state 70+) than at low mileage (state 10). This is the basic economic prediction of the Rust bus model. Also add: if parameters are returned, operating cost parameter should be positive.

b) test_gail_rust_bus (line 256): Checks policy validity and conditional RMSE < 2.0 (a tolerance so loose it is meaningless). Add: policy replacement probability at state 80 should exceed replacement probability at state 10. This validates the method learned the basic mileage-replacement gradient.

c) test_airl_rust_bus (line 291): Same pattern as GAIL. Add the same directional economic check.

d) test_gladius_rust_bus (line 231): RMSE < 1.0 is documented as a structural limitation (40% overestimate on operating cost in IRL setting without observed rewards). Add a comment explaining this is a known limitation per CLAUDE.md. Add: result should have replacement_cost > operating_cost (the basic structure of the problem). Add: policy should show increasing replacement probability with mileage.

### 3.3 Smoke tests for completely untested estimators

Four estimators have zero tests of any kind. Each needs at minimum a smoke test that validates the estimator can be instantiated and run on a small problem.

a) f-IRL (src/econirl/estimation/f_irl.py): Create tests/test_f_irl.py with:
- test_f_irl_init: Creates FIRLEstimator without error.
- test_f_irl_estimate_small_problem: Runs on a 3-state, 2-action problem with 20 expert trajectories. Returns a result with a policy.
- test_f_irl_policy_valid: Policy rows sum to 1 and are non-negative.

b) Behavioral Cloning (src/econirl/estimation/behavioral_cloning.py): Create tests/test_behavioral_cloning.py with:
- test_bc_init: Creates BehavioralCloningEstimator without error.
- test_bc_matches_frequencies: On a simple panel where action 0 is taken 80% of the time, BC policy should give P(action 0) near 0.8.
- test_bc_policy_valid: Valid probability distribution.

c) Deep MaxEnt IRL (src/econirl/contrib/deep_maxent_irl.py): Create tests/test_deep_maxent_irl.py with:
- test_deep_maxent_init: Creates DeepMaxEntIRLEstimator without error.
- test_deep_maxent_estimate: Runs on a small problem, returns result.
- test_deep_maxent_reward_finite: Learned reward matrix has no NaN or Inf values.
- test_deep_maxent_policy_valid: Valid policy.

d) Bayesian IRL (src/econirl/contrib/bayesian_irl.py): Create tests/test_bayesian_irl.py with:
- test_bayesian_irl_init: Creates BayesianIRLEstimator without error.
- test_bayesian_irl_estimate: Runs on a small problem with few MCMC samples, returns result.
- test_bayesian_irl_posterior_finite: All posterior samples are finite.

### 3.4 E2E estimate-then-counterfactual test

No test currently chains estimation with counterfactual analysis. This is a critical gap because counterfactuals are the main value proposition.

Create tests/integration/test_estimate_counterfactual_e2e.py with:

a) test_nfxp_type3_counterfactual:
1. Simulate Rust bus data with known parameters (operating_cost=0.001, replacement_cost=3.0).
2. Estimate with NFXP and validate RMSE < 0.1.
3. Run Type 3 counterfactual: double the replacement cost to 6.0.
4. Assert counterfactual policy replaces LESS often at every state (higher RC means fewer replacements).
5. Assert the replacement threshold shifts to higher mileage.

b) test_nfxp_type2_counterfactual:
1. Same estimation.
2. Change transitions to faster deterioration (shift mileage increment distribution toward higher increments).
3. Assert replacement happens at LOWER mileage (faster deterioration means earlier replacement).

c) test_ccp_matches_nfxp_counterfactual:
1. Estimate with both NFXP and CCP on same data.
2. Run the same Type 3 counterfactual on both.
3. Assert the counterfactual policies are similar (CCP and NFXP should agree on counterfactual predictions).

### 3.5 Test infrastructure improvements

a) Fix deprecation warnings in test_parameter_recovery.py:
- Line 16: `from econirl.estimation import GAILEstimator, GAILConfig` triggers deprecation warning. Update to `from econirl.contrib.gail import GAILEstimator, GAILConfig`.
- Same for test_adversarial_estimators.py line 8.

b) The benchmark _rmse helper uses torch.Tensor. After the torch migration (Chunk 1), update to use numpy or JAX arrays.

c) Add test categorization in conftest.py or pytest markers:
- "unit": logic and invariants
- "contract": public API guarantees (sklearn protocol)
- "numerical": parameter recovery within tolerances
- "packaging": import on clean env
- "slow": nightly benchmarks

d) Add a deterministic seed fixture that is used consistently across all tests. The current conftest.py has `seed = 42` but not all tests use it.

### 3.6 CI pipeline

Create .github/workflows/ci.yml:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install .[dev]
      - run: python -c "import econirl; print(econirl.__version__)"
      - run: python -m pytest tests/ -v -m "not slow" --timeout=120
  
  smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install .
      - run: |
          python -c "
          import econirl
          from econirl import NFXP, CCP
          from econirl.datasets import load_rust_bus
          df = load_rust_bus()
          model = CCP(discount=0.99).fit(df, state='mileage_bin', action='replaced', id='bus_id')
          print(model.params_)
          "
```

---

## Chunk 4: Documentation Alignment and Completeness

This chunk addresses docs drift, missing estimator docs, and the gap between what the docs promise and what the code delivers.

### 4.1 Fix installation.rst

See Chunk 1 item 1.4 for the specific line-level fixes. After the torch migration:
- Replace "PyTorch >= 2.0" with the actual JAX dependency list.
- Fix clone URL from `github.com/econirl/econirl.git` to `github.com/rawatpranjal/econirl.git`.
- Add a "Verifying Backend" section that shows how to check float64 is enabled.

### 4.2 Fix orphaned IQ-Learn page

docs/estimators/iq_learn.md exists with full content (background, key equations, strengths/limitations) but is NOT listed in docs/estimators.md toctree. The main index.rst table at line 72 references IQ-Learn, so users see it listed but cannot navigate to its page.

**Work item:** Add `estimators/iq_learn` to docs/estimators.md toctree between `estimators/gladius` and `estimators/bc`:

```
estimators/gladius
estimators/iq_learn
estimators/bc
```

### 4.3 Add GLADIUS to estimator guide

docs/estimators_guide_page (the estimator selection decision flowchart) covers 9 estimators but excludes GLADIUS. GLADIUS is one of the focus estimators and has a full docs page. It should appear in the guide.

**Work item:** Add a GLADIUS section to the estimator guide. Position it after AIRL in the "Model-Free / Neural" category. Key points to cover:
- When to use: continuous state spaces where tabular methods cannot be applied, or when rewards are observed in the data.
- Limitation: in IRL setting (without observed rewards), recovers replacement cost within 8 percent but overestimates operating cost by about 40 percent due to structural bias from the Q-identification-up-to-constant issue.
- Contrast with AIRL: GLADIUS uses bi-conjugate Bellman error, AIRL uses adversarial discriminator. Both are model-free but have different identification properties.

### 4.4 Create docs pages for undocumented estimators

The following estimators have code but no documentation page. Each needs at minimum a stub page with: one-paragraph description, key equation, when to use it, limitations, and a code example.

Priority order (based on whether users are likely to encounter them):

a) MPEC: Structural estimator, same likelihood as NFXP but different optimization. Users who know MPEC from Su and Judd (2012) will look for it. Create docs/estimators/mpec.md.

b) GCL (Guided Cost Learning): In contrib/ and estimators/. Used in the ddc-irl-equivalence example. Create docs/estimators/gcl.md.

c) MaxEnt IRL: Distinct from MCE-IRL (uses state-only features, does not account for causal structure). Has its own sklearn wrapper. Create docs/estimators/maxent_irl.md. Explain the difference from MCE-IRL.

d) Max Margin IRL: Has sklearn wrapper and integration tests. Create docs/estimators/max_margin_irl.md.

e) MCE-IRL Neural: Deep variant of MCE-IRL with neural reward network. Brief mention exists under MCE-IRL page but deserves its own page. Create docs/estimators/mceirl_neural.md.

f) NeuralAIRL: Currently covered under AIRL page. Should have its own page explaining the neural discriminator variant versus tabular AIRL. Create docs/estimators/neural_airl.md.

g) AIRL-Het: Specialized variant with EM heterogeneity and anchor identification from Lee, Sudhir and Wang (2026). Create docs/estimators/airl_het.md.

After creating these pages, add them to docs/estimators.md toctree.

### 4.5 Update counterfactual support matrix

docs/counterfactuals.rst has a table showing which estimators support which counterfactual types, but it only covers 8 estimators (NFXP, CCP, NNES, TDCCP, MCE-IRL, NeuralGLADIUS, NeuralAIRL, BC). The following estimators are missing from the matrix:

| Estimator | Type 1 | Type 2 | Type 3 | Type 4 | Rationale |
|-----------|--------|--------|--------|--------|-----------|
| MPEC | Yes | Yes | Yes | Yes | Same as NFXP (structural, recovers theta) |
| SEES | Yes | Yes | Yes | Yes | Same as NFXP (structural, recovers theta) |
| IQ-Learn | Yes | Partial | No | Partial | Recovers Q not theta. Type 2 possible via tabularization. No Type 3 (no interpretable params). |
| MCE-IRL Neural | Yes | Yes (tabularized) | No | Yes (tabularized) | Same as NeuralGLADIUS/NeuralAIRL |
| MaxEnt IRL | Yes | Yes | Yes | Yes | Linear reward, recovers params |
| Max Margin IRL | Yes | Yes | Yes | Yes | Linear reward, recovers params |
| GCL | Yes | Yes (tabularized) | No | Yes (tabularized) | Neural reward |
| GAIL | Yes | No | No | No | Policy only, no reward recovery |
| f-IRL | Yes | Partial | No | Partial | Tabular reward, no structural params |
| AIRL-Het | Yes | Yes (tabularized) | No | Yes (tabularized) | Per-segment neural reward |
| Deep MaxEnt | Yes | Yes (tabularized) | No | Yes (tabularized) | Neural reward |
| Bayesian IRL | Yes | Yes | Partial | Yes | Posterior over rewards, can do posterior counterfactuals |

**Work item:** Add these rows to the table in docs/counterfactuals.rst.

### 4.6 List undocumented examples in index

docs/examples/index.rst lists 20 examples across 8 categories. But these example directories exist and are not listed:

- examples/ddc-irl-equivalence/ (3 files, theoretical comparison of DDC and IRL)
- examples/ddc_eda/ (1 file, data exploration)
- examples/eda/ (7 files, exploratory data analysis suite)
- examples/gym-irl/ (4 files, GLADIUS and NeuralAIRL on Gym environments)
- examples/mce-irl/ (3 files, MaxMargin and MCE comparison)
- examples/ziebart-mce-irl/ (2 files with README, original paper replication)

**Work item:** Add these to docs/examples/index.rst under appropriate categories. The gym-irl examples should go under a new "Gym Environments" category. The ddc-irl-equivalence and ziebart-mce-irl examples should go under a "Theory and Replication" category.

### 4.7 Failure mode documentation

The docs do not explain what to do when things go wrong. Users of structural estimation tools need this.

**Work item:** Create docs/troubleshooting.rst (or .md) covering:
- NFXP not converging: check feature normalization, try lower discount factor for testing, increase inner_max_iter.
- MCE-IRL oscillating: reduce learning rate, check feature scaling.
- Standard errors invalid: check Hessian condition number, verify identification.
- Float64 warnings: explain that econirl enables x64 automatically, but if running JAX before importing econirl, x64 may not be set.
- GLADIUS operating cost bias: explain the structural identification issue (Q identified up to state-dependent constant).
- IRL reward scale: explain that IRL rewards are identified only up to additive constants and scale (Kim et al. 2021).

### 4.8 Executable docs check

After all doc fixes, every code block in the docs should be verifiable.

**Work item:** Add a CI step (or Makefile target) that extracts all Python code blocks from .rst and .md files and runs them. Start with just the quickstart and installation verification blocks.

### 4.9 Docs build after changes

After all documentation changes, build locally and verify:
```bash
python3 -m sphinx -b html docs docs/_build/html
```

Then trigger a manual RTD build (webhook is broken):
```bash
curl -X POST -H "Authorization: Token $RTD_TOKEN" https://readthedocs.org/api/v3/projects/econirl/versions/latest/builds/
```

---

## Chunk 5: Papers, Examples, and Research Infrastructure

This chunk addresses the research completeness gaps: missing papers, missing examples for key estimators, and the docling pipeline for searchability.

### 5.1 Fix empty paper placeholders

Three papers in papers/priority/ are 0-byte placeholder files:

a) papers/priority/hotz_miller_1993_ccp.pdf (0 bytes) — Hotz and Miller (1993), "Conditional Choice Probabilities and the Estimation of Dynamic Models." This is the foundational CCP paper. Must be downloaded.

b) papers/priority/su_judd_2012_mpec.pdf (0 bytes) — Su and Judd (2012), "Constrained Optimization Approaches to Estimation of Structural Models." The foundational MPEC paper. Must be downloaded.

c) papers/priority/adusumilli_eckardt_2025_td_ccp.pdf (0 bytes) — Adusumilli and Eckardt (2025) on Temporal Difference CCP. The base paper for the TD-CCP estimator. Must be downloaded.

**Work item:** Download the actual PDFs for these three papers and replace the empty placeholders.

### 5.2 Reorganize Rust (1987)

The Rust (1987) paper ("Optimal Replacement of GMC Bus Engines") is present at papers/ore/appendix_papers/Rust (1987) Bus Engines.pdf but is NOT in papers/foundational/. Since this is the single most important reference paper for the package (the Rust bus environment is the primary test case), it should be in the foundational directory.

**Work item:** Copy papers/ore/appendix_papers/Rust (1987) Bus Engines.pdf to papers/foundational/rust_1987_optimal_replacement.pdf.

### 5.3 Docling remaining priority papers

Only 5 of approximately 15 key papers have been converted to searchable markdown. The docling'd papers are:
- Fu et al. (2018) AIRL
- Cao et al. (2021) Identifiability
- Kalouptsidi et al. (2021) Counterfactual ID
- Garg et al. (2022) IQ-Learn
- Kang et al. (2025) GLADIUS/ERM-IRL

The following priority papers need to be docling'd (converted to markdown for searchability):

a) Rust (1987) — NFXP foundational paper
b) Hotz and Miller (1993) — CCP foundational paper (after downloading)
c) Ziebart (2010) — MCE-IRL (present at papers/priority/ziebart_2010_mce_irl.pdf)
d) Su and Judd (2012) — MPEC (after downloading)
e) Ho and Ermon (2016) — GAIL (present at papers/priority/ho_ermon_2016_gail.pdf)
f) Finn and Abbeel (2016) — GCL (present at papers/priority/finn_abbeel_2016_gcl.pdf)
g) Ratliff (2006) — Max Margin IRL (present at papers/priority/ratliff_2006_max_margin.pdf)
h) Lee, Sudhir and Wang (2026) — AIRL-Het (present in papers/literature/)
i) Adusumilli and Eckardt (2025) — TD-CCP (after downloading)

**Work item:** Run docling on each paper to produce a .md file alongside the PDF. Place the markdown files in papers/foundational/ alongside existing ones. Follow the naming convention: YYYY_short_description.md.

Important: Run docling conversions one at a time, never in parallel (per project feedback memory).

### 5.4 Find and add missing estimator papers

Two estimators do not have their base paper clearly in the repo:

a) NNES: The base paper is by Hoang Nguyen (2025). It may be implied by the GLADIUS paper or be a working paper. Check with Hoang. If available, add to papers/foundational/.

b) SEES: The base paper may be Kasahara and Shimotsu or another sieve estimation reference. The src/econirl/estimation/sees.py docstring should reference the paper. Add the correct paper to papers/foundational/.

### 5.5 Add examples for estimators with zero examples

Three focus estimators have zero working examples:

a) IQ-Learn: Create examples/iq-learn/iq_learn_rust_bus.py. Should demonstrate:
- Loading Rust bus data
- Fitting IQ-Learn with linear Q mode
- Extracting recovered rewards
- Comparing policy with NFXP policy
- Showing Bellman consistency of recovered rewards

b) f-IRL: Create examples/f-irl/f_irl_gridworld.py. Should demonstrate:
- Setting up a small gridworld
- Generating expert demonstrations
- Recovering tabular reward
- Plotting recovered reward heatmap vs true reward

c) MPEC: Create examples/rust-bus-engine/mpec_rust_bus.py. Should demonstrate:
- Loading Rust bus data
- Fitting MPEC
- Comparing parameters with NFXP (should be near-identical)
- Showing Bellman constraint satisfaction

### 5.6 Add examples for estimators with only one example

These estimators have only one example (all on Rust bus). They need a second example on a different problem to demonstrate generality:

a) TD-CCP: Add examples/citibike-usage/tdccp_citibike.py or similar on a different dataset.

b) SEES: Add examples/frozen-lake/sees_frozen_lake.py or similar.

c) BC: Add examples/frozen-lake/bc_frozen_lake.py showing BC as a baseline with comparison to a structural estimator.

### 5.7 Numerical diagnostics and estimator guarantees

The external reviewer noted that "a reliable library in this domain must specify exactly when standard errors are valid, what optimizer was used, what convergence means, what default regularization exists, and how sensitive results are to initialization and scaling."

**Work item:** For each estimator docs page, add a "Diagnostics and Guarantees" section covering:
- Identification conditions (what assumptions must hold)
- Convergence criterion (what does "converged" mean for this estimator)
- Standard error availability (which SE methods are valid)
- Known limitations (bias, scale sensitivity, state space requirements)
- Default hyperparameters and their rationale

This can be done incrementally, starting with the 9 core estimators that already have docs pages.

### 5.8 Expose diagnostics in API

The fitted result objects should expose diagnostic information, not just final parameters.

**Work item:** Ensure every estimation result includes:
- result.converged (bool)
- result.n_iterations (int)
- result.objective_value (float, final log-likelihood or loss)
- result.convergence_info (dict with optimizer-specific details)
- result.metadata (dict with estimator-specific outputs like Q-tables, reward matrices, Hessian condition number)

Most of these likely already exist in the EstimationResult class. Audit and fill in any gaps.

---

## Priority Ordering Across Chunks

If resources are limited, work in this order:

Tier 1 (must fix before telling anyone to use the package):
- Chunk 1 items 1.1 through 1.4 (backend consolidation, silent imports, float64, install docs)
- Chunk 2 item 2.2 (DataFrame vs Panel mismatch)
- Chunk 4 item 4.2 (orphaned IQ-Learn page)

Tier 2 (must fix before calling it a reliable library):
- Chunk 1 items 1.5 and 1.6 (README alignment, CI)
- Chunk 2 items 2.1 and 2.3 (API boundary, protocol enforcement)
- Chunk 3 items 3.1 through 3.4 (all parameter recovery tests)
- Chunk 4 items 4.1 and 4.3 through 4.5 (install docs, GLADIUS guide, new estimator docs, counterfactual matrix)

Tier 3 (must fix before calling it mature):
- Chunk 3 items 3.5 and 3.6 (test infrastructure, CI pipeline)
- Chunk 4 items 4.6 through 4.9 (examples index, troubleshooting, executable docs)
- Chunk 5 all items (papers, examples, diagnostics)
