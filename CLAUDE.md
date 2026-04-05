# CLAUDE.md - Project Context for Claude Code


# How to report and validate own work: 

When a check did pass or a task is complete, state it plainly — do not hedge confirmed results with unnecessary disclaimers, downgrade finished work to 'partial,' or re-verify things you already checked.

Never claim 'all tests pass' when output shows failures. Never suppress or simplify failing checks to manufacture a green result. Never characterize incomplete or broken work as done.

## Writing Rules

Write in complete sentences. Do not use em dashes, colons, semicolons, plus signs, equals signs, or brackets in prose. Do not write lists or bullet points in documentation prose. Every paragraph should read as flowing English. Do not overload sentences with multiple ideas. Say one thing per sentence.

Do not repeat what a table already shows. Prose should interpret and explain, not describe or restate. If something is not important enough to call out specifically, cut it. Do not add hedging language or qualifications unless they change the meaning.

Do not use jargon in user-facing documentation. Technical terms belong in code comments and appendices. The case study pages are for practitioners, not developers.

When reporting results, write the number into the sentence naturally. Do not use shorthand like "+1.6%" or "r=0.77" in prose. Write "the lift is 1.6 percent" or "the correlation is 0.77".

Write heavy code comments in the code itself as you refine your understanding, with sources from papers in the repo.

## Pre-Estimation Diagnostics

Before running any estimator on a new dataset, always run and report these checks.

Feature matrix rank: compute `np.linalg.matrix_rank` on the flattened feature matrix `(n_states * n_actions, n_features)`. If rank is less than `n_features`, the model is under-identified before even looking at data. Drop or redesign collinear features. Common trap: two action-specific intercepts that are both 1 for action A and 0 otherwise are identical columns.

Condition number: compute `np.linalg.cond` on the nonzero rows of the feature matrix. Values above 1e6 signal near-collinearity that will inflate standard errors even if the matrix is technically full rank.

State coverage: report how many of the `n_states` states have at least one observation. States with zero observations produce degenerate CCP estimates.

Single-action states: count states where only one action is ever observed. CCPs at the boundary (0 or 1) are structurally degenerate and produce uninformative Bellman updates.

Print all four diagnostics before calling `estimator.estimate()`. If the feature matrix is rank deficient, stop and fix it before running.

## Benchmarking and Case Studies

When running ML benchmarks across datasets, use 5-fold cross-validation as the definitive estimate. Single holdout splits overfit at small N and produce inflated lifts. Always report in-sample versus out-of-sample metrics so overfitting is visible. When a result does not replicate under CV, say so honestly.

Start with 250 users to verify the pipeline works end to end. Then scale to 2000 for quick results. Only run full scale after the code is stable. Save results per dataset in JSON files so runs can be resumed and nothing is lost.

When the numbers say the lift is zero, say the lift is zero. Do not dress up a null result. A null result with interpretable features is more useful than a positive result with opaque ones.

## Project Vision

The aspiration is EconML for DDC/IRL. The estimators are interchangeable parts. The value is in the shared post-estimation infrastructure that sits on top of all of them. Pick any estimator and get inference (standard errors, confidence intervals, Wald tests, identification diagnostics), validation (in-sample, out-of-sample, transfer), and counterfactual simulation (parameter changes, transition changes, elasticity analysis) through the same unified pipeline.

Sieve compression is a post-estimation step, not an estimation method. Estimate a high-dimensional reward with AIRL or deep MaxEnt, then project it onto a sieve basis (polynomials, splines) for interpretability. The neural step gets accuracy, the sieve step gets understanding.

Structural estimators (NFXP, CCP, NNES) use linear utility by design. Neural reward plug-and-play applies only to IRL estimators (AIRL, deep MaxEnt, GLADIUS). Do not add neural reward support to structural estimators.

Examples and showcases should run on real data, not simulated data. Use simulation only when ground truth is needed for parameter recovery tests.

## Project Overview

**econirl** is a Python library for structural estimation and inverse reinforcement learning (IRL) in dynamic discrete choice models. It bridges econometrics (NFXP, CCP) with machine learning (MaxEnt IRL, MCE IRL, GAIL, AIRL).

## Key Architecture

```
src/econirl/
├── core/           # Types (DDCProblem, Panel, Trajectory), Bellman operators, solvers
├── estimation/     # 10 estimators: NFXP, CCP, MaxEnt IRL, MCE IRL, Max Margin,
│   │                 GCL, TD-CCP, GLADIUS, GAIL, AIRL
│   └── adversarial/  # GAIL and AIRL implementations
├── environments/   # RustBus, MultiComponentBus, Gridworld
├── preferences/    # LinearUtility, ActionDependentReward, NeuralCost
├── inference/      # Standard errors, identification diagnostics, EstimationSummary
├── simulation/     # Synthetic data generation, Monte Carlo, counterfactuals
└── visualization/  # Policy and value function plots
```

Each subdirectory has its own CLAUDE.md with detailed interface documentation. See:
- `docs/estimator_guide.md` - **Estimator selection guide**: why each of the 9 core estimators exists, paper-backed theorems, and when to use which
- `src/econirl/core/CLAUDE.md` - Types, Bellman operator, solvers
- `src/econirl/estimation/CLAUDE.md` - All 10 estimators and base contract
- `src/econirl/environments/CLAUDE.md` - Environment base class and 3 implementations
- `src/econirl/preferences/CLAUDE.md` - Utility function protocol and LinearUtility
- `src/econirl/inference/CLAUDE.md` - SE methods, identification, EstimationSummary
- `src/econirl/simulation/CLAUDE.md` - Data generation, Monte Carlo, counterfactuals
- `tests/CLAUDE.md` - Test conventions, fixtures, running tests

## Focus Estimators (Replication Priority)

The following estimators are the core focus for real-data replication and benchmarking:

1. **NFXP-NK** — Nested Fixed Point with SA→NK polyalgorithm (Iskhakov et al. 2016). BHHH optimizer, analytical gradient via implicit differentiation. The main structural estimator.
2. **CCP** 
2. **NNES** — Neural Network Estimation of Structural models (neural version of NFXP)
7. **TD-CCP** — Temporal Difference CCP (AE paper 2022, neural approximate VI) (VERY IMP)
3. **SEES** — Simulation-based Estimation of Economic Structural models
4. **MCE-IRL** — Maximum Causal Entropy IRL (Ziebart 2010). Feature matching with soft Bellman.
5. **MCE-IRL (Deep)** — Deep MaxEnt with neural reward function (in the style of imitation learning)
6. **GLADIUS** — Model-free DDC estimation (Q-network + EV-network)
8. **AIRL** — Adversarial IRL (Fu et al. 2018). Reward recovery via discriminator.

Each estimator should be validated on at least one real-data replication (Rust bus, Keane-Wolpin, NGSIM, etc.) before being considered production-ready.

## Critical Implementation Details

### MCE IRL Expected Features (IMPORTANT)
The `_compute_expected_features()` method in `mce_irl.py` uses occupancy measures (state visitation frequencies under the current policy) following Ziebart (2010) Algorithm 1 and the `imitation` library. The E_π term is: `E_π[φ] = Σ_s D_π(s) Σ_a π(a|s) φ(s,a,k)` where D_π is computed via the forward pass. This correctly handles both action-dependent and state-only features. Falls back to empirical-state iteration only when transitions are unavailable.

### Parameter Identification in IRL
- Rewards are only identified up to constants (Kim et al. 2021, Cao & Cohen 2021)
- For well-conditioned optimization, normalize parameters to unit norm
- Normalize features to [-1, 1] range
- Rust bus parameters (0.001, 3.0) have poor scaling - consider normalization

### GLADIUS Without Observed Rewards (IRL Setting)
GLADIUS trains Q by NLL only, which identifies Q up to a state-dependent constant c(s). That constant leaks into the implied rewards r(s,a) through the transition structure asymmetrically. For the bus engine, maintaining stays near state s while replacing jumps to state 0, so the constant propagates differently per action. Action-difference projection removes a global constant but not this state-dependent one, producing a systematically flat reward-difference curve. On the Rust bus with beta of 0.95, GLADIUS recovers replacement cost within 8 percent but overestimates operating cost by about 40 percent regardless of network size, training length, or data volume. This is a structural bias, not a tuning problem. NFXP solves the Bellman equation exactly and recovers both parameters within 5 percent. Use GLADIUS for continuous-state environments where tabular methods cannot be applied, or when rewards are observed in the data (the paper's intended use case, where the bi-conjugate Bellman error anchors Q-values). Do not pass Bellman gradients through V_Q to Q in the IRL setting because this causes Q-value explosion without observed rewards to anchor the scale.

### Transition Matrix Conventions
- Estimators expect: `(n_actions, n_states, n_states)` i.e., `transitions[a, s, s']`
- Some internal code uses: `(n_states, n_actions, n_states)` i.e., `transitions[s, a, s']`
- Always check the expected format when passing transitions

## Testing Commands

```bash
# Run quick tests (skip slow)
python3 -m pytest tests/ -v -m "not slow"

# Run all tests
python3 -m pytest tests/ -v

# Run specific test file
python3 -m pytest tests/test_mce_irl_action_features.py -v

# Run with coverage
python3 -m pytest tests/ --cov=econirl
```

## Common Debugging

### MCE IRL not converging
1. Check feature normalization (should be [-1, 1])
2. Check parameter initialization (don't use zeros)
3. Verify inner loop (soft VI) converges - increase `inner_max_iter` for high gamma
4. Try smaller learning rate if oscillating

### NFXP slow convergence
1. High discount factor (gamma > 0.99) requires many inner iterations
2. Consider lowering gamma for testing
3. Check that transitions are properly normalized (rows sum to 1)

## Benchmarking Philosophy

Controlled simulated experiments use in-sample, out-of-sample, and out-of-transfer evaluation. Benchmarks are modular (`ESTIMATORS` dict) so new algorithms slot in with one entry. Use high trajectory counts — the goal is not standard errors but verifying that estimators correctly recover reward parameters and produce good policies. Working examples live in `examples/`.

MCE IRL and NFXP converge to the same answer with enough data and proper features — with 2000 trajectories on a 5x5 gridworld both achieve cosine similarity 0.9999 to true parameters and identical policy performance across in-sample, out-of-sample, and all three transfer scenarios. Features MUST be action-dependent (vary across the choice set) for NFXP identification; state-only features that are identical across actions make R(s,a) constant across actions, collapsing the likelihood surface and causing parameter blowup. IRL rewards are identified only up to additive constants and scale (Kim et al. 2021), so evaluate on cosine similarity and policy quality rather than raw RMSE. MaxEnt IRL underperforms MCE IRL even in deterministic environments because it doesn't account for causal structure in the state visitation computation.

## Relationship to HumanCompatibleAI/imitation

The `imitation` library (Gleave, Toyer et al.) is the most mature open-source IRL library. econirl borrows the direct linear solve for occupancy measures in `core/occupancy.py` and the MCE IRL feature matching framework from the Gleave and Toyer (2022) primer. econirl's infinite-horizon formulation with action-dependent features is more general than imitation's finite-horizon tabular MCE IRL.

econirl's core differentiators over imitation are statistical inference (standard errors, Wald tests, identification diagnostics), counterfactual analysis (welfare elasticity, policy simulation), structural estimators (NFXP, CCP, NNES), and the unified post-estimation pipeline. imitation has none of these. Do not adopt imitation's Stable Baselines 3 dependency or Gymnasium environment wrappers. Soft value iteration is the correct policy solver for structural estimation.

MCE IRL uses dual stopping criteria following imitation. The gradient path checks both gradient norm and occupancy distance (L-infinity between demo and policy state visitation). The `occupancy_tol` parameter in `MCEIRLConfig` controls the occupancy distance threshold.

`BehavioralCloningEstimator` serves as the lower bound baseline for every benchmark. Any estimator that cannot beat BC is not learning from MDP structure. BC is just weighted logistic regression on state-action frequencies.

`RunningNorm` in `preprocessing/running_norm.py` provides numerically stable online mean and variance tracking (Chan, Golub, and LeVeque 1979) for feature normalization during training. Use it when feature columns span different scales.

`Panel.save_npz()` and `Panel.load_npz()` serialize panel data to compressed NumPy archives for reproducibility.

## ReadTheDocs

The documentation is hosted at https://econirl.readthedocs.io. The Sphinx config lives in `docs/conf.py` and the RTD config in `.readthedocs.yaml`. The GitHub webhook for automatic builds is broken. After pushing documentation changes, you must manually trigger a build on RTD by running `curl -X POST -H "Authorization: Token $RTD_TOKEN" https://readthedocs.org/api/v3/projects/econirl/versions/latest/builds/` or by going to https://app.readthedocs.org/projects/econirl/builds/ and clicking "Build Version". Do not assume pushing to main will trigger a build.

After triggering a build, RTD takes 2 to 3 minutes to finish. The CDN can cache old pages for several minutes after the build finishes. Hard refresh (Cmd+Shift+R) or appending `?v=X` to the URL busts the browser cache. If the site still shows stale content, check the build status at https://app.readthedocs.org/projects/econirl/builds/ before debugging further.

Place figures where they logically belong in the narrative flow, not at the top as hero images. Each figure should appear immediately after the prose that motivates it, so the reader sees the explanation first and the visualization second. Do not front-load pages with decorative images.

Example page titles must follow a consistent "Subject Problem" pattern with no colons, semicolons, parentheses, or subtitles. Write "Objectworld Deep MaxEnt IRL" not "Deep MaxEnt IRL: Objectworld and Binaryworld".

Tables in documentation must show actual numbers from running the code. Do not use qualitative placeholders like "positive" or "near zero" when you can run the script and report the estimate.

After making documentation changes, commit and push, then trigger an RTD build with `curl -X POST -H "Authorization: Token $RTD_TOKEN" https://readthedocs.org/api/v3/projects/econirl/versions/latest/builds/`. Do not build docs locally.

Keep `docs/conf.py` release version in sync with `pyproject.toml` and `src/econirl/__init__.py`.

## Key References

- Rust (1987): Optimal Replacement of GMC Bus Engines
- Ziebart (2010): Maximum Causal Entropy IRL
- Kim et al. (2021): Reward Identification in IRL
- Cao & Cohen (2021): Identifiability in IRL
- Lee, Sudhir & Wang (2026): AIRL with unobserved heterogeneity for structural DDC on sequential content
