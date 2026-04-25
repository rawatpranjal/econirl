# Plan: shape-shifter alignment validation

## Headline claim

On a single shape-shifting synthetic DGP whose ground truth is exact
along eight axes (linear or neural reward, linear or neural state
features, action-dependent or state-only features, stochastic or
deterministic transitions, stochastic or deterministic rewards,
finite or infinite horizon, discount factor, state dimension), each
of the fifteen estimators in the package recovers ground truth in
the regimes its source paper claims to support and fails predictably
in the regimes its source paper does not. The capability table in
the JSS paper (Section 3) documents both halves of this claim.

## Estimators in scope

Twelve focus estimators plus MPEC, BC, and AIRL-Het. NFXP, CCP,
MPEC, MCE-IRL, NNES, SEES, TD-CCP, GLADIUS, AIRL, IQ-Learn, f-IRL,
BC are run on the spine cell. Subsequent axis-flip cells run only
the subset declared by the paper as supporting that regime; cells
where the paper says "not supported" are documented as expected
skips, not failures, and do not appear in the per-cell estimator
list in matrix.py.

## Cell layout

Twelve cells, each a single-axis flip from a spine config. Cell ids
are `tier4_<suffix>_<estimator_slug>`.

| Suffix | Spine override | Estimators run | Purpose |
| --- | --- | --- | --- |
| spine | none | all 12 (BC..TD-CCP) | Baseline; everyone should pass |
| state_only | action_dependent=False | classical 4 + MCE-IRL | MCE-IRL is the canonical failure |
| det_T | stochastic_transitions=False | all 12 | Tests degeneracy handling |
| stoch_r | stochastic_rewards=True | structural 6 | Should be unaffected (epsilon integrated out) |
| finite | num_periods=20 | finite-capable 5 | Backward induction path |
| high_gamma | discount_factor=0.999 | inner-solver-stress 5 | Validates hybrid_iteration switch |
| large_S | num_states=128 | tabular-vs-neural 6 | Identifies threshold where neural pays off |
| neural_r | reward_type=neural | neural-reward IRL 6 | AIRL, f-IRL, deep MCE-IRL, NNES, GLADIUS, TD-CCP |
| neural_phi | feature_type=neural | feature-consuming 11 | All but neural-reward-only methods |
| neural_r_phi | both neural | neural-reward IRL 6 | Hardest case |
| multi_action | num_actions=6 | multi-action capable 6 | Tests CCP/MCE-IRL action scaling |
| product_state | state_dim=2, num_states=8 | product-state capable 5 | Tests mixed-radix encoding |

Spine config: `num_states=32, num_actions=3, num_features=4,
discount_factor=0.95, reward_type=linear, feature_type=linear,
action_dependent=True, stochastic_transitions=True,
stochastic_rewards=False, state_dim=1, seed=0`.

## Inference validity classification

Same per-estimator classification as plan_rust_small.md. The shape-
shifter does not introduce new SE methods; it stresses the existing
ones across regimes. The CSV per fit carries the SE method label so
inference validity is auditable from the artifact.

## Metrics reported per fit

- Point estimate per parameter (linear-reward cells only).
- Standard error per parameter, plus the SE method label.
- 95 percent confidence interval.
- In-sample log-likelihood.
- Policy KL: KL divergence between the estimated logit choice
  probabilities and the optimal logit choice probabilities under the
  true reward, averaged over visited states.
- Reward correlation: Pearson correlation between the estimated R(s,
  a) tensor and the true R(s, a) tensor (handles neural-reward cells
  where there is no theta to compare).
- Cosine similarity (linear-reward cells only).
- Wall-clock time and converged flag.
- Failure mode column.

## Monte Carlo extension

R = 20 replications per cell. Replications use seed = 42 + r and
both the env and the simulation re-seed, so the Monte Carlo also
varies the frozen-network weights and transition matrices across
replications. This ensures the variance estimate reflects real
sampling variability across DGP draws, not just trajectory noise.

## Compute spec

Twelve axis cells with up to twelve estimators each. Counted:

- spine: 12 estimators × 20 reps × 60s = 14400s ≈ 4h CPU.
- state_only, det_T, multi_action, product_state, large_S: similar
  per-cell budgets, totalling ~12h CPU.
- finite: 5 estimators × 20 reps × 60s = 6000s ≈ 1.7h CPU.
- stoch_r: 6 estimators × 20 reps × 60s = 7200s ≈ 2h CPU.
- high_gamma: 5 estimators × 20 reps × 300s = 30000s ≈ 8h CPU.
- neural_r, neural_phi, neural_r_phi: 6 + 11 + 6 estimators ×
  20 reps × 600-900s = 14h GPU.

Sequential single-pod wall-clock around 36-40h. At 8-way RunPod
parallelism the wall-clock floors at the high_gamma cell at roughly
2h CPU plus the 14h GPU tier in serial, total around 16h. Cost at
RunPod community rates around 25-35 USD.

## Acceptance criteria

The plan succeeds when each of the following holds.

- All 12 estimators on the spine cell return converged = True on at
  least 90 percent of replications, and policy KL below 0.05.
- Linear-reward classical estimators (NFXP, CCP, MPEC) on the spine
  cell achieve relative parameter error below 5 percent on every
  theta_i and 95 percent CI coverage in [0.85, 0.99].
- MCE-IRL on the state_only cell either fails or produces a flat
  reward; this is the canonical failure mode and is the headline
  example for the unidentification footnote in the paper.
- CCP with NPL=10 on the state_only and det_T cells matches NFXP
  within 5 percent (per the existing CCP NPL audit).
- Neural-reward IRL methods on the neural_r cell achieve reward
  correlation above 0.7; classical estimators on the same cell are
  expected to fail and their failure_mode column is documented.
- All five finite-horizon-capable estimators on the finite cell
  return converged = True; non-finite-capable estimators do not run.
- Inner-solver-heavy methods (NFXP-NK, MPEC, hybrid_iteration users)
  on the high_gamma cell converge in a wall-clock budget below 5x
  the spine-cell wall-clock.
- Each ✗ in the empirical capability table (Section 3, Table 3) is
  traceable to a specific cell+estimator failure_mode entry in the
  artifact CSV.

## Failure modes documented in the artifact

Same `failure_mode` column convention as the other plans. Allowed
values: `none`, `did_not_converge`, `parameter_drift`,
`policy_drift`, `inference_unsupported`, `runtime_exceeded`,
`identification_collapsed`, `reward_uncovered`. The last two are
shape-shifter specific and capture the regimes where an estimator
runs but produces an uninformative answer.

## What the paper says now

Until this plan fires, Section 3.4 of the paper carries the
hand-written theoretical capability table (Table 2) and a
placeholder Table 3 marked
`\todo[inline]{Generated by emit_capability_table.py after Tier 4 fires}`.
Once the plan fires, Table 3 is auto-generated from the per-cell
artifacts by `experiments/jss_deep_run/emit_capability_table.py`,
which writes
`papers/econirl_package/tables/capability_empirical.tex` for the
paper to `\input{}`.

## Artifact paths

- `experiments/jss_deep_run/results/shapeshifter/<cell_id>.csv` per
  cell, R rows each.
- `experiments/jss_deep_run/results/shapeshifter/headline.csv` one
  row per (estimator, axis-cell), Monte Carlo summary.
- `papers/econirl_package/tables/capability_empirical.tex` the LaTeX
  snippet consumed by Section 3 Table 3.

## Run command

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 4 --image econirl-deep-run:v1 --max-parallel 8
```

For a smoke test on the spine cell only:

```
python -m experiments.jss_deep_run.cloud_test \
    --module experiments.jss_deep_run.run_cell \
    --shell "python -m experiments.jss_deep_run.run_cell --cell-id tier4_spine_nfxp" \
    --max-spend-usd 2
```
