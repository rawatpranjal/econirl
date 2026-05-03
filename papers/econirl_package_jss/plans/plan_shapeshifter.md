# Plan: shape-shifter alignment validation

## Headline claim

On a single shape-shifting synthetic DGP whose ground truth is exact
along the stress axes that matter for the package (state-only versus
action-dependent rewards, low- versus high-dimensional rewards, low-
versus high-dimensional states, stochastic versus deterministic
transitions, finite versus infinite horizon, discount factor, latent
segments, and Type A/B/C counterfactual interventions), each of the
12 focus estimators recovers the truth in the regimes its source
paper claims to support and fails predictably in the regimes its
source paper does not. The capability table in the JSS paper
(Section 3) documents both halves of this claim.

## Estimators in scope

The validation scope is exactly 12 estimators: NFXP, CCP, MPEC,
MCE-IRL, NNES, SEES, TD-CCP, GLADIUS, AIRL, IQ-Learn, f-IRL, and
AIRL-Het. BC is a diagnostic baseline, not part of the 12-estimator
truth-recovery claim. Subsequent axis-flip cells run only the subset
declared by the paper as supporting that regime; cells where the
paper says "not supported" are documented as expected skips, not
failures.

## Code placement and file discipline

This plan is deliberately sparse in the codebase. The known-truth
DGP is experiment infrastructure, not a public package API.

- `experiments/known_truth.py`: the single compact harness for DGP
  construction, exact Bellman truth, pre-estimation checks, estimator
  contracts, recovery gates, artifact writing, and the local/RunPod
  CLI.
- `tests/test_known_truth.py`: the only fast local test surface for
  DGP identities, counterfactual oracle identities, estimator
  contracts, NFXP smoke metrics, and hard-gate failure behavior.
- `src/econirl/estimation/<estimator>.py`: only estimator fixes live
  in package source. The synthetic harness is not exported from
  `econirl.simulation` and does not create a
  `src/econirl/simulation/known_truth/` subtree.
- `papers/econirl_package_jss/plans/alignment/<NN>_<estimator>.md`:
  one paper-to-code audit per estimator. These are planning/audit
  files only.
- `docs/estimators/<estimator>.md`: final tutorial pages are written
  at the end, after the estimator has passed known-truth validation.
  No interim tutorial files are created.

Forbidden bloat: no per-helper source modules for the DGP, no
per-estimator experiment modules, no placeholder result files, no
soft gates that merely warn on a recovery failure, and no swallowed
exceptions. A failed non-smoke estimator run writes a failure artifact
for diagnosis and then re-raises.

## Identification guardrails from Rust 1987

The DGP must be identified before it is used to judge an estimator.
For NFXP and the classical structural estimators, the Rust-style
guardrails are: conditional independence, transitions treated as
known or separately estimated, a low-dimensional full-rank utility
restriction, utility shock scale/location normalization, and enough
empirical action support to identify action-specific payoffs. The
canonical NFXP cell therefore uses action-dependent features, an
exit/absorbing-state normalization, and balanced action shares. If a
cell intentionally violates these conditions, it is labeled
`identification_collapsed`; it is not counted as an estimator bug.

## Cell layout

The DGP has one implementation and many declarative cells. The
currently checked-in canonical cells live in `DEFAULT_CELLS` inside
`experiments/known_truth.py`; future axis cells are added there, not
as new modules. The logical Tier 4 grid remains twelve cells, each a
single-axis flip from the spine config.

| Suffix | Spine override | Estimators run | Purpose |
| --- | --- | --- | --- |
| spine | none | all 12 focus estimators | Baseline; everyone should pass |
| state_only | action_dependent=False | classical 4 + MCE-IRL | MCE-IRL is the canonical failure |
| det_T | stochastic_transitions=False | all 12 focus estimators | Tests degeneracy handling |
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

R = 5 replications per cell. Replications use seed = 42 + r and
both the env and the simulation re-seed, so the Monte Carlo also
varies the frozen-network weights and transition matrices across
replications. R = 5 is enough to surface "always passes" vs "always
fails" vs "intermittent" behaviour for the alignment table; tighter
SE coverage estimates would need R = 20+ but coverage is not the
headline of this tier.

## Compute spec

Twelve axis cells with up to twelve estimators each. The shape-
shifter spine is small (S = 32, A = 3) so per-fit times are seconds,
not minutes. Counted at R = 5 with realistic per-fit times:

- 8 tabular axis cells (spine, state_only, det_T, stoch_r, finite,
  multi_action, product_state, large_S): ~5–12 estimators × 5 reps ×
  10–30s ≈ 2 CPU-hours total.
- high_gamma: 5 estimators × 5 reps × up to 60s ≈ 0.5 CPU-hours.
- neural_r, neural_phi, neural_r_phi: ~6 estimators × 5 reps ×
  1–3 min ≈ 1–2 GPU-hours total.

At RunPod community rates (CPU 0.40, GPU 1.20 per hour) and 8-way
parallelism, expected wall-clock under one hour and expected total
spend **5 to 10 USD**. The 50 USD ceiling on the dispatcher is a
safety cap, not the budget.

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
hand-written theoretical capability table (Table 2) and no empirical
Table 3 values. The empirical table is absent until the run artifacts
exist.
Once the plan fires, Table 3 is generated only from known-truth run
artifacts. The table writer should consume the `result.json` files
from `outputs/known_truth/` and write
`papers/econirl_package/tables/capability_empirical.tex` for the
paper to `\input{}`. No hand-entered table values.

## Artifact paths

- `outputs/known_truth/<cell_id>_<estimator>_<hash>/result.json`:
  one estimator run, including diagnostics, compatibility, summary,
  recovery metrics, hard gates, and exception text if the run failed.
- `outputs/known_truth/<cell_id>_<hash>/oracle.json`: exact truth,
  oracle policy/value objects, diagnostics, and Type A/B/C
  counterfactual oracles for one DGP cell.
- `papers/econirl_package/tables/capability_empirical.tex`: the
  final LaTeX snippet consumed by Section 3 Table 3 after aggregation.

## Run command

```
PYTHONPATH=src:. python -m experiments.known_truth \
    --estimator NFXP \
    --cell-id canonical_low_action \
    --output-dir outputs/known_truth \
    --show-progress \
    --verbose
```

For oracle artifacts without fitting an estimator:

```
PYTHONPATH=src:. python -m experiments.known_truth \
    --cell-id all \
    --output-dir outputs/known_truth/oracles
```

RunPod workers use the same module command inside the pod. The
dispatcher can parallelize cell/estimator pairs, but it should not
introduce separate per-estimator runners.
