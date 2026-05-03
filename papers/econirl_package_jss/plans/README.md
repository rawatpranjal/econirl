# Per-dataset plans for the JSS deep run

Five plans, one per dataset in the curated registry. Each plan is a
spec sheet that names the dataset, the estimators in scope, the
metrics that get reported, the inference-validity classification, the
compute spec, the acceptance criteria, and what the paper says about
the dataset until the run produces real numbers.

## Why five separate plans

The reviewer evaluation in our brief was direct. The current draft
mixes three standards of evidence (software interface, econometric
inference, empirical benchmark), claims unsupported things ("twelve
production estimators"), uses misleading metrics (cosine similarity
of 1.0000 hides materially different parameter magnitudes), reports
inconsistent numbers between listings and tables, and ships
placeholder figures. The single biggest fix is to add rigorous
estimator validation per dataset so the package's breadth becomes
evidence rather than an API claim.

Five separate plans make that fix explicit. Each plan is the
authoritative artifact for one dataset. Each plan declares which
metrics and which estimators it owns. The paper Section 4 is
rewritten from the artifacts each plan produces. Anything that has
not yet run is withheld except for a plain pending-results note. No
prose about a dataset is committed before the corresponding plan has
fired.

## The five plans

| Plan | Dataset | Estimators | Headline claim |
| --- | --- | --- | --- |
| [plan_rust_small.md](plan_rust_small.md)       | rust-small     | All 12 | Equivalence on the canonical small panel; classical SEs achieve nominal coverage |
| [plan_rust_big.md](plan_rust_big.md)           | rust-big       | NFXP, CCP, GLADIUS, NNES, TD-CCP | Failure-and-recovery: classical breaks at 31-dim state, neural recovers |
| [plan_ziebart_small.md](plan_ziebart_small.md) | ziebart-small  | MCE-IRL, AIRL, IQ-Learn, BC | IRL canonical: reward recovered to high precision under known ground truth |
| [plan_ziebart_big.md](plan_ziebart_big.md)     | ziebart-big    | Deep MCE-IRL, tabular MCE-IRL, AIRL | IRL at scale: deep variant matches tabular at lower wall-clock |
| [plan_lsw_synthetic.md](plan_lsw_synthetic.md) | lsw-synthetic  | AIRL-Het, AIRL, MCE-IRL, BC | Latent types: heterogeneity matters when the DGP has it |

## Conventions across all five plans

**Metrics over cosine similarity.** Every plan reports relative
parameter error, policy KL divergence against the optimal policy,
replacement-probability or analogous policy RMSE, value loss, and
counterfactual error where applicable. Cosine similarity, when
included at all, is reported alongside the proper diagnostics rather
than instead of them.

**Inference validity classification.** Every plan classifies each
estimator's standard-error method as one of *valid asymptotic*,
*bootstrap-supported*, *heuristic*, or *unavailable*. The paper
Section 3 inherits this classification verbatim. We do not claim a
universal inference layer; we claim an inference layer that documents
its own validity per estimator.

**Single artifact per claim.** Every numerical claim in the paper
traces to one machine-written artifact produced by one script. No
hand-edited values. The paper compiles from the artifacts plus prose;
reruns of the scripts update the paper end to end without manual
intervention.

**Monte Carlo, not single fits.** Every claim that depends on the
quality of an estimator (not just on whether the API works) is backed
by a Monte Carlo experiment with R replications under known DGP. The
plan declares R per claim.

**Documented failure modes.** Every plan has a "Failure modes"
section that lists the conditions under which an estimator in scope
fails. Failures are reported, not hidden. The plan declares the
threshold beyond which a failure is itself the headline.

**Compact experiment infrastructure.** Known-truth validation lives
in `experiments/known_truth.py`, with fast checks in
`tests/test_known_truth.py`. Package source only receives estimator
fixes; the synthetic harness is not exported as public API and does
not create helper subtrees under `src/econirl/simulation/`. Final
RTD tutorial pages are written only after the estimator has passed
its known-truth run.

## Plan order and dependencies

The plans are independent at the data-generating level. Each one can
be executed without the others. The recommended execution order
optimizes total wall-clock and validates the cheap end first:

1. plan_rust_small.md (cheapest, also feeds the equivalence headline)
2. plan_ziebart_small.md (cheap IRL canonical)
3. plan_rust_big.md (GPU, but cells are individually cheap)
4. plan_lsw_synthetic.md (GPU, AIRL-Het EM is the long pole)
5. plan_ziebart_big.md (GPU, longest single cells)

Once all five have fired, paper Section 4 is rewritten in one pass
from the resulting artifacts.

## Until each plan fires

Section 4 of the paper currently has subsections that make claims
about each dataset. Until the corresponding plan has fired and
produced its artifact, those subsections carry only a pending-results
note: "Results for this dataset are forthcoming. The methodology,
hardware, and acceptance criteria are documented in
plans/plan_<dataset>.md." No numerical claims. No tables. No figures.

The paper does not lie about results that have not yet been
produced. Once a plan fires the prose for that subsection is written
from the artifact.
