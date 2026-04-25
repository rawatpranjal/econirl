# JSS deep run

The benchmark matrix for the `econirl` JSS paper. Reads from a single
matrix definition, fans out one job per cell on RunPod (or runs
locally), aggregates the per-cell CSVs into headline tables, and emits
LaTeX snippets for the paper.

## Files

- `matrix.py` — typed `Cell` records for every benchmark cell. The source of truth.
- `run_cell.py` — single-cell entry point. Runs one estimator on one dataset for R replications.
- `dispatch_runpod.py` — fans out cells as RunPod pods. Has a `--local` fallback for dry runs.
- `aggregate.py` — merges per-cell CSVs into the six headline CSVs plus a manifest.
- `report.py` — emits LaTeX snippets for the JSS paper from the headline CSVs.
- `Dockerfile` — container baked with the package and JAX-CUDA. Pushed once, pulled per pod.
- `results/` — per-cell CSVs and aggregated outputs. Gitignored.

## Tiers

| Tier | Headline | Dataset(s) | Estimators | R | Hardware |
| --- | --- | --- | --- | --- | --- |
| 1 | Hero panel | each of the five | the intended winner per dataset | 1 | mixed |
| 2 | A equivalence | rust-small | 12 estimators | 20 | CPU |
| 3a | C failure-and-recovery | rust-big | NFXP, CCP, GLADIUS, NNES, TD-CCP | 20 | GPU |
| 3b | D IRL scalability | ziebart-big | Deep MCE-IRL, tabular MCE-IRL, AIRL | 20 | GPU |
| 3c | E heterogeneity | lsw-synthetic | AIRL-Het, AIRL, MCE-IRL, BC | 20 | GPU |
| 3d | F transfer | rust-small (perturbed) | AIRL, MCE-IRL, IQ-Learn, f-IRL | 50 | CPU |
| 3e | G GPU speedup | five | neural estimators where applicable | 3 | CPU+GPU |

## Quickstart

Local dry run of one cell:

```bash
python -m experiments.jss_deep_run.run_cell --cell-id tier1_rust_small_nfxp --output-dir experiments/jss_deep_run/results/
```

Local sequential run of an entire tier:

```bash
python -m experiments.jss_deep_run.dispatch_runpod --tier 1 --local
```

Build the Docker image:

```bash
cd experiments/jss_deep_run
docker build -t econirl-deep-run:v1 .
docker push <your-registry>/econirl-deep-run:v1
```

Full RunPod run with parallelism:

```bash
export RUNPOD_API_KEY=<key>
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 2,3a,3b,3c,3d,3e \
    --image <your-registry>/econirl-deep-run:v1 \
    --max-parallel 8
```

Aggregate and report:

```bash
python -m experiments.jss_deep_run.aggregate
python -m experiments.jss_deep_run.report
```

## Cost ceiling

The dispatcher halts if any of the following bounds is exceeded:

- Total RunPod spend: 50 USD
- Wall-clock time: 4 hours
- Per-cell wall-clock time: 30 minutes (with one retry)

## Reproducibility

Every cell uses seeds `[seed_base, seed_base + 1, ..., seed_base + R - 1]`
with `seed_base = 42` by default. Each replication writes one row to
the cell's CSV. The aggregator's CSVs and the LaTeX snippets are
deterministic given the per-cell CSVs.

## Status

**Built.** Matrix definition, single-cell worker with per-replication
exception handling, local sequential dispatcher, RunPod GraphQL
dispatcher (stub when the SDK is missing), aggregator, reporter,
Dockerfile.

**To-do before the matrix actually fires.** The `_run_one_replication`
function in `run_cell.py` uses a generic `LinearUtility(parameter_names=...)`
constructor and a placeholder identity transition tensor. Each
estimator-dataset pair needs its own utility-function selector and
transition tensor wiring. The right shape for that wiring is a small
registry keyed on `(estimator, dataset)` that returns
`(utility, transitions, problem_kwargs)`. Until that registry exists,
every cell will record an exception and the headline tables will be
empty. The pipeline catches and records the exceptions cleanly, so
nothing else breaks.

**Tier 3d transfer perturbation** also needs implementation. The cell
metadata declares `perturb_transitions=True`, but the worker does not
yet act on that flag. Adding it amounts to a small helper that takes
the estimated transitions, multiplies each row by Beta(2, 2) noise,
and renormalizes.

**Tier 3b deep MCE-IRL** needs the `reward_type='neural'` extra_kwarg
to route to the neural variant of `MCEIRLEstimator`. The constructor
currently accepts `reward_type` and the worker passes it through, but
the actual neural-reward implementation in `econirl.estimation.mce_irl`
needs to be confirmed end-to-end on `ziebart-big`.

These are integration tasks, not architectural ones. The deep-run
scaffolding is finished and locally verified.
