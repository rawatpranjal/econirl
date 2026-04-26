# Cloud Verification Queue

Things that have been written or changed locally but cannot be
verified on the laptop per the no-laptop-execution rule. Each entry
is one command to dispatch. Run them in order; later entries depend
on earlier entries having succeeded.

The format is intentionally compact: one entry per line under each
heading, with the exact command to run.

## State at end of session

Phase A (cloud_test, shape-shifter env, Tier 4 cells, plan) — code
written and pushed (commits aaa31618, f6aee513, 49821b44).

Phase B (per-estimator alignment audits) — 14 audit docs written
(BC, NFXP, CCP, MPEC, MCE-IRL, MCE-IRL Deep, AIRL, AIRL-Het, f-IRL,
IQ-Learn, GLADIUS, NNES, SEES, TD-CCP). VALIDATION_LOG.md updated
(commit a6e8b266). Two paper-side gaps surfaced: Wulfmeier-Ondruska-
Posner 2015 and Lee-Sudhir-Wang 2026 are not in
papers/foundational/ and need to be added.

Phase D (capability tables) — Table 2 (theoretical) hand-written
into estimators.tex; Table 3 (empirical) is the auto-generated
placeholder; emit_capability_table.py is built and ready to refresh
Table 3 from Tier 4 CSV (commit 96b3c711).

Phase C (full Tier 4 dispatch) — **blocked on operator authorization
for spend**. Code is ready; nothing has been dispatched.

## Step 1 — Build and push the Docker image

```
docker build -f experiments/jss_deep_run/Dockerfile -t econirl-deep-run:v1 .
docker push econirl-deep-run:v1
```

The image now uses bash entrypoint and includes TeX Live so paper
builds work in-pod. Build context is the repo root.

## Step 2 — Sanity check on a CPU pod

```
python -m experiments.jss_deep_run.cloud_test \
    --shell "echo hello && python -c 'import econirl, jax; print(jax.__version__)'" \
    --max-spend-usd 1
```

Expected: prints "hello" and the JAX version. If this fails, the
image build did not succeed.

## Step 3 — Shape-shifter unit tests

```
python -m experiments.jss_deep_run.cloud_test \
    --pytest "tests/test_shapeshifter_env.py -v" \
    --max-spend-usd 2
```

Expected: 19 tests pass. Most likely failure modes if any: equinox
MLP API drift (in_size keyword vs positional) or the mixed-radix
coords loop being slow for state_dim==2 (4096-cell cap should hold).

## Step 4 — Single-cell smoke

```
python -m experiments.jss_deep_run.cloud_test \
    --shell "python -m experiments.jss_deep_run.run_cell --cell-id tier4_spine_nfxp --output-dir /workspace/results/shapeshifter" \
    --max-spend-usd 2
```

Expected: writes
``/workspace/results/shapeshifter/tier4_spine_nfxp.csv`` with one
row per replication and a summary JSON.

## Step 4b — Single-cell smoke via the dispatcher (alternative to Step 4)

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --cell-id tier4_spine_nfxp \
    --max-spend-usd 2 \
    --image econirl-deep-run:v1
```

The `--cell-id` flag is now supported on the dispatcher (added in
the same commit as Step 5's flags) so single-cell smoke tests do not
require cloud_test.py.

## Step 5 — Full Tier 4 dispatch (operator authorization required)

```
python -m experiments.jss_deep_run.dispatch_runpod \
    --tier 4 \
    --max-parallel 8 \
    --max-spend-usd 15 \
    --max-wallclock-hours 1 \
    --image econirl-deep-run:v1
```

Expected total cost **5 to 10 USD** at R = 5 reps per cell on the
small (S = 32, A = 3) shape-shifter (the rust-small runtime budgets
in the plan were over-estimates). The `--max-spend-usd 15` is a
safety cap. Tier 4 has 12 axis cells with up to 12 estimators each;
at R = 5 the variance is enough to surface "always passes" vs
"always fails" vs "intermittent" for the alignment table.

## Step 6 — Aggregate Tier 4 results

```
python -m experiments.jss_deep_run.cloud_test \
    --shell "python -m experiments.jss_deep_run.aggregate --tier 4" \
    --max-spend-usd 1
```

Expected: writes
``experiments/jss_deep_run/results/shapeshifter/headline.csv``.

## Step 7 — Regenerate the empirical capability table

```
python -m experiments.jss_deep_run.cloud_test \
    --shell "python experiments/jss_deep_run/emit_capability_table.py" \
    --max-spend-usd 1
```

Expected: rewrites
``papers/econirl_package/tables/capability_empirical.tex`` with
\\checkmark, $\\times$, na, and pending cells based on Tier 4
results. Pull the file back via ``runpodctl receive`` and commit.

## Step 8 — Build the paper PDF on a CPU pod

```
python -m experiments.jss_deep_run.cloud_test \
    --shell "cd papers/econirl_package && latexmk -pdf main.tex" \
    --max-spend-usd 2
```

Expected: builds papers/econirl_package/main.pdf with both Table 2
and Table 3. Pull the PDF back via the persistent volume.

## Step 9 — Per-estimator follow-up audits

For estimators where the Phase B audit recommended runtime
verification (MPEC at high beta, SEES with the default penalty
weight, IQ-Learn with sparse state coverage), dispatch the relevant
ss-* cell with R=20 and check whether the audit's predicted failure
mode actually surfaces. These are smoke tests; the data lives in
the Tier 4 CSV from Step 5.

## Tracked follow-ups (no command yet)

- Add Wulfmeier-Ondruska-Posner 2015 (Deep MaxEnt IRL) PDF to
  ``papers/foundational/`` and docling.
- Add Lee-Sudhir-Wang 2026 (AIRL-Het) PDF to ``papers/foundational/``
  and docling.
- Decide whether to change the package default
  ``num_policy_iterations`` for CCP from 1 to 10 (breaking API
  change, currently worked around at the matrix-cell level via
  ``extra_per_estimator``).
- Decide whether to change the MCE-IRL wrapper to raise an error or
  auto-construct features when ``feature_matrix=None`` and
  ``num_actions > 1``.
- Decide whether to change the SEES default
  ``bellman_penalty_weight`` to follow the paper's adaptive
  schedule or keep the constant default.

These are package-design decisions, not code-vs-paper alignment
issues, and are deliberately not applied in the alignment-audit
pass.
