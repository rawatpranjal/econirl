# Cloud Verification Queue

Things that have been written or changed locally but cannot be
verified on the laptop per the no-laptop-execution rule. Each entry
should be dispatched to RunPod via `cloud_test.py` or
`dispatch_runpod.py` once the operator is available to authorize the
spend.

The format is intentionally compact: one entry per line under each
heading, with the exact command to run.

## Phase A0 — Image and dispatcher

- Rebuild and push the image once after the Dockerfile changes:
  `docker build -f experiments/jss_deep_run/Dockerfile -t econirl-deep-run:v1 . && docker push econirl-deep-run:v1`
- Sanity check the new bash entrypoint:
  `python -m experiments.jss_deep_run.cloud_test --shell "echo hello && python -c 'import econirl, jax; print(jax.__version__)'" --max-spend-usd 1`

## Phase A1 — Shape-shifter unit tests

- Run the env unit tests on a CPU pod:
  `python -m experiments.jss_deep_run.cloud_test --pytest "tests/test_shapeshifter_env.py -v" --max-spend-usd 2`
- If any test fails, the most likely suspects are:
  - `equinox` MLP API drift (in_size kw vs positional in newer versions)
  - jax.random.split key arity mismatch
  - The mixed-radix coords loop being slow for state_dim==2 with num_states==32 (4096 cap should hold)

## Phase A2 — Tier 4 cells (when added)

- Single-cell smoke after registering Tier 4:
  `python -m experiments.jss_deep_run.dispatch_runpod --cell-id ss-spine --max-spend-usd 2`

## Phase B — Per-estimator alignment audit follow-up

- For each estimator where the audit applied a code fix, dispatch
  the relevant ss-* cell to confirm parameter recovery:
  `python -m experiments.jss_deep_run.dispatch_runpod --cell-id <ss-cell> --max-spend-usd 2`
- The fixes need this verification before we move them out of
  "Fail (fix in PR)" into "Pass" in VALIDATION_LOG.md.

## Phase C — Full Tier 4 dispatch

- Authorize the operator for spend, then:
  `python -m experiments.jss_deep_run.dispatch_runpod --tier 4 --max-spend-usd 35 --max-wallclock-hours 2`
- Aggregate after dispatch:
  `python -m experiments.jss_deep_run.cloud_test --shell "python -m experiments.jss_deep_run.aggregate --tier 4" --max-spend-usd 1`

## Phase D — Paper builds

- Regenerate the empirical capability table once Tier 4 results are aggregated:
  `python -m experiments.jss_deep_run.cloud_test --shell "python experiments/jss_deep_run/emit_capability_table.py" --max-spend-usd 1`
- Build the JSS paper PDF on a CPU pod with TeX Live:
  `python -m experiments.jss_deep_run.cloud_test --shell "cd papers/econirl_package && latexmk -pdf main.tex" --max-spend-usd 2`

## Notes

The `dispatch_runpod.py` CLI does not yet support `--cell-id` or
`--max-spend-usd`. Those flags are listed in the plan and are added
opportunistically while writing the audits, but if missing, the
operator can either edit the in-script ceiling temporarily or call
`cloud_test.py --module experiments.jss_deep_run.run_cell ...` which
already supports `--max-spend-usd`.
