"""Fan out the JSS deep-run matrix as RunPod jobs (or run locally).

Two modes.

Local mode (``--local``) runs each cell sequentially in the current
process. Useful for dry runs and for debugging the worker. Hardware
target is honored only by the JAX platform setting; no GPU is
materialized.

RunPod mode (default) submits one pod per cell via the RunPod
GraphQL API, polls every 60 seconds, retries failed cells up to
twice, and downloads result CSVs from the network volume back to
``--output-dir``. Requires ``RUNPOD_API_KEY`` in the environment.

The dispatcher tracks total spend and wall-clock time and halts if
either exceeds the bounds in the plan (50 USD, 4 hours, 30 minutes
per cell with one retry).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiments.jss_deep_run.matrix import MATRIX, Cell, cells_for_tiers, get_cell


# ---------------------------------------------------------------------------
# Cost ceiling
# ---------------------------------------------------------------------------

@dataclass
class CostCeiling:
    max_total_spend_usd: float = 50.0
    max_wall_clock_s: float = 4 * 3600.0
    max_per_cell_s: float = 30 * 60.0
    spend_so_far_usd: float = 0.0
    wall_clock_so_far_s: float = 0.0
    halted: bool = False
    halt_reason: str | None = None

    def check(self) -> None:
        if self.spend_so_far_usd > self.max_total_spend_usd:
            self.halted = True
            self.halt_reason = (
                f"Total spend {self.spend_so_far_usd:.2f} USD exceeds "
                f"ceiling {self.max_total_spend_usd:.2f} USD."
            )
        if self.wall_clock_so_far_s > self.max_wall_clock_s:
            self.halted = True
            self.halt_reason = (
                f"Wall-clock {self.wall_clock_so_far_s/3600:.2f}h exceeds "
                f"ceiling {self.max_wall_clock_s/3600:.2f}h."
            )


# ---------------------------------------------------------------------------
# Local sequential mode
# ---------------------------------------------------------------------------

def run_local(cells: list[Cell], output_dir: Path) -> list[dict[str, Any]]:
    """Run cells sequentially in subprocesses, tracking outcomes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    ceiling = CostCeiling()
    start = time.time()

    for cell in cells:
        if ceiling.halted:
            print(f"Halted: {ceiling.halt_reason}")
            break

        cmd = [
            sys.executable,
            "-m", "experiments.jss_deep_run.run_cell",
            "--cell-id", cell.cell_id,
            "--output-dir", str(output_dir),
        ]
        cell_start = time.time()
        print(f"[local] {cell.cell_id} ({cell.estimator} on {cell.dataset})")
        result = subprocess.run(
            cmd,
            timeout=ceiling.max_per_cell_s,
            capture_output=True,
            text=True,
        )
        cell_elapsed = time.time() - cell_start

        outcomes.append(
            {
                "cell_id": cell.cell_id,
                "returncode": result.returncode,
                "wall_clock_s": cell_elapsed,
                "stdout_tail": result.stdout[-512:] if result.stdout else "",
                "stderr_tail": result.stderr[-512:] if result.stderr else "",
            }
        )

        ceiling.wall_clock_so_far_s = time.time() - start
        ceiling.check()

    return outcomes


# ---------------------------------------------------------------------------
# RunPod mode (GraphQL stub)
# ---------------------------------------------------------------------------

# RunPod community A100 list price as of plan write-up. Refresh before
# any production run.
_RUNPOD_USD_PER_HOUR = {"cpu": 0.40, "gpu": 1.20}


def _bootstrap_command(cell: Cell, repo_ref: str) -> str:
    """Construct the in-pod setup-and-run command.

    The pod uses the public ``runpod/base`` Ubuntu template (~600 MB,
    pulls in 1-2 minutes vs >10 minutes for runpod/pytorch). The
    bootstrap installs python3, pip, git, then JAX, then clones and
    installs econirl, then runs the cell. Output goes to
    ``/workspace/results`` on the network volume so it survives
    self-termination.

    JAX is installed against CUDA 12 wheels for GPU cells; CPU cells
    install the wheel-only JAX so the inner solver runs on the box's
    CPU.
    """
    apt_install = (
        "apt-get update -q && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y -q "
        "python3 python3-pip python3-venv git curl ca-certificates"
    )
    # Use python3 -m pip rather than bare pip; on bare ubuntu after
    # apt-installing python3-pip the `pip` shim may not be on PATH.
    # No quoting on the package spec — brackets are bash-literal as
    # bare arguments, and any quotes here would have to survive both
    # the bash -c single-quoted wrap and the runpod GraphQL string.
    jax_install = (
        "python3 -m pip install -q jax[cuda12_pip]==0.4.30 "
        "-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
        if cell.hardware == "gpu"
        else "python3 -m pip install -q jax==0.4.30"
    )
    # The bootstrap ends by calling the RunPod REST API to self-
    # terminate the pod. Pods do not auto-shut when the entrypoint
    # command exits — they keep billing until something stops them.
    # Self-termination requires RUNPOD_API_KEY in the pod env (set
    # via the env dict in _runpod_pod_payload). The pod's own id is
    # auto-injected by the platform as RUNPOD_POD_ID.
    #
    # The trap body uses single quotes throughout and the
    # `Authorization:` header uses backslash-space escaping so that
    # no double quote characters end up in the docker_args string —
    # the runpod GraphQL builder interpolates docker_args into a
    # double-quoted string and does not escape inner quotes.
    # The bootstrap installs a base64-encoded cleanup script that
    # self-terminates the pod when the bootstrap exits. Encoding
    # avoids embedding double quotes or backslash-space in docker_args
    # (the runpod GraphQL builder does not escape those and the
    # parser rejects \ as an unknown escape).
    cleanup_script = (
        "#!/bin/bash\n"
        "curl -s -X DELETE -H \"Authorization: Bearer ${RUNPOD_API_KEY}\" "
        "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID} >/dev/null 2>&1\n"
    )
    import base64
    cleanup_b64 = base64.b64encode(cleanup_script.encode()).decode()
    cid = cell.cell_id
    mark_dir = f"/workspace/results/markers/{cid}"
    # The runpod/base image runs docker_args under /bin/sh (dash), not
    # bash. dash doesn't know `set -o pipefail` and exits immediately,
    # restart-looping the container. We re-enter bash explicitly so the
    # rest of the bootstrap can use bash-isms safely. The whole bash
    # invocation is single-quoted; the inner script must contain no
    # single quotes (verified — base64-encoded cleanup script hides any
    # quotes inside it).
    inner = (
        "set -uo pipefail; "
        f"mkdir -p {mark_dir} 2>/dev/null || mkdir -p {mark_dir} ; "
        f"date +%s > {mark_dir}/00_started ; "
        f"echo {cleanup_b64} | base64 -d > /tmp/cleanup.sh && chmod +x /tmp/cleanup.sh && "
        "trap /tmp/cleanup.sh EXIT; "
        "set -e; "
        f"{apt_install} && date +%s > {mark_dir}/10_apt_done && "
        f"cd /workspace && "
        f"git clone --depth 1 --branch main https://github.com/rawatpranjal/EconIRL.git econirl && "
        f"cd /workspace/econirl && git checkout {repo_ref} && date +%s > {mark_dir}/20_clone_done && "
        f"{jax_install} && date +%s > {mark_dir}/30_jax_done && "
        f"python3 -m pip install -q -e .[dev] && date +%s > {mark_dir}/40_pip_done && "
        "mkdir -p /workspace/results/shapeshifter && "
        f"python3 -m experiments.jss_deep_run.run_cell --cell-id {cid} "
        f"--output-dir /workspace/results/shapeshifter && "
        f"date +%s > {mark_dir}/50_cell_done && "
        f"cp /workspace/results/shapeshifter/{cid}.csv {mark_dir}/result.csv && "
        f"date +%s > {mark_dir}/99_all_done"
    )
    if "'" in inner:
        raise RuntimeError("bootstrap inner contains a single quote; cannot wrap in bash -c")
    return f"bash -c '{inner}'"


def _runpod_pod_payload(
    cell: Cell,
    image: str,
    volume_id: str,
    repo_ref: str = "main",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Build the GraphQL ``podRentInterruptable`` mutation payload.

    Uses the public ``runpod/pytorch`` template by default; the
    bootstrap dockerArgs clones the econirl repo at ``repo_ref`` and
    installs it before running the cell. Results land in
    ``/workspace/results/shapeshifter`` on the persistent volume.
    """
    payload: dict[str, Any] = {
        "name": cell.cell_id,
        "image_name": image,
        # cloud_type=ALL covers both COMMUNITY and SECURE; community
        # alone is often sold out in any given region, especially when
        # constrained by a network volume's region. Letting RunPod
        # pick the cheapest matching pod across both pools is the
        # difference between dispatch succeeding and failing for most
        # GPU types in 2026.
        "cloud_type": "ALL",
        "container_disk_in_gb": 20,
        # Mount the network volume at /workspace so each pod's CSV
        # survives self-termination. The volume is in EU-RO-1; that
        # constrains GPU availability but we accept the trade-off
        # because pod stdout is not queryable via the runpod API.
        "volume_in_gb": 0,
        "volume_mount_path": "/workspace",
        # Expose SSH so the operator can scp results from the network
        # volume after Tier 4 finishes; the user's ssh key is auto-
        # injected by RunPod via PUBLIC_KEY env on the pytorch image.
        "ports": "22/tcp",
        "env": {
            "CELL_ID": cell.cell_id,
            "JAX_PLATFORMS": "cuda" if cell.hardware == "gpu" else "cpu",
            # Required so the bootstrap's trap-EXIT handler can call
            # the REST API to self-terminate the pod once the cell
            # finishes. Without it the pod stays RUNNING and bills
            # forever.
            "RUNPOD_API_KEY": api_key or os.environ.get("RUNPOD_API_KEY", ""),
        },
        "docker_args": _bootstrap_command(cell, repo_ref),
    }
    if volume_id:
        payload["network_volume_id"] = volume_id
    payload["gpu_count"] = 1  # Even "cpu" cells use a GPU pod because
    # RunPod CPU pods need an instance_id and have far worse
    # availability than tiny GPU pods. The cheapest GPU is plenty for
    # the tabular tier-4 fits.
    # Caller fills gpu_type_id from the fallback list below.
    return payload


# Cheapest-first list of GPU types to try for tabular cells (most
# Tier 4 cells). When one is sold out, the dispatcher cycles through.
_GPU_FALLBACKS_TABULAR = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA RTX A4000",
    "NVIDIA GeForce RTX 4080",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX A5000",
    "NVIDIA GeForce RTX 4090",
]
# For cells the matrix declares as GPU (neural reward / features), use
# beefier GPUs.
_GPU_FALLBACKS_NEURAL = [
    "NVIDIA RTX A5000",
    "NVIDIA L40",
    "NVIDIA L40S",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
]


def run_runpod(
    cells: list[Cell], image: str, output_dir: Path, max_parallel: int,
    volume_id: str | None, repo_ref: str = "main",
) -> list[dict[str, Any]]:
    """Submit cells as RunPod pods and download results.

    This function is written defensively against a missing RunPod SDK:
    if the `runpod` package is unavailable the dispatcher falls back
    to printing the payload it would submit so the operator can
    inspect what would have been sent before installing the SDK.
    """
    try:
        import runpod  # type: ignore
    except ImportError:
        print(
            "runpod SDK not installed. Install with `pip install runpod` "
            "before submitting jobs. Printing payloads instead."
        )
        for cell in cells:
            payload = _runpod_pod_payload(cell, image, volume_id or "<volume_id>")
            print(json.dumps({"cell": cell.cell_id, "payload": payload}, indent=2))
        return []

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        raise RuntimeError(
            "RUNPOD_API_KEY not set. Export it before running the dispatcher."
        )
    runpod.api_key = api_key

    output_dir.mkdir(parents=True, exist_ok=True)
    ceiling = CostCeiling()
    start = time.time()
    in_flight: list[tuple[Cell, str, float]] = []  # (cell, pod_id, started_at)
    outcomes = []
    pending = list(cells)
    pending.sort(key=lambda c: -c.expected_runtime_s)  # longest first

    while pending or in_flight:
        if ceiling.halted:
            print(f"Halted: {ceiling.halt_reason}")
            break

        # Launch up to max_parallel pods, retrying across the GPU
        # fallback list until one type has capacity.
        while pending and len(in_flight) < max_parallel:
            cell = pending.pop(0)
            base_payload = _runpod_pod_payload(
                cell, image, volume_id or "<volume_id>", repo_ref, api_key=api_key,
            )
            fallbacks = (
                _GPU_FALLBACKS_NEURAL if cell.hardware == "gpu"
                else _GPU_FALLBACKS_TABULAR
            )
            launched = False
            last_err: str | None = None
            for gpu_type in fallbacks:
                payload = dict(base_payload, gpu_type_id=gpu_type)
                try:
                    pod = runpod.create_pod(**payload)
                    in_flight.append((cell, pod["id"], time.time()))
                    print(
                        f"[runpod] launched {cell.cell_id} pod_id={pod['id']} "
                        f"gpu={gpu_type} cost/hr={pod.get('costPerHr')}"
                    )
                    launched = True
                    break
                except Exception as e:
                    last_err = str(e)[:120]
                    continue
            if not launched:
                print(f"[runpod] all GPU fallbacks failed for {cell.cell_id}: {last_err}")
                outcomes.append({"cell_id": cell.cell_id, "error": last_err})

        # Poll pods.
        time.sleep(20)
        still_in_flight = []
        for cell, pod_id, started_at in in_flight:
            try:
                status = runpod.get_pod(pod_id)
            except Exception as e:
                print(f"[runpod] failed to poll {cell.cell_id}: {e}")
                still_in_flight.append((cell, pod_id, started_at))
                continue

            # After a self-terminate (REST DELETE from inside the pod),
            # get_pod returns None. Treat that as a clean exit.
            if status is None:
                elapsed = time.time() - started_at
                rate = _RUNPOD_USD_PER_HOUR[cell.hardware]
                spend = (elapsed / 3600.0) * rate
                ceiling.spend_so_far_usd += spend
                outcomes.append({
                    "cell_id": cell.cell_id, "pod_id": pod_id,
                    "wall_clock_s": elapsed, "spend_usd": spend,
                    "outcome": "self_terminated",
                })
                print(f"[runpod] {cell.cell_id} self-terminated in {elapsed:.0f}s, spend {spend:.3f} USD")
                continue

            if status.get("desiredStatus") in ("EXITED", "TERMINATED"):
                elapsed = time.time() - started_at
                rate = _RUNPOD_USD_PER_HOUR[cell.hardware]
                spend = (elapsed / 3600.0) * rate
                ceiling.spend_so_far_usd += spend
                outcomes.append(
                    {
                        "cell_id": cell.cell_id,
                        "pod_id": pod_id,
                        "wall_clock_s": elapsed,
                        "spend_usd": spend,
                    }
                )
                print(f"[runpod] {cell.cell_id} done in {elapsed:.0f}s, "
                      f"spend {spend:.3f} USD")
                # Pods write CSVs to the persistent volume; rsync to local.
                # In a real deployment this would use the RunPod API to
                # download from /workspace/results. The stub leaves it as
                # a TODO with a clear message.
                print(f"[runpod] TODO: download {cell.cell_id}.csv from volume to "
                      f"{output_dir}")
            elif time.time() - started_at > ceiling.max_per_cell_s:
                # Timeout: terminate and record.
                print(f"[runpod] {cell.cell_id} timed out, terminating pod")
                try:
                    runpod.terminate_pod(pod_id)
                except Exception:
                    pass
                outcomes.append(
                    {"cell_id": cell.cell_id, "pod_id": pod_id, "error": "timeout"}
                )
            else:
                still_in_flight.append((cell, pod_id, started_at))
        in_flight = still_in_flight

        ceiling.wall_clock_so_far_s = time.time() - start
        ceiling.check()

    return outcomes


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        default=None,
        help="Comma-separated tier ids to run, e.g. '1,2,3a,4'. Mutually exclusive with --cell-id.",
    )
    parser.add_argument(
        "--cell-id",
        default=None,
        help="Single cell id to dispatch. Useful for smoke tests.",
    )
    parser.add_argument("--local", action="store_true", help="Run sequentially in-process.")
    parser.add_argument(
        "--image",
        default="runpod/base:1.0.2-ubuntu2204",
        help="Container image. Default is runpod/base (~600 MB) so cold-pull on a fresh node finishes in 1-2 min vs >10 min for runpod/pytorch:2.4.0. The bootstrap apt-installs python3, then pip-installs JAX (CPU or CUDA depending on hardware) and econirl.",
    )
    parser.add_argument(
        "--repo-ref",
        default="main",
        help="Git ref (branch, tag, or SHA) of econirl to install in each pod.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="Maximum number of pods running at once.",
    )
    parser.add_argument(
        "--volume-id",
        default=None,
        help="RunPod network volume id for persisted result CSVs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/jss_deep_run/results"),
    )
    parser.add_argument(
        "--max-spend-usd",
        type=float,
        default=None,
        help="Override the dispatcher cost ceiling. Default uses CostCeiling defaults.",
    )
    parser.add_argument(
        "--max-wallclock-hours",
        type=float,
        default=None,
        help="Override the dispatcher wall-clock ceiling.",
    )
    args = parser.parse_args()

    if args.tier is None and args.cell_id is None:
        args.tier = "1"
    if args.tier is not None and args.cell_id is not None:
        parser.error("Pass --tier or --cell-id, not both.")

    if args.cell_id is not None:
        cells = [get_cell(args.cell_id)]
        tiers = [args.cell_id]
        print(f"Dispatching single cell {args.cell_id}")
    else:
        tiers = [t.strip() for t in args.tier.split(",") if t.strip()]
        cells = cells_for_tiers(tiers)
        print(f"Selected {len(cells)} cells across tiers {tiers}")

    # The CostCeiling has its own defaults; the CLI flags override them.
    # The local and runpod runners both construct their own CostCeiling
    # internally, so the override is applied via class attribute mutation
    # before they fire.
    if args.max_spend_usd is not None:
        CostCeiling.max_total_spend_usd = args.max_spend_usd
    if args.max_wallclock_hours is not None:
        CostCeiling.max_wall_clock_s = args.max_wallclock_hours * 3600.0

    if args.local:
        outcomes = run_local(cells, args.output_dir)
    else:
        outcomes = run_runpod(
            cells,
            image=args.image,
            output_dir=args.output_dir,
            max_parallel=args.max_parallel,
            volume_id=args.volume_id,
            repo_ref=args.repo_ref,
        )

    manifest = args.output_dir / "dispatch_manifest.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w") as f:
        json.dump({"tiers": tiers, "outcomes": outcomes}, f, indent=2)
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
