"""Single-cell worker for the JSS deep run.

CLI: ``python -m experiments.jss_deep_run.run_cell --cell-id <id>
--output-dir <dir>``. The script runs one cell of the matrix end to
end, captures one row per Monte Carlo replication into a CSV, and
writes a single-row summary JSON. Per-replication exceptions are
caught and recorded so a single failure does not poison the cell.

The same script runs locally on a CPU laptop or inside a RunPod
container with a GPU. The hardware target comes from the cell
definition. The worker sets the JAX platform accordingly via the
`JAX_PLATFORMS` environment variable before importing JAX.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from experiments.jss_deep_run.matrix import Cell, get_cell


# ---------------------------------------------------------------------------
# Estimator and dataset registries
# ---------------------------------------------------------------------------

def _estimator_factory(name: str) -> Callable[..., Any]:
    """Return the estimator class for the matrix-level name."""
    from econirl import estimation

    table = {
        "NFXP":     estimation.NFXP,
        "MPEC":     estimation.MPEC,
        "CCP":      estimation.CCP,
        "MCE-IRL":  estimation.MCEIRL,
        "TD-CCP":   estimation.TDCCP,
        "NNES":     estimation.NNES,
        "SEES":     estimation.SEES,
        "IQ-Learn": estimation.IQLearn,
        "GLADIUS":  estimation.GLADIUS,
        "AIRL":     estimation.AIRL,
        "AIRL-Het": estimation.AIRLHet,
        "f-IRL":    estimation.FIRL,
        "BC":       estimation.BC,
    }
    if name not in table:
        raise KeyError(f"Unknown estimator name {name!r}")
    return table[name]


def _dataset_loader(name: str) -> Callable[..., Any]:
    """Return the loader function for the matrix-level dataset name."""
    from econirl import datasets

    table = {
        "rust-small":    datasets.load_rust_small,
        "rust-big":      datasets.load_rust_big,
        "ziebart-small": datasets.load_ziebart_small,
        "ziebart-big":   datasets.load_ziebart_big,
        "lsw-synthetic": datasets.load_lsw_synthetic,
    }
    if name not in table:
        raise KeyError(f"Unknown dataset name {name!r}")
    return table[name]


# ---------------------------------------------------------------------------
# Per-replication record
# ---------------------------------------------------------------------------

_RESULT_COLUMNS = [
    "cell_id",
    "tier",
    "headline_tag",
    "estimator",
    "dataset",
    "hardware",
    "replication",
    "seed",
    "converged",
    "log_likelihood",
    "wall_clock_s",
    "n_iterations",
    "n_parameters",
    "n_observations",
    "n_individuals",
    "parameters_json",
    "standard_errors_json",
    "cosine_similarity",
    "exception",
]


def _vector_to_json(v: Any) -> str:
    """Serialize a parameter or SE vector to a compact JSON string."""
    if v is None:
        return "null"
    arr = np.asarray(v).astype(float)
    return json.dumps(arr.tolist())


def _cosine_similarity(estimated: Any, truth: Any) -> float | None:
    """Cosine similarity between estimated and true parameter vectors.

    Returns None if either vector is missing or zero-norm.
    """
    if estimated is None or truth is None:
        return None
    a = np.asarray(estimated).astype(float).flatten()
    b = np.asarray(truth).astype(float).flatten()
    if a.shape != b.shape:
        return None
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# Main worker
# ---------------------------------------------------------------------------

def _run_one_replication(
    cell: Cell,
    seed: int,
    replication_index: int,
) -> dict[str, Any]:
    """Run one fit and return a flat result dict for the CSV row."""
    from econirl.core.types import DDCProblem
    from econirl.preferences.linear import LinearUtility

    record = {col: None for col in _RESULT_COLUMNS}
    record.update(
        {
            "cell_id": cell.cell_id,
            "tier": cell.cell_id.split("_", 1)[0],
            "headline_tag": cell.headline_tag,
            "estimator": cell.estimator,
            "dataset": cell.dataset,
            "hardware": cell.hardware,
            "replication": replication_index,
            "seed": seed,
            "converged": False,
        }
    )

    try:
        loader = _dataset_loader(cell.dataset)
        loader_kwargs = {"seed": seed, "as_panel": True}
        # The loader signatures accept seed but not all keyword names
        # match. Best effort: drop unsupported kwargs.
        try:
            panel = loader(**loader_kwargs)
        except TypeError:
            panel = loader(as_panel=True)

        record["n_observations"] = int(panel.num_observations)
        record["n_individuals"] = int(panel.num_individuals)

        estimator_cls = _estimator_factory(cell.estimator)
        # Filter cell.extra_kwargs into constructor and estimate-time
        # buckets. Anything the constructor does not accept is passed
        # through to estimate. The cell's extra_kwargs are documented
        # inline in matrix.py.
        constructor_kwargs = {
            k: v for k, v in cell.extra_kwargs.items()
            if k in {
                "num_segments", "exit_action", "absorbing_state",
                "reward_type",
            }
        }
        try:
            estimator = estimator_cls(**constructor_kwargs)
        except TypeError:
            # Estimator does not accept these kwargs.
            estimator = estimator_cls()

        problem = DDCProblem(
            num_states=int(np.max(np.asarray(panel.get_all_states())) + 1),
            num_actions=int(np.max(np.asarray(panel.get_all_actions())) + 1),
            discount_factor=0.95,
        )
        utility = LinearUtility(parameter_names=["theta_0", "theta_1"])

        start = time.time()
        # Estimators expect (num_actions, num_states, num_states).
        n_a, n_s = problem.num_actions, problem.num_states
        transitions = np.eye(n_s)[None, :, :].repeat(n_a, axis=0)
        result = estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )
        elapsed = time.time() - start

        record["wall_clock_s"] = elapsed
        record["converged"] = bool(getattr(result, "converged", True))
        record["log_likelihood"] = float(
            getattr(result, "log_likelihood", float("nan"))
        )
        record["n_iterations"] = int(getattr(result, "num_iterations", -1))
        record["n_parameters"] = int(np.asarray(result.parameters).size)
        record["parameters_json"] = _vector_to_json(result.parameters)
        record["standard_errors_json"] = _vector_to_json(
            getattr(result, "standard_errors", None)
        )
        # Truth vector for cosine similarity. Only available for
        # rust-small with the synthetic ground-truth parameters.
        truth = None
        if cell.dataset == "rust-small":
            truth = np.array([0.001, 3.0])
        record["cosine_similarity"] = _cosine_similarity(
            result.parameters, truth
        )

    except Exception:
        record["exception"] = traceback.format_exc()
    return record


def run_cell(cell: Cell, output_dir: Path, seed_base: int = 42) -> Path:
    """Run all R replications of a cell and write the per-cell artifacts.

    Returns the path to the per-cell CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{cell.cell_id}.csv"
    summary_path = output_dir / f"{cell.cell_id}_summary.json"

    rows = []
    for r in range(cell.n_replications):
        seed = seed_base + r
        row = _run_one_replication(cell, seed=seed, replication_index=r)
        rows.append(row)

    df = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
    df.to_csv(csv_path, index=False)

    convergence_rate = float(df["converged"].mean()) if len(df) else 0.0
    median_runtime = float(df["wall_clock_s"].median()) if len(df) else 0.0
    summary = {
        "cell_id": cell.cell_id,
        "estimator": cell.estimator,
        "dataset": cell.dataset,
        "hardware": cell.hardware,
        "headline_tag": cell.headline_tag,
        "n_replications": cell.n_replications,
        "convergence_rate": convergence_rate,
        "median_wall_clock_s": median_runtime,
        "n_exceptions": int(df["exception"].notna().sum()),
        "csv_path": str(csv_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path


def _set_jax_platform(hardware: str) -> None:
    """Pin JAX to CPU or GPU before any econirl import touches JAX."""
    if hardware == "cpu":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    elif hardware == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument(
        "--output-dir",
        default="experiments/jss_deep_run/results",
        type=Path,
    )
    parser.add_argument("--seed-base", type=int, default=42)
    args = parser.parse_args()

    cell = get_cell(args.cell_id)
    _set_jax_platform(cell.hardware)

    print(f"Running {cell.cell_id} ({cell.estimator} on {cell.dataset}, "
          f"R={cell.n_replications}, hardware={cell.hardware})")
    csv_path = run_cell(cell, output_dir=args.output_dir, seed_base=args.seed_base)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
