#!/usr/bin/env python3
"""Generate MCE-IRL primer validation results from the known-truth harness.

The MCE-IRL primer validates the low-level tabular estimator directly with
known transitions and known action-dependent reward features. The primary path
uses root feature matching, not the likelihood optimizer, because this is the
paper-faithful Ziebart-style MCE-IRL stationarity condition.

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python papers/econirl_package/primers/mce_irl/mce_irl_run.py --enforce-gates
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEX_OUT = HERE / "mce_irl_results.tex"
JSON_OUT = HERE / "mce_irl_results.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_mce_irl_primer_known_truth")
DEFAULT_CELL_IDS = ("canonical_low_action", "mce_low_high_reward")
PRIMARY_CELL_ID = "mce_low_high_reward"
ESTIMATOR = "MCE-IRL"
CELL_ROLES = {
    "canonical_low_action": "sanity",
    "mce_low_high_reward": "primary",
}
CELL_LABELS = {
    "canonical_low_action": "Canonical low-dimensional",
    "mce_low_high_reward": "Low-state high-reward-feature",
}

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.known_truth import (  # noqa: E402
    KnownTruthCell,
    KnownTruthDGPConfig,
    RecoveryGateFailure,
    SimulationConfig,
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    stable_hash,
    write_json,
)


MCE_HARD_CELL = KnownTruthCell(
    cell_id="mce_low_high_reward",
    dgp_config=KnownTruthDGPConfig(
        state_mode="low_dim",
        reward_mode="action_dependent",
        reward_dim="high",
        heterogeneity="none",
        num_regular_states=24,
        high_reward_features=8,
        transition_noise=0.02,
        seed=742,
    ),
    simulation_config=SimulationConfig(
        n_individuals=3_000,
        n_periods=100,
        seed=742,
    ),
    description=(
        "MCE-friendly hard case: low-dimensional states, high-dimensional "
        "action-dependent reward features, known transitions, and strong "
        "state-action coverage."
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help=(
            "Known-truth cell to run. May be repeated. Defaults to "
            "canonical_low_action and mce_low_high_reward."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)

    print("MCE-IRL primer: running known-truth validation")
    print(f"  cells: {', '.join(cell_ids)}")
    print(f"  estimator: {ESTIMATOR}")
    print(f"  output_dir: {args.output_dir}")

    records = [run_validation_cell(cell_id, args) for cell_id in cell_ids]

    write_json(JSON_OUT, compact_payload(records))
    TEX_OUT.write_text(render_results_tex(records), encoding="utf-8")

    failed_records: list[tuple[str, list[Any]]] = []
    total_gates = 0
    failed_gates = 0
    for record in records:
        payload = record["payload"]
        failed = [gate for gate in payload["gates"] if not gate.passed]
        total_gates += len(payload["gates"])
        failed_gates += len(failed)
        if failed:
            failed_records.append((payload["cell"].cell_id, failed))
        print(f"  result ({payload['cell'].cell_id}): {record['run_dir'] / 'result.json'}")
        print(
            f"  hard gates ({payload['cell'].cell_id}): "
            f"{len(payload['gates']) - len(failed)} pass, {len(failed)} fail"
        )

    print(f"  wrote: {JSON_OUT}")
    print(f"  wrote: {TEX_OUT}")
    print(f"  hard gates total: {total_gates - failed_gates} pass, {failed_gates} fail")

    if args.enforce_gates and failed_records:
        details = []
        for cell_id, failed in failed_records:
            details.append(
                f"{cell_id}: "
                + ", ".join(
                    f"{gate.name}={gate.value} {gate.operator} {gate.threshold}"
                    for gate in failed
                )
            )
        raise RecoveryGateFailure("; ".join(details))


def run_validation_cell(cell_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cell = _get_mce_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = replace(
        cell.simulation_config,
        show_progress=args.show_progress,
    )
    panel = simulate_panel(dgp, simulation_config)
    main_run = run_estimator(
        ESTIMATOR,
        dgp,
        panel,
        smoke=False,
        verbose=args.verbose,
        enforce_gates=False,
    )

    payload = {
        "cell": cell,
        "simulation": simulation_config,
        "estimator": ESTIMATOR,
        "diagnostics": main_run.diagnostics,
        "compatibility": main_run.compatibility,
        "summary": main_run.summary,
        "metrics": main_run.metrics,
        "gates": main_run.gates,
    }
    run_hash = stable_hash(
        {
            "cell": cell.dgp_config.to_dict(),
            "simulation": simulation_config,
            "estimator": ESTIMATOR,
            "primer": "mce_irl",
            "enforce_gates": bool(args.enforce_gates),
        }
    )
    run_dir = args.output_dir / f"{cell.cell_id}_mce_irl_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def _get_mce_cell(cell_id: str) -> KnownTruthCell:
    if cell_id == MCE_HARD_CELL.cell_id:
        return MCE_HARD_CELL
    return get_cell(cell_id)


def simulate_panel(dgp: Any, simulation_config: SimulationConfig) -> Any:
    from experiments.known_truth import simulate_known_truth_panel

    return simulate_known_truth_panel(dgp, simulation_config)


def render_results_tex(records: list[dict[str, Any]]) -> str:
    records_by_id = {record["payload"]["cell"].cell_id: record for record in records}
    low_record = records_by_id.get("canonical_low_action")
    hard_record = records_by_id.get(PRIMARY_CELL_ID)

    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by mce_irl_run.py. Do not edit by hand.")
    add("% Known-truth DGP cells: " + ", ".join(records_by_id))
    add("")
    add(r"\subsection{Generated Known-Truth Results}")
    add(
        "This section is generated by \\texttt{mce\\_irl\\_run.py} from the "
        "\\texttt{experiments.known\\_truth} harness. The estimator is the "
        "low-level \\texttt{MCEIRLEstimator} with "
        "\\texttt{optimizer=root} and "
        "\\texttt{compute\\_se=False}. Hard gates are enforced on feature "
        "matching, occupancy moments, normalized reward/value/Q recovery, "
        "policy distance, and Type A/B/C counterfactual regret."
    )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{MCE-IRL known-truth validation cells.}")
    add(r"\begin{tabular}{llrrrrr}")
    add(r"\toprule")
    add(r"Cell & Role & States & Actions & Reward features & Individuals & Periods \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        dgp = record["dgp"]
        cell_id = payload["cell"].cell_id
        sim = payload["simulation"]
        add(
            f"{tex_text(CELL_LABELS.get(cell_id, cell_id))} & "
            f"{tex_text(CELL_ROLES.get(cell_id, 'validation'))} & "
            f"{dgp.problem.num_states} & {dgp.problem.num_actions} & "
            f"{dgp.feature_matrix.shape[-1]} & {sim.n_individuals:,} & "
            f"{sim.n_periods} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{MCE-IRL recovery metrics.}")
    add(r"\begin{tabular}{lrr}")
    add(r"\toprule")
    add(r"Metric & Sanity cell & Primary cell \\")
    add(r"\midrule")
    metric_rows = (
        ("Feature residual", "feature_residual", 6),
        ("Occupancy moment residual", "occupancy_moment_residual", 6),
        ("Reward normalized RMSE", "reward_normalized_rmse", 6),
        ("Policy TV", "policy_tv", 6),
        ("Value normalized RMSE", "value_normalized_rmse", 6),
        ("Q normalized RMSE", "q_normalized_rmse", 6),
    )
    for label, key, digits in metric_rows:
        add(
            f"{tex_text(label)} & "
            f"{fmt(metric_value(low_record, key), digits)} & "
            f"{fmt(metric_value(hard_record, key), digits)} \\\\"
        )
    for kind, label in (
        ("type_a", "Type A regret"),
        ("type_b", "Type B regret"),
        ("type_c", "Type C regret"),
    ):
        add(
            f"{label} & "
            f"{fmt(counterfactual_value(low_record, kind), 6)} & "
            f"{fmt(counterfactual_value(hard_record, kind), 6)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Hard-gate audit by cell.}")
    add(r"\begin{tabular}{lrr}")
    add(r"\toprule")
    add(r"Cell & Passed gates & Failed gates \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        add(f"{tt(payload['cell'].cell_id)} & {passed} & {len(gates) - passed} \\\\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(
        "The JSON artifact \\texttt{mce\\_irl\\_results.json} contains the "
        "full parameter vectors, empirical and expected feature moments, "
        "occupancy residuals, raw RMSE values, normalized RMSE values, and all "
        "gate records. Raw parameter cosine is intentionally not a validation "
        "gate for MCE-IRL."
    )
    add("")

    return "\n".join(lines)


def metric_value(record: dict[str, Any] | None, key: str) -> float | None:
    if record is None:
        return None
    payload = record["payload"]
    metrics = payload["metrics"]
    if key == "feature_residual":
        return payload["summary"].metadata.get("feature_difference")
    if key == "occupancy_moment_residual":
        return payload["summary"].metadata.get("occupancy_moment_residual")
    if key == "policy_tv":
        return metrics["policy"].tv
    return metrics.get(key)


def counterfactual_value(record: dict[str, Any] | None, kind: str) -> float | None:
    if record is None:
        return None
    return record["payload"]["metrics"]["counterfactuals"][kind].regret


def compact_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "estimator": ESTIMATOR,
        "primary_cell_id": PRIMARY_CELL_ID,
        "results": [compact_cell_payload(record["payload"]) for record in records],
    }


def compact_cell_payload(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload["summary"]
    metadata = summary.metadata
    return {
        "cell_id": payload["cell"].cell_id,
        "estimator": payload["estimator"],
        "simulation": payload["simulation"],
        "compatibility": payload["compatibility"],
        "diagnostics": payload["diagnostics"],
        "summary": {
            "parameter_names": summary.parameter_names,
            "parameters": summary.parameters,
            "standard_errors": finite_list(summary.standard_errors),
            "log_likelihood": summary.log_likelihood,
            "converged": summary.converged,
            "num_iterations": summary.num_iterations,
            "num_observations": summary.num_observations,
            "estimation_time": summary.estimation_time,
            "convergence_message": summary.convergence_message,
            "metadata": {
                "optimizer": metadata.get("optimizer"),
                "feature_difference": metadata.get("feature_difference"),
                "occupancy_moment_residual": metadata.get(
                    "occupancy_moment_residual"
                ),
                "empirical_features": metadata.get("empirical_features"),
                "final_expected_features": metadata.get("final_expected_features"),
            },
        },
        "metrics": payload["metrics"],
        "gates": payload["gates"],
    }


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "---"
    number = float(value)
    if not math.isfinite(number):
        return "---"
    return f"{number:.{digits}f}"


def finite_list(values: Any) -> list[float | None]:
    array = np.asarray(values, dtype=float)
    return [float(value) if math.isfinite(float(value)) else None for value in array]


def tt(value: str) -> str:
    return r"\texttt{" + tex_text(value) + "}"


def tex_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
    )


if __name__ == "__main__":
    main()
