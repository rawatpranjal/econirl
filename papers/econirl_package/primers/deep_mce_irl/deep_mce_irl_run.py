#!/usr/bin/env python3
"""Generate Deep MCE-IRL known-truth validation artifacts."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEX_OUT = HERE / "deep_mce_irl_results.tex"
JSON_OUT = HERE / "deep_mce_irl_results.json"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_deep_mce_irl_primer_known_truth")
DEFAULT_CELL_IDS = (
    "canonical_low_state_only",
    "deep_mce_neural_reward",
    "deep_mce_neural_features",
    "deep_mce_neural_reward_features",
)
PRIMARY_CELL_ID = "deep_mce_neural_reward"
ESTIMATOR = "MCE-IRL Deep"
SIMULATION_OVERRIDES = {
    "canonical_low_state_only": {"n_individuals": 1_000, "n_periods": 80, "seed": 43},
}
CELL_ROLES = {
    "canonical_low_state_only": "sanity",
    "deep_mce_neural_reward": "primary",
    "deep_mce_neural_features": "finite-theta check",
    "deep_mce_neural_reward_features": "stress test",
}
CELL_LABELS = {
    "canonical_low_state_only": "Canonical projected reward",
    "deep_mce_neural_reward": "Neural reward",
    "deep_mce_neural_features": "Neural features",
    "deep_mce_neural_reward_features": "Neural reward + features",
}

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.known_truth import (  # noqa: E402
    RecoveryGateFailure,
    build_known_truth_dgp,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    stable_hash,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell-id",
        action="append",
        default=None,
        help=(
            "Known-truth cell to run. May be repeated. Defaults to "
            "the canonical sanity cell and the three Deep MCE Shapeshifter cells."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--enforce-gates", action="store_true")
    args = parser.parse_args()

    cell_ids = args.cell_id if args.cell_id is not None else list(DEFAULT_CELL_IDS)

    print("Deep MCE-IRL primer: running known-truth validation")
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
    cell = get_cell(cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = replace(
        cell.simulation_config,
        **SIMULATION_OVERRIDES.get(cell_id, {}),
        show_progress=args.show_progress,
    )
    panel = simulate_known_truth_panel(dgp, simulation_config)
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
            "primer": "deep_mce_irl",
            "enforce_gates": bool(args.enforce_gates),
        }
    )
    run_dir = args.output_dir / f"{cell.cell_id}_deep_mce_irl_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)
    return {"payload": payload, "dgp": dgp, "run_dir": run_dir}


def render_results_tex(records: list[dict[str, Any]]) -> str:
    records_by_id = {record["payload"]["cell"].cell_id: record for record in records}

    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by deep_mce_irl_run.py. Do not edit by hand.")
    add("% Known-truth DGP cells: " + ", ".join(records_by_id))
    add("")
    add(r"\subsection{Generated Known-Truth Results}")
    add(
        "This section is generated by \\texttt{deep\\_mce\\_irl\\_run.py} "
        "from the \\texttt{experiments.known\\_truth} harness. The primary "
        "validation target is an anchored raw nonlinear reward map learned by "
        "\\texttt{MCEIRLNeural} from demonstrations under known stochastic "
        "transitions and supplied state encodings. The Shapeshifter neural "
        "cells use action 0 as the zero-reward anchor. Neural network weights "
        "are not treated as structural parameters. When the Shapeshifter truth "
        "has a frozen neural reward, the gated artifact is the learned "
        "\\texttt{reward\\_matrix}; finite projected parameters are gated only "
        "when the projection basis is numerically identifiable."
    )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Deep MCE-IRL known-truth validation cells.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{lllrrrr}")
    add(r"\toprule")
    add(r"Cell & Role & Truth & States & Actions & Individuals & Periods \\")
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        dgp = record["dgp"]
        cell_id = payload["cell"].cell_id
        sim = payload["simulation"]
        target = payload["summary"].metadata.get("reward_validation_target", "linear")
        add(
            f"{tex_text(CELL_LABELS.get(cell_id, cell_id))} & "
            f"{tex_text(CELL_ROLES.get(cell_id, 'validation'))} & "
            f"{tex_text(str(target))} & {dgp.problem.num_states} & "
            f"{dgp.problem.num_actions} & {sim.n_individuals:,} & {sim.n_periods} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Deep MCE-IRL structural recovery metrics.}")
    add(r"\resizebox{\textwidth}{!}{%")
    add(r"\begin{tabular}{lrrrrrrrr}")
    add(r"\toprule")
    add(
        "Cell & Gates & Occ. resid. & Reward nRMSE & Policy TV & "
        "Value nRMSE & Q nRMSE & Param cosine & Type A/B/C regret \\\\"
    )
    add(r"\midrule")
    for record in records:
        payload = record["payload"]
        cell_id = payload["cell"].cell_id
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        regrets = "/".join(
            cell_metric(record, f"{kind}_regret", 4)
            for kind in ("type_a", "type_b", "type_c")
        )
        add(
            f"{tex_text(CELL_LABELS.get(cell_id, cell_id))} & "
            f"{passed}/{len(gates)} & "
            f"{cell_metric(record, 'occupancy_moment_residual', 4)} & "
            f"{cell_metric(record, 'reward_normalized_rmse', 4)} & "
            f"{cell_metric(record, 'policy_tv', 4)} & "
            f"{cell_metric(record, 'value_normalized_rmse', 4)} & "
            f"{cell_metric(record, 'q_normalized_rmse', 4)} & "
            f"{cell_metric(record, 'projected_parameter_cosine', 4)} & "
            f"{regrets} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"}")
    add(r"\end{table}")
    add("")

    for record in records:
        payload = record["payload"]
        cell_id = payload["cell"].cell_id
        gates = payload["gates"]
        passed = sum(gate.passed for gate in gates)
        failed = [gate for gate in gates if not gate.passed]
        add(
            f"Cell \\texttt{{{tex_text(cell_id)}}} passes {passed}/{len(gates)} "
            "hard gates."
        )
        if failed:
            add(
                " Failed gates: "
                + ", ".join(
                    f"\\texttt{{{tex_text(gate.name)}}}={fmt(gate.value, 4)}"
                    for gate in failed
                )
                + "."
            )
        if (
            payload["summary"].metadata.get("projected_parameter_identified") is False
            and payload["metrics"].get("parameters") is not None
        ):
            add(
                " Projected finite-theta statistics are diagnostic in this "
                "cell because the projection condition number is "
                f"{fmt(payload['summary'].metadata.get('projection_condition_number'), 3)}."
            )

    return "\n".join(lines) + "\n"


def cell_metric(record: dict[str, Any] | None, key: str, digits: int) -> str:
    if record is None:
        return "---"
    payload = record["payload"]
    summary = payload["summary"]
    metrics = payload["metrics"]
    if key == "projection_r2":
        return fmt(summary.metadata.get("projection_r2"), digits)
    if key == "policy_tv":
        return fmt(metrics["policy"].tv, digits)
    if key in {"reward_normalized_rmse", "value_normalized_rmse", "q_normalized_rmse"}:
        return fmt(metrics[key], digits)
    if key in {"projected_parameter_cosine", "projected_parameter_relative_rmse"}:
        parameter_metrics = metrics["parameters"]
        if parameter_metrics is None:
            return "---"
        attr = "cosine_similarity" if key.endswith("cosine") else "relative_rmse"
        return fmt(getattr(parameter_metrics, attr), digits)
    if key.endswith("_regret"):
        cf_key = key.removesuffix("_regret")
        return fmt(metrics["counterfactuals"][cf_key].regret, digits)
    return fmt(summary.metadata.get(key), digits)


def compact_payload(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "estimator": ESTIMATOR,
        "primary_cell": PRIMARY_CELL_ID,
        "records": [
            {
                "cell": record["payload"]["cell"],
                "simulation": record["payload"]["simulation"],
                "diagnostics": record["payload"]["diagnostics"],
                "compatibility": record["payload"]["compatibility"],
                "summary": record["payload"]["summary"],
                "metrics": record["payload"]["metrics"],
                "gates": record["payload"]["gates"],
                "run_dir": str(record["run_dir"]),
            }
            for record in records
        ],
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "---"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return tex_text(str(value))
    if math.isnan(x):
        return "---"
    if abs(x) >= 10 ** digits or (0 < abs(x) < 10 ** (-(digits - 1))):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def tex_text(value: str) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


if __name__ == "__main__":
    main()
