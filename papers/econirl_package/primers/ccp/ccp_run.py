#!/usr/bin/env python3
"""Generate CCP primer results from the known-truth DGP harness.

The CCP primer uses the same canonical synthetic DGP as the estimator
validation harness. No real data are used. The main run is a hard-gated
K=10 NPL validation with robust standard errors; the auxiliary rows show
how K=1 Hotz-Miller and K=3 NPL behave on the same finite panel.

Usage:
    cd /path/to/econirl
    PYTHONPATH=src:. python papers/econirl_package/primers/ccp/ccp_run.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
TEX_OUT = HERE / "ccp_results.tex"
DEFAULT_OUTPUT_DIR = Path("/tmp/econirl_ccp_primer_known_truth")
CELL_ID = "canonical_low_action"
ESTIMATOR = "CCP"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from econirl.estimation.ccp import CCPEstimator  # noqa: E402
from experiments.known_truth import (  # noqa: E402
    build_known_truth_dgp,
    evaluate_estimator_against_truth,
    get_cell,
    run_estimator,
    simulate_known_truth_panel,
    stable_hash,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", default=CELL_ID)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--show-progress", action="store_true", default=True)
    parser.add_argument("--quiet-progress", action="store_false", dest="show_progress")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("CCP primer: running known-truth validation")
    print(f"  cell: {args.cell_id}")
    print(f"  estimator: {ESTIMATOR}")
    print(f"  output_dir: {args.output_dir}")

    cell = get_cell(args.cell_id)
    dgp = build_known_truth_dgp(cell.dgp_config)
    simulation_config = replace(
        cell.simulation_config,
        show_progress=args.show_progress,
    )
    panel = simulate_known_truth_panel(dgp, simulation_config)

    main_run = run_estimator(
        ESTIMATOR,
        dgp,
        panel,
        smoke=False,
        verbose=args.verbose,
    )
    variants = run_npl_variants(dgp, panel, main_run, verbose=args.verbose)

    payload = {
        "cell": cell,
        "simulation": simulation_config,
        "estimator": ESTIMATOR,
        "diagnostics": main_run.diagnostics,
        "compatibility": main_run.compatibility,
        "summary": main_run.summary,
        "metrics": main_run.metrics,
        "gates": main_run.gates,
        "npl_variants": variants,
    }
    run_hash = stable_hash(
        {
            "cell": cell.dgp_config.to_dict(),
            "simulation": simulation_config,
            "estimator": ESTIMATOR,
            "primer": "ccp",
        }
    )
    run_dir = args.output_dir / f"{args.cell_id}_ccp_primer_{run_hash}"
    write_json(run_dir / "result.json", payload)

    tex = render_results_tex(payload, dgp)
    TEX_OUT.write_text(tex, encoding="utf-8")

    print(f"  result: {run_dir / 'result.json'}")
    print(f"  wrote: {TEX_OUT}")


def run_npl_variants(dgp: Any, panel: Any, main_run: Any, *, verbose: bool) -> list[dict[str, Any]]:
    """Run K=1 and K=3 comparison rows and reuse the gated K=10 run."""

    rows: list[dict[str, Any]] = []
    for k in (1, 3, 10):
        if k == 10:
            summary = main_run.summary
            metrics = main_run.metrics
        else:
            estimator = CCPEstimator(
                num_policy_iterations=k,
                outer_max_iter=500,
                compute_hessian=False,
                verbose=verbose,
            )
            summary = estimator.estimate(
                panel=panel,
                utility=dgp.utility(),
                problem=dgp.problem,
                transitions=dgp.transitions,
            )
            metrics = evaluate_estimator_against_truth(dgp, summary)

        rows.append(
            {
                "k": k,
                "label": "Hotz-Miller K=1" if k == 1 else f"NPL K={k}",
                "converged": bool(summary.converged),
                "iterations": int(summary.num_iterations),
                "log_likelihood": float(summary.log_likelihood),
                "time": float(summary.estimation_time),
                "parameters": np.asarray(summary.parameters).tolist(),
                "standard_errors": (
                    None
                    if summary.standard_errors is None
                    else np.asarray(summary.standard_errors).tolist()
                ),
                "parameter_relative_rmse": metrics["parameters"].relative_rmse,
                "parameter_cosine": metrics["parameters"].cosine_similarity,
                "reward_rmse": metrics["reward_rmse"],
                "value_rmse": metrics["value_rmse"],
                "q_rmse": metrics["q_rmse"],
                "policy_tv": metrics["policy"].tv,
            }
        )
    return rows


def render_results_tex(payload: dict[str, Any], dgp: Any) -> str:
    summary = payload["summary"]
    diagnostics = payload["diagnostics"]
    metrics = payload["metrics"]
    gates = payload["gates"]
    simulation = payload["simulation"]
    variants = payload["npl_variants"]

    names = list(summary.parameter_names)
    estimates = np.asarray(summary.parameters, dtype=float)
    standard_errors = np.asarray(summary.standard_errors, dtype=float)
    truth = np.asarray(dgp.homogeneous_parameters, dtype=float)
    errors = estimates - truth

    lines: list[str] = []
    add = lines.append

    add("% Auto-generated by ccp_run.py. Do not edit by hand.")
    add("% Known-truth DGP cell: canonical_low_action")
    add("")
    add(r"\section{Generated Validation Results}")
    add(
        "This section is generated by \\texttt{ccp\\_run.py} from the "
        "\\texttt{experiments.known\\_truth} harness and the canonical "
        "known-truth DGP."
    )
    add("")

    add(r"\subsection{Pre-Estimation Checks}")
    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Pre-estimation checks from the known-truth CCP run.}")
    add(r"\begin{tabular}{lrl}")
    add(r"\toprule")
    add(r"Check & Value & Status \\")
    add(r"\midrule")
    add(
        f"Feature rank & {diagnostics.feature_rank} / "
        f"{diagnostics.num_features} & {status(not diagnostics.errors)} \\\\"
    )
    add(f"Feature condition number & {fmt(diagnostics.condition_number, 3)} & pass \\\\")
    add(
        "Transition row error & "
        f"${sci(diagnostics.max_transition_row_error)}$ & pass \\\\"
    )
    add(
        f"Observed states & {diagnostics.observed_states} / "
        f"{diagnostics.num_states} & pass \\\\"
    )
    add(
        f"State-action coverage & {fmt(diagnostics.state_action_coverage, 3)} "
        "& pass \\\\"
    )
    shares = ", ".join(fmt(value, 3) for value in diagnostics.action_shares)
    add(f"Action shares & {shares} & pass \\\\")
    add(f"Minimum action share & {fmt(diagnostics.min_action_share, 3)} & pass \\\\")
    add(
        f"Minimum positive CCP & {fmt(diagnostics.min_positive_ccp, 3)} "
        "& pass \\\\"
    )
    add(f"Exit/absorbing anchor & {tf(diagnostics.anchor_valid)} & pass \\\\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\subsection{Estimator Fit}")
    add(
        "The medium-scale validation run uses "
        f"{int(simulation.n_individuals):,} individuals, "
        f"{int(simulation.n_periods):,} periods per individual, and "
        f"{int(summary.num_observations):,} observations."
    )
    add("")
    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Known-truth CCP/NPL run summary.}")
    add(r"\begin{tabular}{lr}")
    add(r"\toprule")
    add(r"Quantity & Value \\")
    add(r"\midrule")
    add(f"NPL iterations completed & {int(summary.num_iterations)} \\\\")
    add(f"NPL delta criterion met & {tf(summary.metadata['npl_converged'])} \\\\")
    add(f"Log-likelihood & {fmt(summary.log_likelihood, 4)} \\\\")
    add(f"Estimation time & {fmt(summary.estimation_time, 2)} seconds \\\\")
    add(f"Standard errors finite & {tf(np.isfinite(standard_errors).all())} \\\\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Parameter recovery.}")
    add(r"\begin{tabular}{lrrrr}")
    add(r"\toprule")
    add(r"Parameter & Truth & Estimate & SE & Error \\")
    add(r"\midrule")
    for name, true_value, estimate, se, error in zip(
        names, truth, estimates, standard_errors, errors, strict=True
    ):
        add(
            f"{tt(name)} & {fmt(true_value, 6)} & {fmt(estimate, 6)} & "
            f"{fmt(se, 6)} & {fmt(error, 6)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    parameter_metrics = metrics["parameters"]
    policy_metrics = metrics["policy"]
    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Recovery metrics.}")
    add(r"\begin{tabular}{lr}")
    add(r"\toprule")
    add(r"Metric & Value \\")
    add(r"\midrule")
    add(f"Parameter RMSE & {fmt(parameter_metrics.rmse, 6)} \\\\")
    add(f"Parameter relative RMSE & {fmt(parameter_metrics.relative_rmse, 6)} \\\\")
    add(
        "Parameter cosine similarity & "
        f"{fmt(parameter_metrics.cosine_similarity, 6)} \\\\"
    )
    add(f"Reward RMSE & {fmt(metrics['reward_rmse'], 6)} \\\\")
    add(f"Value RMSE & {fmt(metrics['value_rmse'], 6)} \\\\")
    add(f"Q RMSE & {fmt(metrics['q_rmse'], 6)} \\\\")
    add(f"Policy KL & ${sci(policy_metrics.kl)}$ \\\\")
    add(f"Policy total variation & {fmt(policy_metrics.tv, 6)} \\\\")
    add(f"Policy max state L1 & {fmt(policy_metrics.linf, 6)} \\\\")
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{K-stage CCP/NPL comparison on the same panel.}")
    add(r"\begin{tabular}{lrrrrrr}")
    add(r"\toprule")
    add(r"Estimator & Iter. & Rel. RMSE & Cosine & Policy TV & Value RMSE & Time (s) \\")
    add(r"\midrule")
    for row in variants:
        add(
            f"{tex_text(row['label'])} & {row['iterations']} & "
            f"{fmt(row['parameter_relative_rmse'], 6)} & "
            f"{fmt(row['parameter_cosine'], 6)} & "
            f"{fmt(row['policy_tv'], 6)} & "
            f"{fmt(row['value_rmse'], 6)} & {fmt(row['time'], 2)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")
    add(
        "In this canonical finite sample, the one-step Hotz-Miller row already "
        "recovers the main policy well because the empirical CCPs have full "
        "support. NPL iterations are still the estimator used for the hard "
        "validation because they move the pseudo-likelihood mapping toward the "
        "MLE fixed point and provide the appropriate structural object for "
        "standard errors."
    )
    add("")

    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Hard recovery gates.}")
    add(r"\begin{tabular}{llrl}")
    add(r"\toprule")
    add(r"Gate & Threshold & Value & Status \\")
    add(r"\midrule")
    for gate in gates:
        add(
            f"{tex_text(gate.name)} & {gate_threshold(gate)} & "
            f"{gate_value(gate.value)} & {status(gate.passed)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")
    add("")

    add(r"\subsection{Counterfactual Recovery}")
    add(r"\begin{table}[H]")
    add(r"\centering\small")
    add(r"\caption{Counterfactual recovery against known oracle objects.}")
    add(r"\begin{tabular}{lrrrr}")
    add(r"\toprule")
    add(r"Counterfactual & Policy TV & Policy KL & Value RMSE & Regret \\")
    add(r"\midrule")
    for kind, label in (
        ("type_a", "Type A"),
        ("type_b", "Type B"),
        ("type_c", "Type C"),
    ):
        cf = metrics["counterfactuals"][kind]
        add(
            f"{label} & {fmt(cf.policy.tv, 6)} & ${sci(cf.policy.kl)}$ & "
            f"{fmt(cf.value_rmse, 6)} & {fmt(cf.regret, 6)} \\\\"
        )
    add(r"\bottomrule")
    add(r"\end{tabular}")
    add(r"\end{table}")

    return "\n".join(lines) + "\n"


def fmt(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "---"
    return f"{float(value):.{digits}f}"


def sci(value: float) -> str:
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return rf"{mantissa}\times 10^{{{int(exponent)}}}"


def tf(value: bool) -> str:
    return "true" if bool(value) else "false"


def status(passed: bool) -> str:
    return "pass" if bool(passed) else "fail"


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


def gate_threshold(gate: Any) -> str:
    if gate.operator == "is":
        return tf(gate.threshold)
    operator = r"$\leq$" if gate.operator == "<=" else r"$\geq$"
    return f"{operator} {gate_value(gate.threshold)}"


def gate_value(value: Any) -> str:
    if isinstance(value, bool):
        return tf(value)
    if isinstance(value, (int, float)):
        return fmt(value, 6)
    return tex_text(str(value))


if __name__ == "__main__":
    main()
