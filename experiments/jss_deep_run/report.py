"""Emit LaTeX snippets for the JSS paper from the headline CSVs.

CLI: ``python -m experiments.jss_deep_run.report
--results-dir experiments/jss_deep_run/results
--figures-dir papers/econirl_package_jss/figures``.

Reads the headline CSVs produced by ``aggregate.py`` and writes one
``\\input{}``-able LaTeX table per JSS section that depends on the
deep run. The paper picks the snippets up unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def _table_a_equivalence(df: pd.DataFrame) -> str:
    """Section 4.4 cross-estimator equivalence table on rust-small."""
    df = df.sort_values("median_log_likelihood", ascending=False)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['estimator']:<10s} & "
            f"{r['median_log_likelihood']:.2f} & "
            f"{r['median_wall_clock_s']:.1f} & "
            f"{r['median_cosine_similarity']:.4f} & "
            f"{r['convergence_rate']*100:.0f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrrr}\n"
        "\\toprule\n"
        "Estimator & log-lik. & time (s) & cos.\\ sim. & conv. (\\%) \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _table_c_failure_recovery(df: pd.DataFrame) -> str:
    """Section 4.3a failure-and-recovery table on rust-big."""
    df = df.sort_values("convergence_rate", ascending=True)
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"{r['estimator']:<10s} & "
            f"{r['convergence_rate']*100:.0f} & "
            f"{r['median_wall_clock_s']:.1f} & "
            f"{r['median_cosine_similarity']:.4f} \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Estimator & conv. (\\%) & time (s) & cos.\\ sim. \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _table_g_gpu_speedup(df: pd.DataFrame) -> str:
    """Section 4.6 GPU-versus-CPU speedup table."""
    pivot = df.pivot_table(
        index="estimator",
        columns="hardware",
        values="median_wall_clock_s",
        aggfunc="median",
    )
    if "cpu" not in pivot.columns or "gpu" not in pivot.columns:
        return "% no GPU/CPU pairs in the deep-run results yet"
    pivot["speedup"] = pivot["cpu"] / pivot["gpu"]
    rows = []
    for est, r in pivot.iterrows():
        rows.append(
            f"{est:<10s} & {r['cpu']:.1f} & {r['gpu']:.1f} & {r['speedup']:.1f}x \\\\"
        )
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lrrr}\n"
        "\\toprule\n"
        "Estimator & CPU (s) & GPU (s) & speedup \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


_TABLE_BUILDERS = {
    "headline_A_equivalence.csv":      ("table_a_equivalence.tex",      _table_a_equivalence),
    "headline_C_failure_recovery.csv": ("table_c_failure_recovery.tex", _table_c_failure_recovery),
    "headline_G_gpu_speedup.csv":      ("table_g_gpu_speedup.tex",      _table_g_gpu_speedup),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/jss_deep_run/results"),
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=Path("papers/econirl_package_jss/figures"),
    )
    args = parser.parse_args()

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    for csv_name, (tex_name, builder) in _TABLE_BUILDERS.items():
        csv_path = args.results_dir / csv_name
        if not csv_path.exists():
            print(f"Skipping {csv_name}: not found in {args.results_dir}")
            continue
        df = pd.read_csv(csv_path)
        if df.empty:
            print(f"Skipping {csv_name}: empty")
            continue
        snippet = builder(df)
        out_path = args.figures_dir / tex_name
        out_path.write_text(snippet)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
