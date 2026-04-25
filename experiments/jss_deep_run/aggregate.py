"""Aggregate per-cell CSVs into headline tables and a CI manifest.

CLI: ``python -m experiments.jss_deep_run.aggregate
--results-dir experiments/jss_deep_run/results``.

Reads every ``<cell_id>.csv`` under the results directory, joins them
with the cell metadata from ``matrix.py``, and writes one CSV per
headline plus a manifest CSV summarising convergence and runtime per
cell.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.jss_deep_run.matrix import MATRIX


_HEADLINE_FILES = {
    "hero":             "headline_hero.csv",
    "A_equivalence":    "headline_A_equivalence.csv",
    "C_failure_recovery": "headline_C_failure_recovery.csv",
    "D_irl_scalability": "headline_D_irl_scalability.csv",
    "E_heterogeneity":  "headline_E_heterogeneity.csv",
    "F_transfer":       "headline_F_transfer.csv",
    "G_gpu_speedup":    "headline_G_gpu_speedup.csv",
}


def _load_cell_csvs(results_dir: Path) -> pd.DataFrame:
    """Concatenate every per-cell CSV into one long DataFrame."""
    frames = []
    for cell in MATRIX:
        path = results_dir / f"{cell.cell_id}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _summarize_cell(df_cell: pd.DataFrame) -> dict[str, Any]:
    """Compute the per-cell summary row used in headline tables."""
    if df_cell.empty:
        return {}
    converged = df_cell["converged"].astype(bool)
    summary = {
        "estimator": df_cell["estimator"].iloc[0],
        "dataset": df_cell["dataset"].iloc[0],
        "hardware": df_cell["hardware"].iloc[0],
        "cell_id": df_cell["cell_id"].iloc[0],
        "n_replications": int(len(df_cell)),
        "convergence_rate": float(converged.mean()),
        "median_log_likelihood": float(np.nanmedian(df_cell["log_likelihood"])),
        "median_wall_clock_s": float(np.nanmedian(df_cell["wall_clock_s"])),
        "median_cosine_similarity": float(
            np.nanmedian(df_cell["cosine_similarity"])
        ),
        "iqr_wall_clock_s": float(
            np.nanpercentile(df_cell["wall_clock_s"], 75)
            - np.nanpercentile(df_cell["wall_clock_s"], 25)
        ),
        "n_exceptions": int(df_cell["exception"].notna().sum()),
    }
    # Bias and RMSE per parameter when truth is recorded as parameters_json
    # versus the implicit truth column. Skipped for now because the
    # truth vector is dataset-specific; the report layer adds these.
    return summary


def _emit_headline_csv(df_long: pd.DataFrame, results_dir: Path) -> None:
    """Group by headline tag and write one CSV per group."""
    for tag, fname in _HEADLINE_FILES.items():
        sub = df_long[df_long["headline_tag"] == tag]
        if sub.empty:
            continue
        rows = []
        for cell_id, group in sub.groupby("cell_id"):
            rows.append(_summarize_cell(group))
        df_out = pd.DataFrame(rows).sort_values("cell_id")
        out_path = results_dir / fname
        df_out.to_csv(out_path, index=False)
        print(f"Wrote {out_path} ({len(df_out)} cells)")


def _emit_manifest(df_long: pd.DataFrame, results_dir: Path) -> None:
    """One row per cell summarising whether the cell ran at all."""
    rows = []
    for cell in MATRIX:
        cell_df = df_long[df_long["cell_id"] == cell.cell_id]
        rows.append(
            {
                "cell_id": cell.cell_id,
                "estimator": cell.estimator,
                "dataset": cell.dataset,
                "hardware": cell.hardware,
                "n_replications_planned": cell.n_replications,
                "n_replications_observed": int(len(cell_df)),
                "convergence_rate": float(cell_df["converged"].mean())
                    if not cell_df.empty else None,
                "median_wall_clock_s": float(np.nanmedian(cell_df["wall_clock_s"]))
                    if not cell_df.empty else None,
                "n_exceptions": int(cell_df["exception"].notna().sum())
                    if not cell_df.empty else None,
                "status": "ok" if (
                    not cell_df.empty
                    and len(cell_df) == cell.n_replications
                    and int(cell_df["exception"].notna().sum()) == 0
                ) else "missing_or_partial",
            }
        )
    df = pd.DataFrame(rows)
    out_path = results_dir / "manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} cells)")


def _emit_readme_headlines(df_long: pd.DataFrame, results_dir: Path) -> None:
    """Hand-curated README bullets pulled from the headline CSVs."""
    lines: list[str] = ["# Headline numbers for the README"]
    if df_long.empty:
        lines.append("No results available yet.")
    else:
        a = df_long[df_long["headline_tag"] == "A_equivalence"]
        if not a.empty:
            cosines = a["cosine_similarity"].dropna()
            if len(cosines) > 0:
                lines.append(
                    f"- On rust-small, the median cosine similarity to the "
                    f"ground truth across the structural family is "
                    f"{cosines.median():.4f}."
                )
        c = df_long[df_long["headline_tag"] == "C_failure_recovery"]
        if not c.empty:
            nfxp = c[c["estimator"] == "NFXP"]
            gladius = c[c["estimator"] == "GLADIUS"]
            if not nfxp.empty and not gladius.empty:
                lines.append(
                    f"- On rust-big, NFXP converges on "
                    f"{nfxp['converged'].mean()*100:.0f} percent of "
                    f"replications. GLADIUS converges on "
                    f"{gladius['converged'].mean()*100:.0f} percent."
                )
        g = df_long[df_long["headline_tag"] == "G_gpu_speedup"]
        if not g.empty:
            cpu_med = g[g["hardware"] == "cpu"]["wall_clock_s"].median()
            gpu_med = g[g["hardware"] == "gpu"]["wall_clock_s"].median()
            if cpu_med and gpu_med and gpu_med > 0:
                lines.append(
                    f"- Across the neural estimators, the median GPU run is "
                    f"{cpu_med / gpu_med:.1f}x faster than the median CPU run."
                )
    out_path = results_dir / "readme_headlines.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/jss_deep_run/results"),
    )
    args = parser.parse_args()

    df_long = _load_cell_csvs(args.results_dir)
    if df_long.empty:
        print("No per-cell CSVs found. Run the dispatcher first.")
        return

    _emit_headline_csv(df_long, args.results_dir)
    _emit_manifest(df_long, args.results_dir)
    _emit_readme_headlines(df_long, args.results_dir)


if __name__ == "__main__":
    main()
