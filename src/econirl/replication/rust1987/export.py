"""Export functions for publication-ready output."""

from __future__ import annotations

import pandas as pd
from typing import Optional
from pathlib import Path


def table_to_latex(
    df: pd.DataFrame,
    caption: str = "",
    label: str = "",
    float_format: str = "%.4f",
) -> str:
    """Convert DataFrame to LaTeX table.

    Args:
        df: Table to convert
        caption: Table caption
        label: LaTeX label for referencing
        float_format: Format string for floats

    Returns:
        LaTeX table string
    """
    latex = df.to_latex(
        float_format=float_format,
        caption=caption,
        label=label,
        escape=False,
    )
    return latex


def save_all_tables(
    output_dir: str = "output/tables",
    original: bool = True,
    groups: Optional[list[int]] = None,
):
    """Generate and save all replication tables.

    Args:
        output_dir: Directory for output files
        original: Use original Rust data
        groups: Which groups to include in Table V
    """
    from econirl.replication.rust1987.tables import (
        table_ii_descriptives,
        table_iv_transitions,
        table_v_structural,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Table II
    t2 = table_ii_descriptives(original=original)
    t2.to_csv(output_path / "table_ii.csv")
    with open(output_path / "table_ii.tex", "w") as f:
        f.write(table_to_latex(t2, caption="Table II: Descriptive Statistics", label="tab:table_ii"))

    # Table IV
    t4 = table_iv_transitions(original=original)
    t4.to_csv(output_path / "table_iv.csv")
    with open(output_path / "table_iv.tex", "w") as f:
        f.write(table_to_latex(t4, caption="Table IV: Transition Probabilities", label="tab:table_iv"))

    # Table V
    if groups is None:
        groups = [4]
    t5 = table_v_structural(original=original, groups=groups)
    t5.to_csv(output_path / "table_v.csv")
    with open(output_path / "table_v.tex", "w") as f:
        f.write(table_to_latex(t5, caption="Table V: Structural Estimates", label="tab:table_v"))

    print(f"Tables saved to {output_path}")
