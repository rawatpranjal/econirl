"""Rust (1987) replication package."""

from econirl.replication.rust1987.tables import (
    table_ii_descriptives,
    table_iv_transitions,
    table_v_structural,
)
from econirl.replication.rust1987.monte_carlo import (
    run_monte_carlo,
    summarize_monte_carlo,
)
from econirl.replication.rust1987.export import (
    table_to_latex,
    save_all_tables,
)

__all__ = [
    "table_ii_descriptives",
    "table_iv_transitions",
    "table_v_structural",
    "run_monte_carlo",
    "summarize_monte_carlo",
    "table_to_latex",
    "save_all_tables",
]
