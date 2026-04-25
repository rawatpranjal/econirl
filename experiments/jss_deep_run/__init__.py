"""JSS deep run: experiment matrix, worker, dispatcher, aggregator, reporter.

The package fans out a curated benchmark matrix across the five
canonical datasets (rust-small, rust-big, ziebart-small, ziebart-big,
lsw-synthetic) and the twelve production estimators, runs each cell
on RunPod or locally, aggregates the per-cell CSVs into headline
tables, and emits LaTeX snippets for the JSS paper.

See README.md for the operator guide.
"""
