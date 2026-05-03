.. _rust-1987-replication:

Rust (1987) Replication
=======================

This tutorial demonstrates how to replicate the results from Rust's seminal
1987 paper on bus engine replacement using econirl.

.. contents:: Table of Contents
   :local:

Overview
--------

Rust (1987) introduced the Nested Fixed Point (NFXP) algorithm for estimating
dynamic discrete choice models. This replication package provides:

- Original Rust bus data (104 buses across 4 groups)
- Transition probability estimation (Table IV)
- Structural parameter estimation (Table V)
- Multiple estimators: NFXP, Hotz-Miller, NPL
- Monte Carlo validation
- LaTeX export for publication-ready tables

Quick Start
-----------

.. code-block:: python

    from econirl.datasets import load_rust_bus
    from econirl.replication.rust1987 import (
        table_ii_descriptives,
        table_iv_transitions,
        table_v_structural,
    )

    # Load original Rust data
    df = load_rust_bus(original=True)
    print(f"Loaded {len(df):,} observations")

    # Replicate Table II
    table_ii = table_ii_descriptives(df)
    print(table_ii)

    # Replicate Table V
    table_v = table_v_structural(df, groups=[4])
    print(table_v)

API Reference
-------------

.. automodule:: econirl.replication.rust1987
   :members:
   :undoc-members:
