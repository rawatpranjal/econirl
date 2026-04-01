#!/usr/bin/env python3
"""
SCANIA Component X Replacement -- Structural Estimation
========================================================

Estimates the structural parameters of a component replacement model
using real data from the SCANIA IDA 2024 Industrial Challenge. The
model treats component degradation as a Rust-style optimal stopping
problem with right censoring.

The pipeline:
    1. Load SCANIA data (real or synthetic fallback)
    2. PCA-based state construction from 105 sensor features
    3. Estimate structural parameters via NFXP
    4. Estimate via NNES (neural V-network) for comparison
    5. Report results and diagnostics

The real dataset has 23,550 vehicles, 1,122,452 observations, and
105 operational readout features. PCA reduces these to a single
degradation index (PC1 explains 97% of variance), which is then
discretized into 50 bins for tabular estimation.

Usage:
    # With real SCANIA data (download from Kaggle first):
    python examples/scania-component/scania_nfxp.py --data-dir data/scania/Dataset/

    # Synthetic fallback:
    python examples/scania-component/scania_nfxp.py

    # Quick test:
    python examples/scania-component/scania_nfxp.py --data-dir data/scania/Dataset/ --max-vehicles 1000

Reference:
    SCANIA Component X dataset, IDA 2024 Industrial Challenge.
    Rust, J. (1987). Econometrica, 55(5), 999-1033.
"""

import argparse
import time

import numpy as np

from econirl.datasets import load_scania


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def main():
    parser = argparse.ArgumentParser(
        description="Structural estimation on SCANIA Component X data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory with real SCANIA CSV files",
    )
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=None,
        help="Limit number of vehicles (for quick testing)",
    )
    parser.add_argument(
        "--discount",
        type=float,
        default=0.99,
        help="Discount factor (default 0.99 for irregular time steps)",
    )
    args = parser.parse_args()

    print_header("SCANIA Component X -- Structural Estimation")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print_section("Step 1: Load Data")

    data_source = "real" if args.data_dir else "synthetic"
    print(f"  Data source: {data_source}")

    df = load_scania(
        data_dir=args.data_dir,
        max_vehicles=args.max_vehicles,
    )

    n_obs = len(df)
    n_vehicles = df["vehicle_id"].nunique()
    n_replace = df["replaced"].sum()
    replace_rate = df["replaced"].mean()
    n_bins = df["degradation_bin"].nunique()
    obs_per_v = df.groupby("vehicle_id")["period"].count()

    print(f"  Observations:     {n_obs:,}")
    print(f"  Vehicles:         {n_vehicles:,}")
    print(f"  Replacements:     {n_replace} ({replace_rate:.2%} per period)")
    print(f"  Periods/vehicle:  median={obs_per_v.median():.0f}, "
          f"min={obs_per_v.min()}, max={obs_per_v.max()}")
    print(f"  Degradation bins: {n_bins}")
    print(f"  Mean degradation: bin {df['degradation_bin'].mean():.1f}")

    rep_deg = df.loc[df["replaced"] == 1, "degradation_bin"]
    keep_deg = df.loc[df["replaced"] == 0, "degradation_bin"]
    if len(rep_deg) > 0:
        print(f"  Degradation at replace: mean={rep_deg.mean():.1f}")
        print(f"  Degradation at keep:    mean={keep_deg.mean():.1f}")

    # =========================================================================
    # Step 2: NFXP Estimation
    # =========================================================================
    print_section("Step 2: NFXP Estimation")

    from econirl import NFXP

    t0 = time.time()
    nfxp = NFXP(n_states=n_bins, discount=args.discount).fit(
        df, state="degradation_bin", action="replaced", id="vehicle_id"
    )
    t_nfxp = time.time() - t0
    print(nfxp.summary())
    print(f"  Time: {t_nfxp:.1f}s")

    # =========================================================================
    # Step 3: NNES Estimation (Neural)
    # =========================================================================
    print_section("Step 3: NNES Estimation (Neural V-network)")

    from econirl import NNES

    t0 = time.time()
    nnes = NNES(
        n_states=n_bins, discount=args.discount, bellman="npl",
        v_epochs=300, n_outer_iterations=2,
    ).fit(
        df, state="degradation_bin", action="replaced", id="vehicle_id"
    )
    t_nnes = time.time() - t0
    print(nnes.summary())
    print(f"  Time: {t_nnes:.1f}s")

    # =========================================================================
    # Comparison
    # =========================================================================
    print_header("Results Comparison")

    print(f"\n  {'Estimator':<12} {'theta_c':>10} {'RC':>10} "
          f"{'SE(theta_c)':>12} {'SE(RC)':>10} {'LL':>12} {'Time':>8}")
    print("  " + "-" * 74)

    for name, model, t in [("NFXP", nfxp, t_nfxp), ("NNES", nnes, t_nnes)]:
        pv = list(model.params_.values())
        sv = list((model.se_ or {}).values())
        ll = model.log_likelihood_ if hasattr(model, "log_likelihood_") and model.log_likelihood_ is not None else float("nan")
        tc = pv[0] if len(pv) > 0 else float("nan")
        rc = pv[1] if len(pv) > 1 else float("nan")
        se_tc = sv[0] if len(sv) > 0 else float("nan")
        se_rc = sv[1] if len(sv) > 1 else float("nan")
        print(f"  {name:<12} {tc:>10.4f} {rc:>10.4f} "
              f"{se_tc:>12.4f} {se_rc:>10.4f} {ll:>12.2f} {t:>7.1f}s")

    print_header("Summary")
    print(f"""
  Model:  SCANIA Component X (single-spell optimal stopping)
  Data:   {n_obs:,} observations, {n_vehicles:,} vehicles ({data_source})
  State:  {n_bins} degradation bins (PCA-based, PC1 = 97% variance)
  Beta:   {args.discount}

  NFXP and NNES estimate the operating cost and replacement cost of
  Component X. The operating cost grows linearly with degradation.
  The replacement cost is the fixed utility penalty for replacing.
""")


if __name__ == "__main__":
    main()
