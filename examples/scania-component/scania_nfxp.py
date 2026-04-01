#!/usr/bin/env python3
"""
SCANIA Component X Replacement -- NFXP Estimation
==================================================

Estimates the structural parameters of a component replacement model
using the SCANIA Component X dataset (or synthetic data if the real
data is not available). This is a Rust (1987) style model where
degradation replaces mileage as the state variable.

The pipeline:
    1. Load SCANIA data (synthetic fallback if no data_dir provided)
    2. Build environment and estimate transition probabilities
    3. Estimate structural parameters via NFXP
    4. Cross-validate with CCP-NPL
    5. Report results

Usage:
    python examples/scania-component/scania_nfxp.py

    # With real SCANIA data:
    python examples/scania-component/scania_nfxp.py --data-dir /path/to/scania/

Reference:
    SCANIA Component X dataset, IDA 2024 Industrial Challenge.
    Rust, J. (1987). Econometrica, 55(5), 999-1033.
"""

import argparse
import time

import numpy as np

from econirl.datasets import load_scania
from econirl.environments.scania import ScaniaComponentEnvironment
from econirl.estimation.nfxp import NFXPEstimator, estimate_transitions_from_panel
from econirl.estimation.ccp import CCPEstimator
from econirl.preferences.linear import LinearUtility


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def main():
    parser = argparse.ArgumentParser(
        description="NFXP estimation on SCANIA Component X data"
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
    args = parser.parse_args()

    print_header("SCANIA Component X Replacement -- NFXP Estimation")

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
    panel = load_scania(
        data_dir=args.data_dir,
        as_panel=True,
        max_vehicles=args.max_vehicles,
    )

    n_obs = len(df)
    n_vehicles = df["vehicle_id"].nunique()
    n_replace = df["replaced"].sum()
    replace_rate = df["replaced"].mean()
    periods_min = df.groupby("vehicle_id")["period"].count().min()
    periods_max = df.groupby("vehicle_id")["period"].count().max()

    print(f"  Observations:     {n_obs:,}")
    print(f"  Vehicles:         {n_vehicles}")
    print(f"  Replacements:     {n_replace} ({replace_rate:.2%})")
    print(f"  Periods/vehicle:  {periods_min}-{periods_max}")
    print(f"  Degradation bins: 0-{df['degradation_bin'].max()}")
    print(f"  Mean degradation: bin {df['degradation_bin'].mean():.1f}")

    # =========================================================================
    # Step 2: Set Up Environment and Estimate Transitions
    # =========================================================================
    print_section("Step 2: Estimate Transition Probabilities")

    num_states = 50
    env = ScaniaComponentEnvironment(
        num_degradation_bins=num_states,
        discount_factor=0.9999,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = estimate_transitions_from_panel(
        panel, num_states=num_states, max_increment=2
    )

    # Extract empirical transition probabilities
    trans_keep = np.array(transitions[0])
    p_stay = np.mean([trans_keep[s, s] for s in range(num_states - 2)])
    p_plus1 = np.mean([trans_keep[s, s + 1] for s in range(num_states - 2)])
    p_plus2 = np.mean([trans_keep[s, s + 2] for s in range(num_states - 2)])
    print(f"  Estimated P(+0): {p_stay:.4f}")
    print(f"  Estimated P(+1): {p_plus1:.4f}")
    print(f"  Estimated P(+2): {p_plus2:.4f}")

    # =========================================================================
    # Step 3: NFXP Estimation
    # =========================================================================
    print_section("Step 3: NFXP Estimation")

    nfxp = NFXPEstimator(
        optimizer="BHHH",
        inner_solver="policy",
        inner_tol=1e-12,
        inner_max_iter=200,
        compute_hessian=True,
        outer_tol=1e-3,
        verbose=True,
    )

    t0 = time.time()
    result_nfxp = nfxp.estimate(panel, utility, problem, transitions)
    t_nfxp = time.time() - t0

    op_cost = result_nfxp.parameters[0].item()
    rc = result_nfxp.parameters[1].item()

    print(f"\n  Time: {t_nfxp:.1f}s")
    print(result_nfxp.summary())

    # =========================================================================
    # Step 4: Cross-Validate with NPL
    # =========================================================================
    print_section("Step 4: Cross-Validate with CCP-NPL")

    npl = CCPEstimator(
        num_policy_iterations=20,
        compute_hessian=True,
        verbose=False,
    )

    t0 = time.time()
    result_npl = npl.estimate(panel, utility, problem, transitions)
    t_npl = time.time() - t0

    npl_op = result_npl.parameters[0].item()
    npl_rc = result_npl.parameters[1].item()

    print(f"  NPL:  operating_cost = {npl_op:.6f}, RC = {npl_rc:.4f}, LL = {result_npl.log_likelihood:.4f}  ({t_npl:.1f}s)")
    print(f"  NFXP: operating_cost = {op_cost:.6f}, RC = {rc:.4f}, LL = {result_nfxp.log_likelihood:.4f}  ({t_nfxp:.1f}s)")
    print(f"  Agreement: |dLL| = {abs(result_nfxp.log_likelihood - result_npl.log_likelihood):.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Estimation Summary")

    se_op = result_nfxp.standard_errors[0] if result_nfxp.standard_errors is not None else float("nan")
    se_rc = result_nfxp.standard_errors[1] if result_nfxp.standard_errors is not None else float("nan")

    print(f"""
  Model: SCANIA Component X Replacement (Rust-style DDC)
  Data:  {n_obs:,} observations, {n_vehicles} vehicles ({data_source})

  NFXP Estimates:
    operating_cost  = {op_cost:.6f}  (SE {se_op:.6f})
    replacement_cost = {rc:.4f}  (SE {se_rc:.4f})
    Log-likelihood  = {result_nfxp.log_likelihood:.4f}

  Cross-validation (NFXP vs NPL):
    |dLL| = {abs(result_nfxp.log_likelihood - result_npl.log_likelihood):.4f}

  Algorithm: Iskhakov et al. (2016) SA->NK polyalgorithm
    Inner solver: Policy iteration
    Outer solver: BHHH with analytical gradient
""")


if __name__ == "__main__":
    main()
