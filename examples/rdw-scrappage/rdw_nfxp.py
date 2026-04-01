#!/usr/bin/env python3
"""
RDW Vehicle Scrappage -- NFXP Estimation
=========================================

Estimates the structural parameters of a vehicle scrappage model
using Dutch RDW inspection data (or synthetic data if the real
data is not available). This is a Rust (1987) style optimal stopping
model where age and defect severity replace mileage as the state
variables.

The pipeline:
    1. Load RDW data (synthetic fallback if no data_dir provided)
    2. Build environment and estimate transition probabilities
    3. Estimate structural parameters via NFXP
    4. Cross-validate with CCP-NPL
    5. Report results

Usage:
    python examples/rdw-scrappage/rdw_nfxp.py

    # With real RDW data:
    python examples/rdw-scrappage/rdw_nfxp.py --data-dir /path/to/rdw/

Reference:
    RDW Open Data: https://opendata.rdw.nl
    Rust, J. (1987). Econometrica, 55(5), 999-1033.
"""

import argparse
import time

import numpy as np

from econirl.datasets import load_rdw_scrappage
from econirl.environments.rdw_scrappage import RDWScrapageEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.preferences.linear import LinearUtility


def print_header(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def estimate_rdw_transitions(panel, num_states: int) -> np.ndarray:
    """Estimate transition matrices from panel data.

    Unlike the standard estimate_transitions_from_panel which assumes 1D
    state increments, this counts (s, a=0, s') frequency pairs in the
    full flattened state space and normalizes. For the scrap action,
    the theoretical reset to state 0 is used.

    Args:
        panel: Panel with observed states and actions
        num_states: Total number of states in the flattened space

    Returns:
        Transition matrix of shape (2, num_states, num_states)
    """
    # Count keep-action transitions from data
    keep_counts = np.zeros((num_states, num_states), dtype=np.float64)

    for traj in panel.trajectories:
        states = np.array(traj.states)
        actions = np.array(traj.actions)
        for t in range(len(states) - 1):
            if int(actions[t]) == 0:  # keep action
                s = int(states[t])
                s_next = int(states[t + 1])
                if 0 <= s < num_states and 0 <= s_next < num_states:
                    keep_counts[s, s_next] += 1

    # Normalize rows (add small epsilon to avoid division by zero)
    row_sums = keep_counts.sum(axis=1, keepdims=True)
    # For rows with no observations, use uniform transitions
    empty_rows = (row_sums.flatten() == 0)
    keep_trans = np.where(
        row_sums > 0,
        keep_counts / np.maximum(row_sums, 1),
        np.ones((num_states, num_states)) / num_states,
    )

    if empty_rows.any():
        n_empty = empty_rows.sum()
        print(f"  Warning: {n_empty} states with no keep-action observations, "
              f"using uniform transitions")

    # Build full transition tensor
    transitions = np.zeros((2, num_states, num_states), dtype=np.float64)
    transitions[0] = keep_trans

    # Scrap action: reset to state 0 (new car, no defects)
    transitions[1, :, 0] = 1.0

    return transitions


def main():
    parser = argparse.ArgumentParser(
        description="NFXP estimation on RDW vehicle scrappage data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory with real RDW CSV files",
    )
    parser.add_argument(
        "--max-vehicles",
        type=int,
        default=None,
        help="Limit number of vehicles (for quick testing)",
    )
    args = parser.parse_args()

    print_header("RDW Vehicle Scrappage -- NFXP Estimation")

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print_section("Step 1: Load Data")

    data_source = "real" if args.data_dir else "synthetic"
    print(f"  Data source: {data_source}")

    df = load_rdw_scrappage(
        data_dir=args.data_dir,
        max_vehicles=args.max_vehicles,
    )
    panel = load_rdw_scrappage(
        data_dir=args.data_dir,
        as_panel=True,
        max_vehicles=args.max_vehicles,
    )

    n_obs = len(df)
    n_vehicles = df["vehicle_id"].nunique()
    n_scrap = df["scrapped"].sum()
    scrap_rate = df["scrapped"].mean()
    years_min = df.groupby("vehicle_id")["year"].count().min()
    years_max = df.groupby("vehicle_id")["year"].count().max()

    print(f"  Observations:      {n_obs:,}")
    print(f"  Vehicles:          {n_vehicles}")
    print(f"  Scrappage events:  {n_scrap} ({scrap_rate:.2%})")
    print(f"  Years/vehicle:     {years_min}-{years_max}")
    print(f"  Age bins:          0-{df['age_bin'].max()}")
    print(f"  Mean age:          {df['age_bin'].mean():.1f}")
    print(f"  Defect levels:     {sorted(df['defect_level'].unique())}")
    print(f"  Mean defect:       {df['defect_level'].mean():.2f}")

    # =========================================================================
    # Step 2: Set Up Environment and Estimate Transitions
    # =========================================================================
    print_section("Step 2: Estimate Transition Probabilities")

    env = RDWScrapageEnvironment(discount_factor=0.95)
    num_states = env.num_states
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec

    print(f"  States: {num_states} ({env._num_age_bins} age x {env._num_defect_levels} defect)")
    print(f"  Actions: {env.num_actions}")
    print(f"  Features: {env.feature_matrix.shape[-1]}")

    transitions = estimate_rdw_transitions(panel, num_states)

    # Report transition statistics for keep action
    # Average probability of staying at same defect level (across all ages)
    from econirl.environments.rdw_scrappage import state_to_components, components_to_state
    n_d = env._num_defect_levels
    stay_probs = []
    worsen_probs = []
    for age in range(min(20, env._num_age_bins)):
        for d in range(n_d):
            s = components_to_state(age, d, n_d)
            s_same_defect = components_to_state(min(age + 1, env._num_age_bins - 1), d, n_d)
            stay_probs.append(transitions[0, s, s_same_defect])
            if d < n_d - 1:
                s_worse = components_to_state(min(age + 1, env._num_age_bins - 1), d + 1, n_d)
                worsen_probs.append(transitions[0, s, s_worse])

    print(f"  Mean P(defect stays same | keep): {np.mean(stay_probs):.4f}")
    print(f"  Mean P(defect worsens | keep):    {np.mean(worsen_probs):.4f}")

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

    params = result_nfxp.parameters
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

    npl_params = result_npl.parameters
    nfxp_params = result_nfxp.parameters

    param_names = env.parameter_names
    print(f"  {'Parameter':<22s} {'NFXP':>10s} {'NPL':>10s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    for i, name in enumerate(param_names):
        print(f"  {name:<22s} {nfxp_params[i].item():>10.4f} {npl_params[i].item():>10.4f}")

    print(f"\n  NFXP LL: {result_nfxp.log_likelihood:.4f}  ({t_nfxp:.1f}s)")
    print(f"  NPL  LL: {result_npl.log_likelihood:.4f}  ({t_npl:.1f}s)")
    print(f"  Agreement: |dLL| = {abs(result_nfxp.log_likelihood - result_npl.log_likelihood):.4f}")

    # =========================================================================
    # Summary
    # =========================================================================
    print_header("Estimation Summary")

    se = result_nfxp.standard_errors

    print(f"""
  Model: RDW Vehicle Scrappage (Rust-style DDC)
  Data:  {n_obs:,} observations, {n_vehicles} vehicles ({data_source})
  State: {num_states} states ({env._num_age_bins} age bins x {env._num_defect_levels} defect levels)

  NFXP Estimates:""")

    for i, name in enumerate(param_names):
        se_val = se[i] if se is not None else float("nan")
        print(f"    {name:<22s} = {nfxp_params[i].item():.6f}  (SE {se_val:.6f})")

    print(f"    Log-likelihood       = {result_nfxp.log_likelihood:.4f}")
    print(f"""
  Cross-validation (NFXP vs NPL):
    |dLL| = {abs(result_nfxp.log_likelihood - result_npl.log_likelihood):.4f}

  Algorithm: Iskhakov et al. (2016) SA->NK polyalgorithm
    Inner solver: Policy iteration
    Outer solver: BHHH with analytical gradient
""")


if __name__ == "__main__":
    main()
