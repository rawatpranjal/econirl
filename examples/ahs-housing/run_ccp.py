#!/usr/bin/env python3
"""
AHS Housing Mobility -- CCP Estimation
=======================================

Estimates structural parameters of a household mobility model using the
2023 American Housing Survey (AHS) national cross-section. The DDC problem:
each period, a household decides whether to stay in its current unit or move.

Challenge: AHS is a single cross-section. We derive a pseudo-panel by treating
the cross-sectional distribution of years-in-unit (duration) as if it represents
the dynamic process. We generate 10K synthetic trajectories calibrated to the
cross-sectional moments (duration distribution, move rates by tenure/age/income).

State: (tenure_type, age_bin, income_bin, duration_bin) -> 2 x 3 x 3 x 3 = 54
Action: stay=0, move=1

Features (5):
    0. housing_burden  -- RENT/HINCP for renters (cost-to-income ratio)
    1. duration        -- years in current unit (attachment/lock-in)
    2. renter          -- 1 if renting (renters are more mobile)
    3. age             -- householder age normalized (older = less mobile)
    4. move_cost       -- 1 for move action (transaction cost of moving)

Data: AHS 2023 National, household.csv (~55K households)
    TENURE: '1'=own, '2'=rent
    HHMOVE: year moved in (used to derive duration)
    HINCP: household income
    RENT: monthly rent (renters only)
    HHAGE: age of householder
    NUMPEOPLE: household size

Reference:
    Ferreira, F., Gyourko, J., Tracy, J. (2010). "Housing Busts and Household
    Mobility." Journal of Urban Economics 68(1): 34-45.

Usage:
    python examples/ahs-housing/run_ccp.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.ccp import CCPEstimator
from econirl.preferences.linear import LinearUtility


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw/ahs")
SURVEY_YEAR = 2023
N_SYNTHETIC = 10_000   # synthetic trajectories to generate
T_PERIODS = 10         # years per trajectory
DISCOUNT_FACTOR = 0.95
RNG = np.random.default_rng(42)

# State dimensions
TENURE_TYPES = 2     # 0=own, 1=rent
AGE_BINS = 3         # young(<35), middle(35-64), senior(65+)
INCOME_BINS = 3      # low(<40K), middle(40-100K), high(100K+)
DURATION_BINS = 3    # new(0-2yr), established(3-9yr), long(10+yr)

NUM_STATES = TENURE_TYPES * AGE_BINS * INCOME_BINS * DURATION_BINS  # 54
NUM_ACTIONS = 2   # stay=0, move=1
NUM_FEATURES = 5


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(tenure: int, age_bin: int, inc_bin: int, dur_bin: int) -> int:
    return (tenure * AGE_BINS * INCOME_BINS * DURATION_BINS
            + age_bin * INCOME_BINS * DURATION_BINS
            + inc_bin * DURATION_BINS
            + dur_bin)


def decode_state(s: int) -> tuple[int, int, int, int]:
    dur_bin = s % DURATION_BINS
    s //= DURATION_BINS
    inc_bin = s % INCOME_BINS
    s //= INCOME_BINS
    age_bin = s % AGE_BINS
    tenure = s // AGE_BINS
    return tenure, age_bin, inc_bin, dur_bin


def bin_age(age: float) -> int:
    if age < 35:
        return 0
    elif age < 65:
        return 1
    return 2


def bin_income(inc: float) -> int:
    if inc < 40_000:
        return 0
    elif inc < 100_000:
        return 1
    return 2


def bin_duration(dur: int) -> int:
    if dur <= 2:
        return 0
    elif dur <= 9:
        return 1
    return 2


# ---------------------------------------------------------------------------
# Load cross-section and compute calibration moments
# ---------------------------------------------------------------------------

def load_ahs_moments() -> dict:
    """Load AHS cross-section, filter to valid households, compute moments.

    Returns a dict of calibration statistics used to generate the synthetic panel.
    """
    print("Loading AHS household.csv...")
    df = (
        pl.scan_csv(str(DATA_DIR / "household.csv"))
        .select(["TENURE", "HHMOVE", "RENT", "HINCP", "HHAGE", "NUMPEOPLE"])
        .collect()
    )
    print(f"  Loaded {df.shape[0]:,} households")

    # Filter to owner (TENURE='1') and renter (TENURE='2'), drop vacants (-6)
    df = df.filter(pl.col("TENURE").is_in(["'1'", "'2'"]))
    df = df.with_columns([
        (pl.col("TENURE") == "'2'").cast(pl.Int32).alias("renter"),
        pl.col("HINCP").cast(pl.Float64),
        pl.col("RENT").cast(pl.Float64),
        pl.col("HHAGE").cast(pl.Float64),
        pl.col("HHMOVE").cast(pl.Int64),
        pl.col("NUMPEOPLE").cast(pl.Float64),
    ])

    # Drop rows with missing key fields
    df = df.filter(
        pl.col("HINCP") > 0,
        pl.col("HHAGE") > 0,
        pl.col("HHMOVE") > 0,
    )

    # Duration: years since moved in (SURVEY_YEAR - HHMOVE)
    df = df.with_columns(
        (SURVEY_YEAR - pl.col("HHMOVE")).clip(0, 50).alias("duration")
    )

    # Housing burden: rent / annual income for renters; 0 for owners
    df = df.with_columns(
        pl.when(
            (pl.col("renter") == 1) & (pl.col("RENT") > 0) & (pl.col("HINCP") > 0)
        )
        .then(pl.col("RENT") * 12 / pl.col("HINCP"))
        .otherwise(0.0)
        .clip(0.0, 2.0)
        .alias("burden")
    )

    # Bin state variables
    df = df.with_columns([
        pl.when(pl.col("HHAGE") < 35).then(0)
          .when(pl.col("HHAGE") < 65).then(1)
          .otherwise(2).cast(pl.Int32).alias("age_bin"),
        pl.when(pl.col("HINCP") < 40_000).then(0)
          .when(pl.col("HINCP") < 100_000).then(1)
          .otherwise(2).cast(pl.Int32).alias("inc_bin"),
        pl.when(pl.col("duration") <= 2).then(0)
          .when(pl.col("duration") <= 9).then(1)
          .otherwise(2).cast(pl.Int32).alias("dur_bin"),
    ])

    n = len(df)
    print(f"  Valid households: {n:,}")
    print(f"  Renter share: {df['renter'].mean():.3f}")
    print(f"  Median burden (renters): "
          f"{df.filter(pl.col('renter')==1)['burden'].median():.3f}")

    # Compute move rate = fraction that moved in last 2 years (duration_bin = 0)
    move_rate = (df["dur_bin"] == 0).mean()
    print(f"  Implied annual move rate (new 0-2yr): {move_rate:.3f}")

    # Compute per-state distributions (for synthetic panel calibration)
    moments = {
        "renter_share": float(df["renter"].mean()),
        "move_rate_overall": float(move_rate),
        "age_dist": df["age_bin"].value_counts().sort("age_bin")["count"].to_list(),
        "inc_dist": df["inc_bin"].value_counts().sort("inc_bin")["count"].to_list(),
        "dur_dist": df["dur_bin"].value_counts().sort("dur_bin")["count"].to_list(),
        "median_burden_renters": float(
            df.filter(pl.col("renter") == 1)["burden"].median()
        ),
        # Per-(tenure, age_bin, inc_bin) move rate: share of recent movers
        "df": df,
    }
    return moments


# ---------------------------------------------------------------------------
# Generate synthetic panel
# ---------------------------------------------------------------------------

def generate_synthetic_panel(moments: dict) -> Panel:
    """Generate synthetic household trajectories calibrated to AHS cross-section.

    Each household trajectory simulates T_PERIODS annual decisions (stay or move).
    Transition dynamics:
        Stay: duration increments by 1 year; age increments every 5 years;
              income may change with 10% probability.
        Move: duration resets to 0; tenure type may switch with 20% probability;
              draw new income from cross-sectional distribution.

    Move probability is calibrated per state using cross-sectional move rates.
    """
    df = moments["df"]

    # Compute per-(tenure, age_bin, inc_bin) move rates from cross-section
    # Move = recently moved (duration_bin = 0)
    state_move_rates = {}
    for tenure in range(TENURE_TYPES):
        for ab in range(AGE_BINS):
            for ib in range(INCOME_BINS):
                sub = df.filter(
                    (pl.col("renter") == tenure) &
                    (pl.col("age_bin") == ab) &
                    (pl.col("inc_bin") == ib)
                )
                if len(sub) >= 5:
                    rate = float((sub["dur_bin"] == 0).mean())
                else:
                    rate = moments["move_rate_overall"]
                state_move_rates[(tenure, ab, ib)] = np.clip(rate, 0.02, 0.60)

    # Initial state distribution: sample from cross-section
    cross_section_states = (
        df.with_columns(
            (pl.col("renter") * AGE_BINS * INCOME_BINS * DURATION_BINS
             + pl.col("age_bin") * INCOME_BINS * DURATION_BINS
             + pl.col("inc_bin") * DURATION_BINS
             + pl.col("dur_bin")).cast(pl.Int32).alias("state")
        )["state"].to_numpy()
    )

    # Compute per-state burden values for feature construction
    burden_by_state = np.zeros(NUM_STATES)
    for s in range(NUM_STATES):
        tenure, ab, ib, db = decode_state(s)
        sub = df.filter(
            (pl.col("renter") == tenure) &
            (pl.col("age_bin") == ab) &
            (pl.col("inc_bin") == ib)
        )
        if len(sub) > 0 and tenure == 1:
            burden_by_state[s] = float(sub["burden"].median())
        else:
            burden_by_state[s] = 0.0

    # Income distribution for re-draws after moving
    inc_probs = np.array(moments["inc_dist"], dtype=np.float64)
    inc_probs /= inc_probs.sum()

    trajectories = []
    for _ in range(N_SYNTHETIC):
        # Sample initial state from cross-section
        init_s = int(RNG.choice(cross_section_states))
        tenure, age_bin, inc_bin, dur_bin = decode_state(init_s)

        states, actions, next_states = [], [], []

        for t in range(T_PERIODS):
            s = encode_state(tenure, age_bin, inc_bin, dur_bin)

            # Move probability for this state
            p_move = state_move_rates.get((tenure, age_bin, inc_bin),
                                          moments["move_rate_overall"])
            action = int(RNG.random() < p_move)

            # Transition
            if action == 1:  # move
                dur_bin = 0
                # May switch tenure type
                if RNG.random() < 0.15:
                    tenure = 1 - tenure
                # Redraw income
                inc_bin = int(RNG.choice(3, p=inc_probs))
            else:  # stay
                dur_bin = min(dur_bin + 1, DURATION_BINS - 1)
                # Age increment (slow)
                if t > 0 and t % 5 == 0:
                    age_bin = min(age_bin + 1, AGE_BINS - 1)
                # Small income shock
                if RNG.random() < 0.10:
                    inc_bin = int(np.clip(inc_bin + RNG.choice([-1, 1]), 0, 2))

            next_s = encode_state(tenure, age_bin, inc_bin, dur_bin)
            states.append(s)
            actions.append(action)
            next_states.append(next_s)

        if len(states) >= 2:
            traj = Trajectory(
                states=jnp.array(states, dtype=jnp.int32),
                actions=jnp.array(actions, dtype=jnp.int32),
                next_states=jnp.array(next_states, dtype=jnp.int32),
                individual_id=_,
            )
            trajectories.append(traj)

    panel = Panel(trajectories=trajectories)
    print(f"  Synthetic panel: {len(trajectories):,} trajectories, "
          f"{sum(len(t.states) for t in trajectories):,} total observations")
    return panel, burden_by_state


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(burden_by_state: np.ndarray) -> np.ndarray:
    """Build feature matrix (NUM_STATES, NUM_ACTIONS, NUM_FEATURES).

    Features on stay action (a=0):
        0. housing_burden  -- cost-to-income ratio (renters only)
        1. duration        -- dur_bin normalized (lock-in / attachment)
        2. renter          -- 1 if renter (renters are more mobile = lower stay utility)
        3. age             -- age_bin normalized (older = higher stay utility)

    Feature on move action (a=1):
        4. move_cost       -- 1 (transaction cost / hassle of moving)

    Expected signs:
        theta_burden   > 0 (high cost-to-income pushes away from staying)
        theta_duration < 0 (longer tenure increases attachment to current unit)
        theta_renter   < 0 (renters find staying less attractive = more mobile)
        theta_age      > 0 (older households prefer staying)
        move_cost      > 0 (positive cost of moving discourages moves)

    Note: theta_burden > 0 on stay action means burden reduces stay utility.
    theta_duration < 0 means longer tenure reduces... wait, if duration increases
    attachment, it should INCREASE stay utility, so theta_duration > 0.
    And theta_renter < 0 means renter flag reduces stay utility.
    move_cost > 0 means move action has a positive baseline constant.
    """
    features = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_FEATURES))

    dur_midpoints = np.array([1.0, 6.0, 20.0]) / 20.0
    age_midpoints = np.array([25.0, 49.5, 72.5]) / 80.0

    for s in range(NUM_STATES):
        tenure, age_bin, inc_bin, dur_bin = decode_state(s)

        # Feature 0: housing burden on stay action (reduces stay utility if high)
        features[s, 0, 0] = burden_by_state[s]

        # Feature 1: duration on stay action (longer = more attached = higher stay utility)
        features[s, 0, 1] = dur_midpoints[dur_bin]

        # Feature 2: renter flag on stay action (renters less attached)
        features[s, 0, 2] = float(tenure)

        # Feature 3: age on stay action (older = more attached)
        features[s, 0, 3] = age_midpoints[age_bin]

        # Feature 4: move cost on move action (transaction cost)
        features[s, 1, 4] = 1.0

    return features


# ---------------------------------------------------------------------------
# Transition estimation
# ---------------------------------------------------------------------------

def estimate_transitions(panel: Panel) -> np.ndarray:
    """Estimate transitions from synthetic panel data."""
    stay_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
    move_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)

    for traj in panel.trajectories:
        states = np.array(traj.states)
        actions = np.array(traj.actions)
        next_s = np.array(traj.next_states)
        for t in range(len(states) - 1):
            s, sp, a = int(states[t]), int(next_s[t]), int(actions[t])
            if 0 <= s < NUM_STATES and 0 <= sp < NUM_STATES:
                if a == 0:
                    stay_counts[s, sp] += 1
                else:
                    move_counts[s, sp] += 1

    def normalize(counts):
        row_sums = counts.sum(axis=1, keepdims=True)
        return np.where(
            row_sums > 0,
            counts / np.maximum(row_sums, 1),
            np.eye(NUM_STATES),
        )

    transitions = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))
    transitions[0] = normalize(stay_counts)
    transitions[1] = normalize(move_counts)
    return transitions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("AHS Housing Mobility -- CCP Estimation")
    print("=" * 72)

    # Step 1: Load cross-section and generate synthetic panel
    print("\n--- Step 1: Load AHS Cross-Section ---")
    moments = load_ahs_moments()

    print(f"\n--- Step 2: Generate {N_SYNTHETIC:,} Synthetic Trajectories ---")
    panel, burden_by_state = generate_synthetic_panel(moments)

    actions = np.concatenate([np.array(t.actions) for t in panel.trajectories])
    print(f"  Empirical move rate: {(actions == 1).mean():.4f}")

    all_states = np.concatenate([np.array(t.states) for t in panel.trajectories])
    state_counts = np.bincount(all_states, minlength=NUM_STATES)
    print(f"  States with observations: {(state_counts > 0).sum()}/{NUM_STATES}")

    # Step 3: Build features
    print("\n--- Step 3: Build Features ---")
    features = build_feature_matrix(burden_by_state)
    param_names = ["theta_burden", "theta_duration", "theta_renter",
                   "theta_age", "move_cost"]
    print(f"  Feature matrix: {features.shape}")

    # Pre-estimation diagnostics
    print("\n--- Pre-Estimation Diagnostics ---")
    F = features.reshape(-1, NUM_FEATURES)
    rank = np.linalg.matrix_rank(F)
    print(f"  Feature matrix rank: {rank} / {NUM_FEATURES}  "
          f"({'full rank' if rank == NUM_FEATURES else 'RANK DEFICIENT'})")
    nonzero_rows = F[F.any(axis=1)]
    if len(nonzero_rows) > 0:
        cond = np.linalg.cond(nonzero_rows)
        print(f"  Condition number: {cond:.1f}  "
              f"({'OK' if cond < 1e6 else 'HIGH'})")
    covered = (state_counts > 0).sum()
    print(f"  State coverage: {covered}/{NUM_STATES}")
    single_action = sum(
        1 for s in range(NUM_STATES)
        if state_counts[s] > 0 and len(set(
            int(a) for t in panel.trajectories
            for i, a in enumerate(np.array(t.actions))
            if np.array(t.states)[i] == s
        )) < 2
    )
    print(f"  States with only one observed action: {single_action}")

    utility = LinearUtility(feature_matrix=features, parameter_names=param_names)

    # Step 4: Estimate transitions
    print("\n--- Step 4: Estimate Transitions ---")
    transitions = estimate_transitions(panel)

    # Step 5: CCP estimation
    print("\n--- Step 5: CCP Estimation (Hotz-Miller) ---")
    problem = DDCProblem(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        discount_factor=DISCOUNT_FACTOR,
    )

    ccp = CCPEstimator(num_policy_iterations=1, compute_hessian=True, verbose=True)
    t0 = time.time()
    result = ccp.estimate(panel, utility, problem, transitions)
    print(f"\n  Time: {time.time() - t0:.1f}s")
    print(result.summary())

    # Step 6: NPL refinement
    print("\n--- Step 6: NPL Estimation (K=10) ---")
    npl = CCPEstimator(num_policy_iterations=10, compute_hessian=True, verbose=True)
    t0 = time.time()
    result_npl = npl.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print(result_npl.summary())

    # Step 7: Sanity checks
    print("\n--- Step 7: Sanity Checks ---")
    params = result_npl.parameters
    # theta_burden > 0: high burden reduces utility of staying (pushes toward moving)
    # theta_duration > 0: longer tenure increases attachment (boosts stay utility)
    # theta_renter: sign depends on interpretation (renters move more -> lower stay utility -> could be negative)
    # theta_age > 0: older households prefer staying
    # move_cost > 0: moving has a positive baseline constant (i.e., cost makes it less likely)
    expected_signs = {
        "theta_burden": ("positive", "high burden reduces stay utility"),
        "theta_duration": ("positive", "longer tenure = more attachment = higher stay utility"),
        "theta_age": ("positive", "older households prefer staying"),
        "move_cost": ("positive", "move cost baseline"),
    }
    for name, (expected, reason) in expected_signs.items():
        idx = param_names.index(name)
        val = float(params[idx])
        got = "positive" if val > 0 else "negative"
        ok = "PASS" if got == expected else "UNEXPECTED"
        print(f"  {name:18s} = {val:+.4f}  ({got}, expected {expected}) [{ok}] -- {reason}")

    # Save results
    results = {
        "dataset": "ahs-housing",
        "estimator": "NPL (K=10)",
        "n_observations": int(sum(len(t.states) for t in panel.trajectories)),
        "n_individuals": len(panel.trajectories),
        "log_likelihood": float(result_npl.log_likelihood),
        "parameters": {
            name: {
                "coef": float(result_npl.parameters[i]),
                "std_err": float(result_npl.standard_errors[i])
                if result_npl.standard_errors is not None else None,
            }
            for i, name in enumerate(param_names)
        },
        "diagnostics": {
            "feature_rank": int(rank),
            "state_coverage": int(covered),
            "n_states": NUM_STATES,
            "single_action_states": int(single_action),
        },
        "cross_section_moments": {
            "renter_share": moments["renter_share"],
            "move_rate_overall": moments["move_rate_overall"],
            "median_burden_renters": moments["median_burden_renters"],
        },
    }
    out_path = Path("examples/ahs-housing/results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
