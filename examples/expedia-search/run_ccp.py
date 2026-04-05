#!/usr/bin/env python3
"""
Expedia Hotel Search -- CCP Estimation
=======================================

Estimates structural parameters of a sequential hotel search model using
Expedia ICDM 2013 competition data. Within each search session, a user
decides to scroll past, click, or book a hotel listing. This is a
sequential search DDC problem where position proxies search cost.

State: (position_bin, price_quintile, quality_bin) -> 5 x 3 x 2 = 30 states
Action: scroll=0, click=1, book=2

Features (8):
    0. position     -- normalized position in results (search fatigue)
    1. price_rel    -- price relative to session mean (price sensitivity)
    2. star_rating  -- hotel star rating (quality)
    3. review_score -- review score (quality signal)
    4. brand        -- prop_brand_bool (brand premium)
    5. promotion    -- promotion_flag (deal effect)
    6. click_cost   -- 1 for click action (evaluation cost)
    7. book_value   -- 1 for book action (match value from booking)

Data: Expedia ICDM 2013 Hotel Recommendation Competition
    train.csv: ~9.9M rows, one row per (session, hotel) pair shown

Reference:
    Honka, E., Hortacsu, A., Vitorino, M.A. (2017). "Advertising, Consumer
    Awareness, and Choice." RAND Journal of Economics 48(3): 611-646.

Usage:
    python examples/expedia-search/run_ccp.py
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

DATA_DIR = Path("data/raw/expedia")
MAX_SESSIONS = 10_000    # use first N search sessions
DISCOUNT_FACTOR = 0.95

# State discretization
POSITION_BINS = [1, 9, 17, 25, 33, 41]   # 5 bins: 1-8, 9-16, 17-24, 25-32, 33+
POSITION_LABELS = 5
PRICE_BINS = 3    # cheap / medium / expensive relative to session
QUALITY_BINS = 2  # above / below session median (star + review combined)

NUM_STATES = POSITION_LABELS * PRICE_BINS * QUALITY_BINS  # 30
NUM_ACTIONS = 3   # scroll=0, click=1, book=2
NUM_FEATURES = 5  # position, price, quality, click_cost, book_value


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(pos_bin: int, price_bin: int, quality_bin: int) -> int:
    return pos_bin * (PRICE_BINS * QUALITY_BINS) + price_bin * QUALITY_BINS + quality_bin


def decode_state(s: int) -> tuple[int, int, int]:
    quality_bin = s % QUALITY_BINS
    remainder = s // QUALITY_BINS
    price_bin = remainder % PRICE_BINS
    pos_bin = remainder // PRICE_BINS
    return pos_bin, price_bin, quality_bin


def bin_position(pos: int) -> int:
    for i in range(len(POSITION_BINS) - 1):
        if pos < POSITION_BINS[i + 1]:
            return i
    return POSITION_LABELS - 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_expedia_panel() -> tuple[Panel, pl.DataFrame]:
    """Load Expedia train.csv and build a Panel for CCP estimation.

    Each search session is one trajectory. The user moves through hotel
    listings sequentially: scrolling past (a=0), clicking (a=1), or booking
    (a=2). Booking and reaching max position are absorbing.

    Returns:
        panel: Panel with one trajectory per search session
        df: processed DataFrame
    """
    print("Loading Expedia train.csv (scanning first columns only)...")

    # Only load the columns we need to keep memory manageable
    needed_cols = [
        "srch_id", "position", "price_usd", "prop_starrating",
        "prop_review_score", "prop_brand_bool", "promotion_flag",
        "click_bool", "booking_bool",
    ]

    df = (
        pl.scan_csv(str(DATA_DIR / "train.csv"), null_values=["NULL"])
        .select(needed_cols)
        .collect()
    )
    print(f"  Loaded {df.shape[0]:,} rows")

    # Limit to first MAX_SESSIONS unique sessions
    session_ids = df["srch_id"].unique().sort()[:MAX_SESSIONS]
    df = df.filter(pl.col("srch_id").is_in(session_ids))
    print(f"  Filtered to {MAX_SESSIONS:,} sessions: {df.shape[0]:,} rows")

    # Derive action: book > click > scroll
    df = df.with_columns(
        pl.when(pl.col("booking_bool") == 1).then(2)
          .when(pl.col("click_bool") == 1).then(1)
          .otherwise(0)
          .cast(pl.Int32).alias("action")
    )

    # Per-session price normalization (z-score)
    df = df.with_columns([
        pl.col("price_usd").mean().over("srch_id").alias("price_mean"),
        pl.col("price_usd").std().over("srch_id").alias("price_std"),
    ])
    df = df.with_columns(
        ((pl.col("price_usd") - pl.col("price_mean")) /
         (pl.col("price_std") + 1e-8)).alias("price_norm")
    )

    # Per-session quality (star + review combined into one score)
    df = df.with_columns(
        pl.col("prop_review_score").fill_null(
            pl.col("prop_review_score").median().over("srch_id")
        ).alias("review_filled")
    )
    df = df.with_columns(
        (pl.col("prop_starrating") / 5.0 + pl.col("review_filled") / 5.0).alias("quality_raw")
    )
    df = df.with_columns(
        pl.col("quality_raw").median().over("srch_id").alias("quality_median")
    )
    df = df.with_columns(
        (pl.col("quality_raw") >= pl.col("quality_median")).cast(pl.Int32).alias("quality_bin")
    )

    # Price bin: tertiles within session using rank-based assignment
    df = df.with_columns(
        pl.col("price_norm").rank(method="ordinal").over("srch_id").alias("price_rank"),
        pl.col("srch_id").count().over("srch_id").alias("session_size"),
    )
    df = df.with_columns(
        pl.when(pl.col("price_rank") <= pl.col("session_size") / 3).then(0)
          .when(pl.col("price_rank") <= 2 * pl.col("session_size") / 3).then(1)
          .otherwise(2)
          .cast(pl.Int32).alias("price_bin")
    )

    # Position bin
    df = df.with_columns(
        pl.when(pl.col("position") < 9).then(0)
          .when(pl.col("position") < 17).then(1)
          .when(pl.col("position") < 25).then(2)
          .when(pl.col("position") < 33).then(3)
          .otherwise(4)
          .cast(pl.Int32).alias("pos_bin")
    )

    # Flat state index
    df = df.with_columns(
        (pl.col("pos_bin") * (PRICE_BINS * QUALITY_BINS)
         + pl.col("price_bin") * QUALITY_BINS
         + pl.col("quality_bin")).alias("state")
    )

    # Sort by session then position
    df = df.sort(["srch_id", "position"])

    # Next state: position increments after scroll; book/click keep same position
    # After a booking action, session ends (absorbing). Model: next_state = 0 for absorbing.
    df = df.with_columns(
        pl.col("state").shift(-1).over("srch_id").fill_null(0).alias("next_state")
    )

    # Build Panel via group_by
    print("Building Panel...")
    grouped = (
        df
        .group_by("srch_id", maintain_order=True)
        .agg([
            pl.col("state").alias("states"),
            pl.col("action").alias("actions"),
            pl.col("next_state").alias("next_states"),
            pl.col("price_norm").alias("prices"),
            pl.col("prop_starrating").alias("stars"),
            pl.col("review_filled").alias("reviews"),
            pl.col("prop_brand_bool").alias("brands"),
            pl.col("promotion_flag").alias("promos"),
        ])
    )

    trajectories = []
    for row in grouped.iter_rows(named=True):
        states = np.array(row["states"], dtype=np.int32)
        actions = np.array(row["actions"], dtype=np.int32)
        next_states = np.array(row["next_states"], dtype=np.int32)

        # Truncate at first book (absorbing)
        book_idx = np.where(actions == 2)[0]
        if len(book_idx) > 0:
            end = book_idx[0] + 1
            states = states[:end]
            actions = actions[:end]
            next_states = next_states[:end]

        if len(states) >= 2:
            traj = Trajectory(
                states=jnp.array(states),
                actions=jnp.array(actions),
                next_states=jnp.array(next_states),
                individual_id=int(row["srch_id"]),
            )
            trajectories.append(traj)

    panel = Panel(trajectories=trajectories)
    print(f"  Panel: {len(trajectories):,} trajectories, "
          f"{sum(len(t.states) for t in trajectories):,} total observations")
    return panel, df


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix() -> np.ndarray:
    """Build feature matrix of shape (NUM_STATES, NUM_ACTIONS, NUM_FEATURES).

    Features:
        0. position   -- pos_bin midpoint normalized, scroll action (search fatigue)
        1. price_rel  -- price_bin normalized [-1,0], click+book (price disutility)
        2. quality    -- quality_bin (0/1), click+book (star+review combined quality)
        3. click_cost -- 1 for click action (evaluation/attention cost)
        4. book_value -- 1 for book action (match value from booking)

    star, review, brand, promotion were collapsed into a single quality dimension
    because all four map to the same quality_bin state variable, causing perfect
    collinearity. The state space only distinguishes above/below-median quality.
    """
    features = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_FEATURES))

    pos_midpoints = np.array([4.5, 12.5, 20.5, 28.5, 36.5]) / 40.0
    # price: 0=cheap (0), 1=mid (-0.5), 2=expensive (-1) -- negative = disutility
    price_values = np.array([0.0, -0.5, -1.0])

    for s in range(NUM_STATES):
        pos_bin, price_bin, quality_bin = decode_state(s)

        # Feature 0: position (scroll action only -- fatigue accumulates)
        features[s, 0, 0] = pos_midpoints[pos_bin]

        # Features 1-2 on click (a=1) and book (a=2)
        for a in [1, 2]:
            features[s, a, 1] = price_values[price_bin]   # price disutility
            features[s, a, 2] = float(quality_bin)        # quality preference

        # Feature 3: click cost (evaluation effort)
        features[s, 1, 3] = 1.0

        # Feature 4: book value (match value)
        features[s, 2, 4] = 1.0

    return features


# ---------------------------------------------------------------------------
# Transition estimation
# ---------------------------------------------------------------------------

def estimate_transitions(panel: Panel) -> np.ndarray:
    """Estimate transition matrices from panel data.

    Scroll (a=0): position increments, price/quality change (empirical).
    Click (a=1): stay in same state (user evaluates, position unchanged).
    Book (a=2): absorbing (session ends, return to state 0).
    """
    scroll_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)
    click_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)

    for traj in panel.trajectories:
        states = np.array(traj.states)
        actions = np.array(traj.actions)
        next_s = np.array(traj.next_states)
        for t in range(len(states) - 1):
            s = int(states[t])
            sp = int(next_s[t])
            a = int(actions[t])
            if 0 <= s < NUM_STATES and 0 <= sp < NUM_STATES:
                if a == 0:
                    scroll_counts[s, sp] += 1
                elif a == 1:
                    click_counts[s, sp] += 1

    def normalize(counts):
        row_sums = counts.sum(axis=1, keepdims=True)
        return np.where(
            row_sums > 0,
            counts / np.maximum(row_sums, 1),
            np.eye(NUM_STATES),   # no data: stay in place
        )

    transitions = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES))
    transitions[0] = normalize(scroll_counts)
    transitions[1] = normalize(click_counts)
    transitions[2, :, 0] = 1.0   # book: absorbing

    n_empty_scroll = (scroll_counts.sum(axis=1) == 0).sum()
    n_empty_click = (click_counts.sum(axis=1) == 0).sum()
    if n_empty_scroll > 0:
        print(f"  Warning: {n_empty_scroll}/{NUM_STATES} states with no scroll observations")
    if n_empty_click > 0:
        print(f"  Warning: {n_empty_click}/{NUM_STATES} states with no click observations")

    return transitions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("Expedia Hotel Search -- CCP Estimation")
    print("=" * 72)

    # Step 1: Load data
    print("\n--- Step 1: Load Data ---")
    panel, df = load_expedia_panel()

    actions = np.concatenate([np.array(t.actions) for t in panel.trajectories])
    print(f"  Action shares: scroll={( actions==0).mean():.3f}, "
          f"click={(actions==1).mean():.3f}, book={(actions==2).mean():.3f}")

    all_states = np.concatenate([np.array(t.states) for t in panel.trajectories])
    state_counts = np.bincount(all_states, minlength=NUM_STATES)
    print(f"  States with observations: {(state_counts > 0).sum()}/{NUM_STATES}")

    # Step 2: Build features
    print("\n--- Step 2: Build Features ---")
    features = build_feature_matrix()
    param_names = ["theta_position", "theta_price", "theta_quality",
                   "click_cost", "book_value"]
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
        print(f"  Condition number (nonzero rows): {cond:.1f}  "
              f"({'OK' if cond < 1e6 else 'HIGH'})")
    print(f"  State coverage: {(state_counts > 0).sum()}/{NUM_STATES}")
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

    # Step 3: Estimate transitions
    print("\n--- Step 3: Estimate Transitions ---")
    transitions = estimate_transitions(panel)

    # Step 4: CCP estimation
    print("\n--- Step 4: CCP Estimation (Hotz-Miller) ---")
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

    # Step 5: NPL refinement
    print("\n--- Step 5: NPL Estimation (K=10) ---")
    npl = CCPEstimator(num_policy_iterations=10, compute_hessian=True, verbose=True)
    t0 = time.time()
    result_npl = npl.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")
    print(result_npl.summary())

    # Step 6: Sanity checks
    print("\n--- Step 6: Sanity Checks ---")
    params = result_npl.parameters
    expected_signs = {
        "theta_position": ("negative", "search fatigue increases with position"),
        "theta_price": ("positive", "price feature is -price_midpoint so positive coef = price sensitive"),
        "theta_quality": ("positive", "higher quality increases click/book utility"),
        "book_value": ("positive", "booking has positive match value"),
    }
    for name, (expected, reason) in expected_signs.items():
        idx = param_names.index(name)
        val = float(params[idx])
        got = "positive" if val > 0 else "negative"
        ok = "PASS" if got == expected else "UNEXPECTED"
        print(f"  {name:18s} = {val:+.4f}  (expected {expected}, got {got}) [{ok}] -- {reason}")

    # Save results
    results = {
        "dataset": "expedia-search",
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
            "state_coverage": int((state_counts > 0).sum()),
            "n_states": NUM_STATES,
        },
    }
    out_path = Path("examples/expedia-search/results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
