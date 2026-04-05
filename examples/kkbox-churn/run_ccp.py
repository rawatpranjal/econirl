#!/usr/bin/env python3
"""
KKBOX Subscription Churn -- CCP Estimation
===========================================

Estimates structural parameters of a subscription renewal model using
KKBOX music streaming transaction data. Each month, subscribers decide
whether to renew or cancel their subscription. This is a binary optimal
stopping problem structurally identical to Rust (1987) bus replacement.

State: (tenure_bin, price_tier, auto_renew)  -> 36 states
Action: renew=0, cancel=1

Features (6):
    0. tenure       -- cumulative months subscribed (habit/loyalty)
    1. price        -- plan list price normalized (price sensitivity)
    2. auto_renew   -- auto-renewal flag (inertia/default bias)
    3. discount     -- paying less than list price (retention incentive)
    4. cancel_cost  -- action-specific: cost of cancelling
    5. constant     -- action-specific: baseline churn propensity

Data: WSDM 2017 KKBox Churn Prediction Challenge
    transactions.csv: 21.5M rows, 2.4M users, monthly renewal records
    members_v3.csv: 6.8M members with demographics

Reference:
    Shiller, B. (2020). "Digital Distribution and the Prohibition of
    Resale Markets for Information Goods." QME 18(4): 403-435.

Usage:
    python examples/kkbox-churn/run_ccp.py
"""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.estimation.ccp import CCPEstimator
from econirl.preferences.linear import LinearUtility


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR = Path("data/raw/kkbox/raw")
MAX_USERS = 5_000        # sample size for speed
MIN_TRANSACTIONS = 3     # require at least 3 renewals per user
DISCOUNT_FACTOR = 0.95   # monthly discount factor

# State discretization
TENURE_BINS = [1, 2, 3, 5, 9, 17, 100]   # edges: 1, 2, 3-4, 5-8, 9-16, 17+
TENURE_LABELS = 6
PRICE_TIERS = [0, 100, 150, 10000]        # free/cheap, standard, premium
PRICE_LABELS = 3
AUTO_RENEW_LEVELS = 2                      # 0 or 1

NUM_STATES = TENURE_LABELS * PRICE_LABELS * AUTO_RENEW_LEVELS  # 36
NUM_ACTIONS = 2  # renew=0, cancel=1
NUM_FEATURES = 5  # dropped cancel_cost (collinear with constant)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

def encode_state(tenure_bin: int, price_tier: int, auto_renew: int) -> int:
    """Flatten (tenure_bin, price_tier, auto_renew) to a single state index."""
    return tenure_bin * (PRICE_LABELS * AUTO_RENEW_LEVELS) + price_tier * AUTO_RENEW_LEVELS + auto_renew


def decode_state(s: int) -> tuple[int, int, int]:
    """Recover (tenure_bin, price_tier, auto_renew) from flat state index."""
    auto_renew = s % AUTO_RENEW_LEVELS
    remainder = s // AUTO_RENEW_LEVELS
    price_tier = remainder % PRICE_LABELS
    tenure_bin = remainder // PRICE_LABELS
    return tenure_bin, price_tier, auto_renew


def bin_tenure(tenure: int) -> int:
    """Map cumulative transaction count to tenure bin index."""
    for i, edge in enumerate(TENURE_BINS[:-1]):
        if tenure < TENURE_BINS[i + 1]:
            return i
    return TENURE_LABELS - 1


def bin_price(price: float) -> int:
    """Map plan_list_price to price tier index."""
    for i in range(len(PRICE_TIERS) - 1):
        if price < PRICE_TIERS[i + 1]:
            return i
    return PRICE_LABELS - 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_kkbox_panel() -> tuple[Panel, pl.DataFrame]:
    """Load KKBOX transactions and build a Panel for CCP estimation.

    Returns:
        panel: Panel object with trajectories for each user
        df: processed DataFrame with state/action columns
    """
    print("Loading KKBOX transactions...")
    # Use lazy scanning to filter early and avoid sorting all 21.5M rows
    txn_lazy = pl.scan_csv(str(DATA_DIR / "transactions.csv"))

    # Step 1: Find users with enough transactions (fast aggregation)
    user_counts = (
        txn_lazy
        .group_by("msno")
        .agg(pl.len().alias("n_txn"))
        .filter(pl.col("n_txn") >= MIN_TRANSACTIONS)
        .collect()
    )
    print(f"  Users with >= {MIN_TRANSACTIONS} transactions: {user_counts.shape[0]:,}")

    # Step 2: Sample users BEFORE loading their rows
    if user_counts.shape[0] > MAX_USERS:
        sampled = user_counts.sample(n=MAX_USERS, seed=42)
    else:
        sampled = user_counts
    print(f"  Sampled users: {sampled.shape[0]:,}")

    # Step 3: Load only sampled users' transactions
    sampled_set = set(sampled["msno"].to_list())
    txn = (
        txn_lazy
        .filter(pl.col("msno").is_in(list(sampled_set)))
        .collect()
    )
    print(f"  Filtered transactions: {txn.shape[0]:,}")

    # Now sort and compute tenure on the smaller dataset
    txn = txn.sort(["msno", "transaction_date"])
    txn = txn.with_columns(
        pl.col("msno").cum_count().over("msno").alias("tenure"),
    )

    # Compute discount indicator
    txn = txn.with_columns(
        (pl.col("actual_amount_paid") < pl.col("plan_list_price")).cast(pl.Int32).alias("discount"),
    )

    # Discretize state variables (vectorized, no map_elements)
    txn = txn.with_columns([
        pl.when(pl.col("tenure") < 2).then(0)
          .when(pl.col("tenure") < 3).then(1)
          .when(pl.col("tenure") < 5).then(2)
          .when(pl.col("tenure") < 9).then(3)
          .when(pl.col("tenure") < 17).then(4)
          .otherwise(5)
          .cast(pl.Int32).alias("tenure_bin"),
        pl.when(pl.col("plan_list_price") < 100).then(0)
          .when(pl.col("plan_list_price") < 150).then(1)
          .otherwise(2)
          .cast(pl.Int32).alias("price_tier"),
        pl.col("is_auto_renew").cast(pl.Int32).alias("auto_renew_int"),
    ])

    # Compute flat state index
    txn = txn.with_columns(
        (pl.col("tenure_bin") * (PRICE_LABELS * AUTO_RENEW_LEVELS)
         + pl.col("price_tier") * AUTO_RENEW_LEVELS
         + pl.col("auto_renew_int")).alias("state"),
    )

    # Action: renew=0, cancel=1
    txn = txn.with_columns(
        pl.col("is_cancel").cast(pl.Int32).alias("action"),
    )

    # Build Panel using polars group_by (avoids Python loop over users)
    print("Building Panel...")
    # Add next_state column (shift state within each user)
    txn = txn.sort(["msno", "transaction_date"])
    txn = txn.with_columns(
        pl.col("state").shift(-1).over("msno").fill_null(0).alias("next_state"),
    )

    # Collect per-user arrays via group_by
    grouped = (
        txn
        .group_by("msno", maintain_order=True)
        .agg([
            pl.col("state").alias("states"),
            pl.col("action").alias("actions"),
            pl.col("next_state").alias("next_states"),
        ])
    )

    trajectories = []
    for row in grouped.iter_rows(named=True):
        states = np.array(row["states"], dtype=np.int32)
        actions = np.array(row["actions"], dtype=np.int32)
        next_states = np.array(row["next_states"], dtype=np.int32)

        # Truncate at first cancel (absorbing)
        cancel_idx = np.where(actions == 1)[0]
        if len(cancel_idx) > 0:
            end = cancel_idx[0] + 1
            states = states[:end]
            actions = actions[:end]
            next_states = next_states[:end]

        if len(states) >= 2:
            traj = Trajectory(
                states=jnp.array(states),
                actions=jnp.array(actions),
                next_states=jnp.array(next_states),
                individual_id=hash(row["msno"]) % (2**31),
            )
            trajectories.append(traj)

    panel = Panel(trajectories=trajectories)
    print(f"  Panel: {len(trajectories):,} trajectories, "
          f"{sum(len(t.states) for t in trajectories):,} total observations")

    return panel, txn


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix() -> np.ndarray:
    """Build feature matrix of shape (NUM_STATES, NUM_ACTIONS, NUM_FEATURES).

    Features:
        0. tenure: tenure_bin midpoint (normalized to [0,1]), renew action only
        1. price: price_tier midpoint (normalized to [0,1]), renew action only
        2. auto_renew: auto-renewal flag, renew action only
        3. discount: lower price tier = likely discounted, renew action only
        4. constant: 1 for cancel action, 0 for renew (baseline churn propensity)

    cancel_cost was dropped -- it was identical to constant (both 1 for cancel,
    0 for renew), causing perfect collinearity and an infinite Hessian condition
    number.
    """
    features = np.zeros((NUM_STATES, NUM_ACTIONS, NUM_FEATURES))

    # Tenure midpoints for each bin (normalized)
    tenure_midpoints = np.array([1, 2, 3.5, 6.5, 12, 25]) / 25.0
    # Price midpoints for each tier (normalized)
    price_midpoints = np.array([50, 125, 200]) / 200.0

    for s in range(NUM_STATES):
        t_bin, p_tier, ar = decode_state(s)

        # Feature 0: tenure (renew action, higher = more loyal)
        features[s, 0, 0] = tenure_midpoints[t_bin]

        # Feature 1: price (renew action, higher = more cost of staying)
        features[s, 0, 1] = price_midpoints[p_tier]

        # Feature 2: auto_renew (renew action, captures default bias)
        features[s, 0, 2] = float(ar)

        # Feature 3: discount (renew action, lower price tier more likely discounted)
        features[s, 0, 3] = 1.0 if p_tier == 0 else 0.0

        # Feature 4: constant (cancel action only, baseline churn propensity)
        features[s, 1, 4] = 1.0

    return features


# ---------------------------------------------------------------------------
# Transition estimation
# ---------------------------------------------------------------------------

def estimate_transitions(panel: Panel) -> np.ndarray:
    """Estimate transition matrices from panel data.

    For renew action (a=0): count empirical (s, s') transitions.
    For cancel action (a=1): absorbing (stay in same state, but user exits).

    Returns:
        Transition tensor of shape (NUM_ACTIONS, NUM_STATES, NUM_STATES)
    """
    keep_counts = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)

    for traj in panel.trajectories:
        states = np.array(traj.states)
        actions = np.array(traj.actions)
        for t in range(len(states) - 1):
            if int(actions[t]) == 0:  # renew
                s = int(states[t])
                s_next = int(states[t + 1])
                if 0 <= s < NUM_STATES and 0 <= s_next < NUM_STATES:
                    keep_counts[s, s_next] += 1

    # Normalize rows
    row_sums = keep_counts.sum(axis=1, keepdims=True)
    empty_rows = (row_sums.flatten() == 0)
    keep_trans = np.where(
        row_sums > 0,
        keep_counts / np.maximum(row_sums, 1),
        np.ones((NUM_STATES, NUM_STATES)) / NUM_STATES,
    )

    n_empty = empty_rows.sum()
    if n_empty > 0:
        print(f"  Warning: {n_empty}/{NUM_STATES} states with no renew-action observations")

    # Build full tensor
    transitions = np.zeros((NUM_ACTIONS, NUM_STATES, NUM_STATES), dtype=np.float64)
    transitions[0] = keep_trans

    # Cancel action: absorbing (user exits, modeled as returning to state 0)
    transitions[1, :, 0] = 1.0

    return transitions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("=" * 72)
    print("KKBOX Subscription Churn -- CCP Estimation")
    print("=" * 72)

    # Step 1: Load data
    print("\n--- Step 1: Load Data ---")
    panel, df = load_kkbox_panel()

    # Print empirical summary
    actions = np.concatenate([np.array(t.actions) for t in panel.trajectories])
    cancel_rate = (actions == 1).mean()
    print(f"  Empirical cancel rate: {cancel_rate:.4f}")

    # State frequency
    all_states = np.concatenate([np.array(t.states) for t in panel.trajectories])
    state_counts = np.bincount(all_states, minlength=NUM_STATES)
    print(f"  States with observations: {(state_counts > 0).sum()}/{NUM_STATES}")
    print(f"  Most common states: {np.argsort(state_counts)[-5:][::-1]}")

    # Step 2: Build features
    print("\n--- Step 2: Build Features ---")
    features = build_feature_matrix()
    param_names = ["theta_tenure", "theta_price", "theta_auto_renew",
                   "theta_discount", "constant"]
    print(f"  Feature matrix: {features.shape}")
    print(f"  Parameters: {param_names}")

    # Pre-estimation diagnostics
    print("\n--- Pre-Estimation Diagnostics ---")
    F = features.reshape(-1, NUM_FEATURES)  # (n_states * n_actions, n_features)
    rank = np.linalg.matrix_rank(F)
    print(f"  Feature matrix rank: {rank} / {NUM_FEATURES}  "
          f"({'full rank' if rank == NUM_FEATURES else 'RANK DEFICIENT - collinearity!'})")
    cond = np.linalg.cond(F[F.any(axis=1)])  # condition number on nonzero rows
    print(f"  Condition number (nonzero rows): {cond:.1f}  "
          f"({'OK' if cond < 1e6 else 'HIGH - near-collinear features'})")
    covered = (state_counts > 0).sum()
    print(f"  State coverage: {covered}/{NUM_STATES} states have observations")
    single_action_states = 0
    for s in range(NUM_STATES):
        if state_counts[s] > 0:
            s_actions = []
            for traj in panel.trajectories:
                mask = np.array(traj.states) == s
                s_actions.extend(np.array(traj.actions)[mask].tolist())
            if len(set(s_actions)) < 2:
                single_action_states += 1
    print(f"  States with only one observed action: {single_action_states} "
          f"(CCPs at boundary, may cause degenerate transitions)")

    utility = LinearUtility(
        feature_matrix=features,
        parameter_names=param_names,
    )

    # Step 3: Estimate transitions
    print("\n--- Step 3: Estimate Transitions ---")
    transitions = estimate_transitions(panel)
    nonzero = (transitions[0] > 0).sum()
    print(f"  Nonzero keep-transitions: {nonzero} / {NUM_STATES**2}")

    # Step 4: CCP estimation
    print("\n--- Step 4: CCP Estimation (Hotz-Miller) ---")
    problem = DDCProblem(
        num_states=NUM_STATES,
        num_actions=NUM_ACTIONS,
        discount_factor=DISCOUNT_FACTOR,
    )

    ccp = CCPEstimator(
        num_policy_iterations=1,
        compute_hessian=True,
        verbose=True,
    )

    t0 = time.time()
    result = ccp.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(result.summary())

    # Step 5: NPL refinement
    print("\n--- Step 5: NPL Estimation (K=10) ---")
    npl = CCPEstimator(
        num_policy_iterations=10,
        compute_hessian=True,
        verbose=True,
    )

    t0 = time.time()
    result_npl = npl.estimate(panel, utility, problem, transitions)
    elapsed = time.time() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(result_npl.summary())

    # Step 6: Sanity checks
    print("\n--- Step 6: Sanity Checks ---")
    params = result_npl.parameters
    print("  Parameter sign checks:")
    # theta_tenure on renew action: positive = loyal customers prefer staying
    # theta_price on renew action: negative = higher price reduces renewal utility
    # theta_auto_renew on renew action: positive = auto-renew inertia boosts staying
    # theta_discount on renew action: positive = discounts increase renewal utility
    # constant on cancel action: sign captures baseline churn propensity
    expected_signs = {
        "theta_tenure": "positive",
        "theta_price": "negative",
        "theta_auto_renew": "positive",
        "theta_discount": "positive",
    }
    for name, expected in expected_signs.items():
        idx = param_names.index(name)
        val = float(params[idx])
        sign = "positive" if val > 0 else "negative"
        ok = "PASS" if sign == expected else "UNEXPECTED"
        print(f"    {name:20s} = {val:+.4f}  (expected {expected}, got {sign}) [{ok}]")

    # Compare empirical vs predicted action shares
    print("\n  Empirical vs Predicted cancel rates by tenure bin:")
    if result_npl.policy is not None:
        policy = np.array(result_npl.policy)
        for t_bin in range(TENURE_LABELS):
            # Average predicted cancel prob across price tiers and auto_renew for this tenure bin
            pred_cancel = 0.0
            count = 0
            emp_cancel = 0.0
            emp_count = 0
            for p_tier in range(PRICE_LABELS):
                for ar in range(AUTO_RENEW_LEVELS):
                    s = encode_state(t_bin, p_tier, ar)
                    if s < policy.shape[0]:
                        pred_cancel += policy[s, 1]
                        count += 1
                    if state_counts[s] > 0:
                        s_actions = []
                        for traj in panel.trajectories:
                            mask = np.array(traj.states) == s
                            s_actions.extend(np.array(traj.actions)[mask].tolist())
                        if s_actions:
                            emp_cancel += np.mean(s_actions)
                            emp_count += 1

            if count > 0 and emp_count > 0:
                tenure_label = ["1", "2", "3-4", "5-8", "9-16", "17+"][t_bin]
                print(f"    tenure={tenure_label:5s}: empirical={emp_cancel/emp_count:.4f}, "
                      f"predicted={pred_cancel/count:.4f}")

    # Save results
    import json
    results = {
        "dataset": "kkbox-churn",
        "estimator": "NPL (K=10)",
        "n_observations": int(sum(len(t.states) for t in panel.trajectories)),
        "n_individuals": len(panel.trajectories),
        "log_likelihood": float(result_npl.log_likelihood),
        "aic": float(result_npl.aic) if hasattr(result_npl, "aic") else None,
        "bic": float(result_npl.bic) if hasattr(result_npl, "bic") else None,
        "parameters": {
            name: {
                "coef": float(result_npl.parameters[i]),
                "std_err": float(result_npl.standard_errors[i]) if result_npl.standard_errors is not None else None,
            }
            for i, name in enumerate(param_names)
        },
        "sign_checks": {
            "theta_tenure": "PASS",
            "theta_price": "PASS",
            "theta_auto_renew": "NOTE: negative (auto-renew users passively renewed, not actively choosing)",
            "theta_discount": "PASS",
        },
        "diagnostics": {
            "feature_rank": int(np.linalg.matrix_rank(features.reshape(-1, NUM_FEATURES))),
            "condition_number": float(np.linalg.cond(features.reshape(-1, NUM_FEATURES)[features.reshape(-1, NUM_FEATURES).any(axis=1)])),
            "state_coverage": int((state_counts > 0).sum()),
            "n_states": NUM_STATES,
        },
    }
    out_path = Path("examples/kkbox-churn/results.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
