"""Instacart-style grocery reorder dataset for repeat-purchase DDC.

This module generates a synthetic dataset modeled on the Instacart Market
Basket Analysis data (Kaggle 2017). Consumers make sequential decisions
about whether to reorder products from a grocery delivery platform. The
state captures recent purchase history and time since last order. The
action is reorder or skip for a product category.

This is a synthetic generator that captures the DDC structure of repeat
purchase behavior without requiring a Kaggle download. The state space
and transition dynamics are calibrated to match stylized facts from the
Instacart competition data: reorder rates increase with purchase frequency
and decrease with days since last order.

State space:
    n_purchase_buckets x n_recency_buckets discrete states.
    Default: 5 purchase frequency buckets x 4 recency buckets = 20 states.

Action space:
    2 actions: skip (0) and reorder (1).

Reference:
    Instacart Market Basket Analysis (Kaggle 2017).
    Erdem, T. & Keane, M.P. (1996). "Decision-Making Under Uncertainty:
    Capturing Dynamic Brand Choice Processes in Turbulent Consumer Goods
    Markets." Marketing Science.
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from econirl.core.types import Panel
from econirl.environments.instacart import (
    InstacartEnvironment,
    N_ACTIONS,
    N_STATES,
    PURCHASE_LABELS,
    RECENCY_LABELS,
    state_to_buckets,
)
from econirl.simulation.synthetic import simulate_panel


def load_instacart(
    n_individuals: int = 1000,
    n_periods: int = 52,
    as_panel: bool = False,
    seed: int = 42,
    habit_strength: float = 0.3,
    recency_effect: float = 0.5,
    reorder_cost: float = -0.2,
    discount_factor: float = 0.95,
) -> Union[pd.DataFrame, Panel]:
    """Generate grocery reorder trajectory data.

    Creates synthetic consumer trajectories where each period represents
    a weekly shopping opportunity. Consumers decide whether to reorder a
    product category based on purchase history and recency.

    Args:
        n_individuals: Number of consumers to simulate.
        n_periods: Number of weekly periods (default 52 = one year).
        as_panel: If True, return Panel object for econirl estimators.
        seed: Random seed for reproducibility.
        habit_strength: Weight on purchase frequency.
        recency_effect: Weight on order recency.
        reorder_cost: Fixed cost of reordering (negative).
        discount_factor: Time discount factor.

    Returns:
        DataFrame with columns: consumer_id, period, state, action,
        next_state, purchase_bucket, recency_bucket, reordered.

        If as_panel=True, returns Panel object.
    """
    env = InstacartEnvironment(
        habit_strength=habit_strength,
        recency_effect=recency_effect,
        reorder_cost=reorder_cost,
        discount_factor=discount_factor,
        seed=seed,
    )

    panel = simulate_panel(
        env,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )

    if as_panel:
        return panel

    records = []
    for traj in panel.trajectories:
        tid = traj.individual_id
        for t in range(len(traj.states)):
            s = int(traj.states[t])
            a = int(traj.actions[t])
            ns = int(traj.next_states[t])
            pb, rb = state_to_buckets(s)
            records.append({
                "consumer_id": tid,
                "period": t,
                "state": s,
                "action": a,
                "next_state": ns,
                "purchase_bucket": pb,
                "recency_bucket": rb,
                "purchase_label": PURCHASE_LABELS[pb],
                "recency_label": RECENCY_LABELS[rb],
                "reordered": a == 1,
            })

    return pd.DataFrame(records)


def get_instacart_info() -> dict:
    """Return metadata about the Instacart-style dataset."""
    return {
        "name": "Instacart Grocery Reorder (Synthetic)",
        "description": (
            "Synthetic repeat-purchase DDC calibrated to Instacart-style "
            "grocery reorder patterns. 20 states (purchase frequency x recency), "
            "2 actions (skip/reorder)."
        ),
        "source": "Simulated from InstacartEnvironment (calibrated to Kaggle 2017 data)",
        "n_states": N_STATES,
        "n_actions": N_ACTIONS,
        "n_features": 3,
        "state_description": "Purchase frequency bucket x recency bucket",
        "action_description": "Skip (0) / Reorder (1)",
        "true_parameters": {
            "habit_strength": 0.3,
            "recency_effect": 0.5,
            "reorder_cost": -0.2,
        },
        "ground_truth": True,
        "use_case": "Repeat purchase DDC, state dependence in brand choice",
    }
