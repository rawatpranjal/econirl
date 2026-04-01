"""Supermarket pricing and inventory DDC environment.

This module implements a retailer's dynamic pricing and inventory
management problem based on Aguirregabiria (1999) "The Dynamics of
Markups and Inventories in Retailing Firms" (Review of Economic
Studies). A retailer manages a product by choosing each period whether
to run a promotion and whether to place an order from the supplier.
The state captures the current inventory level and whether the product
was on promotion last period.

State space:
    n_inventory_bins x 2 lagged promotion status = 10 states by default.

Action space:
    4 actions: (no promotion, no order), (no promotion, order),
    (promotion, no order), (promotion, order).

Utility specification:
    U(s,a) = markup * markup_indicator
             + holding_cost * inventory_level
             + stockout_penalty * stockout_indicator
             + promotion_cost * promotion_indicator

Reference:
    Aguirregabiria, V. (1999). "The Dynamics of Markups and Inventories
    in Retailing Firms." Review of Economic Studies, 66(2), 275-308.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


N_INVENTORY_BINS = 5
N_PROMO_STATUS = 2
N_STATES = N_INVENTORY_BINS * N_PROMO_STATUS
N_ACTIONS = 4
N_FEATURES = 4

INVENTORY_LABELS = ["very_low", "low", "medium", "high", "very_high"]
PROMO_LABELS = ["regular", "promoted"]
ACTION_LABELS = [
    "regular_no_order",
    "regular_order",
    "promo_no_order",
    "promo_order",
]


def state_to_components(state: int) -> tuple[int, int]:
    """Convert flat state index to (inventory_bin, lagged_promotion)."""
    return state // N_PROMO_STATUS, state % N_PROMO_STATUS


def components_to_state(inventory_bin: int, lagged_promotion: int) -> int:
    """Convert (inventory_bin, lagged_promotion) to flat state index."""
    return inventory_bin * N_PROMO_STATUS + lagged_promotion


def action_to_components(action: int) -> tuple[int, int]:
    """Convert flat action index to (promotion, ordered)."""
    return action // 2, action % 2


class SupermarketEnvironment(DDCEnvironment):
    """Supermarket pricing and inventory environment for retail DDC.

    A retailer manages inventory and pricing for a product category.
    Each period the retailer decides whether to run a promotion
    (lowering price to boost sales) and whether to place an order
    from the supplier (replenishing inventory). Promotions increase
    sales volume but reduce margins. Orders have a fixed cost but
    prevent stockouts.

    State space:
        10 states = 5 inventory bins x 2 lagged promotion status.

    Action space:
        4 actions: (no promo, no order), (no promo, order),
        (promo, no order), (promo, order).

    Example:
        >>> env = SupermarketEnvironment()
        >>> obs, info = env.reset()
        >>> ib, lp = state_to_components(obs)
        >>> print(f"Inventory: {INVENTORY_LABELS[ib]}, Last promo: {PROMO_LABELS[lp]}")
    """

    def __init__(
        self,
        markup_benefit: float = 0.8,
        holding_cost: float = -0.3,
        stockout_penalty: float = -1.0,
        promotion_cost: float = -0.5,
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        data_path: str | Path | None = None,
        seed: int | None = None,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._markup_benefit = markup_benefit
        self._holding_cost = holding_cost
        self._stockout_penalty = stockout_penalty
        self._promotion_cost = promotion_cost

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Load transitions from data
        if data_path is None:
            data_path = Path(__file__).parent.parent / "datasets" / "supermarket_transitions.npz"
        data = np.load(data_path)
        self._transitions_np = data["transitions"]

        self._transition_matrices = jnp.array(self._transitions_np)
        self._feature_matrix = self._build_feature_matrix()

    @property
    def num_states(self) -> int:
        return N_STATES

    @property
    def num_actions(self) -> int:
        return N_ACTIONS

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "markup_benefit": self._markup_benefit,
            "holding_cost": self._holding_cost,
            "stockout_penalty": self._stockout_penalty,
            "promotion_cost": self._promotion_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["markup_benefit", "holding_cost", "stockout_penalty", "promotion_cost"]

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Four features:
        - markup_indicator: 1 for non-promotion actions (higher margin)
        - holding_cost_level: normalized inventory level (higher = more cost)
        - stockout_indicator: 1 if inventory is very low and no order placed
        - promotion_indicator: 1 if running a promotion
        """
        features = np.zeros((N_STATES, N_ACTIONS, N_FEATURES), dtype=np.float32)

        for s in range(N_STATES):
            inv_bin, _lagged_promo = state_to_components(s)
            inv_level = inv_bin / (N_INVENTORY_BINS - 1)

            for a in range(N_ACTIONS):
                promo, ordered = action_to_components(a)

                # Markup: higher when not promoting
                features[s, a, 0] = 1.0 - promo

                # Holding cost: proportional to inventory
                features[s, a, 1] = inv_level

                # Stockout risk: low inventory and not ordering
                if inv_bin == 0 and ordered == 0:
                    features[s, a, 2] = 1.0

                # Promotion indicator
                features[s, a, 3] = float(promo)

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Start with medium inventory, no promotion."""
        dist = np.zeros(N_STATES)
        dist[components_to_state(2, 0)] = 0.5
        dist[components_to_state(2, 1)] = 0.2
        dist[components_to_state(3, 0)] = 0.2
        dist[components_to_state(1, 0)] = 0.1
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        inv_bin, _lagged_promo = state_to_components(state)
        promo, ordered = action_to_components(action)
        inv_level = inv_bin / (N_INVENTORY_BINS - 1)

        u = self._markup_benefit * (1.0 - promo)
        u += self._holding_cost * inv_level
        if inv_bin == 0 and ordered == 0:
            u += self._stockout_penalty
        u += self._promotion_cost * promo
        return u

    def _sample_next_state(self, state: int, action: int) -> int:
        probs = self._transitions_np[action, state]
        return self._np_random.choice(N_STATES, p=probs)

    def _state_to_record(self, state: int, action: int) -> dict:
        inv_bin, lagged_promo = state_to_components(state)
        promo, ordered = action_to_components(action)
        return {
            "inventory_bin": inv_bin,
            "lagged_promotion": lagged_promo,
            "inventory_label": INVENTORY_LABELS[inv_bin],
            "promo_label": PROMO_LABELS[lagged_promo],
            "promotion": promo,
            "ordered": ordered,
            "action_label": ACTION_LABELS[action],
        }

    @classmethod
    def info(cls) -> dict:
        return {
            "name": "Supermarket Pricing/Inventory",
            "description": (
                "Supermarket pricing and inventory DDC. "
                "10 states (inventory bin x lagged promotion), "
                "4 actions (promotion x order decision)."
            ),
            "source": "Aguirregabiria (1999) REStud",
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "n_features": N_FEATURES,
            "state_description": "Inventory quintile bin x lagged promotion status",
            "action_description": "No promo+no order / No promo+order / Promo+no order / Promo+order",
            "ground_truth": False,
            "use_case": "Retail IO, pricing dynamics, inventory management",
        }

    def describe(self) -> str:
        return f"""Supermarket Pricing/Inventory Environment
{'=' * 45}
States: {N_STATES} ({N_INVENTORY_BINS} inventory bins x {N_PROMO_STATUS} lagged promotion)
Actions: {', '.join(ACTION_LABELS)}

True Parameters:
  Markup benefit:    {self._markup_benefit}
  Holding cost:      {self._holding_cost}
  Stockout penalty:  {self._stockout_penalty}
  Promotion cost:    {self._promotion_cost}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
