"""Dixit entry/exit DDC environment for firm dynamics estimation.

This module implements the Dixit (1989) model of firm entry and exit
under uncertainty, as used in the Abbring and Klein DDC teaching
package. A firm observes a market profitability state and decides
whether to be active (enter or stay) or inactive (exit or stay out).
Entry and exit have sunk costs that create hysteresis: a firm may
remain active in a market that would not justify fresh entry.

State space:
    n_profit_bins x 2 incumbent status = 20 states by default.
    The first dimension is market profitability (discretized AR(1)).
    The second dimension is whether the firm was active last period.

Action space:
    2 actions: Inactive (0) and Active (1).

Utility specification:
    U(s, a=active)   = profit_slope * profit_level
                        - entry_cost * (was_inactive)
                        + operating_cost
    U(s, a=inactive) = - exit_cost * (was_active)

Reference:
    Dixit, A.K. (1989). "Entry and Exit Decisions under Uncertainty."
    Journal of Political Economy, 97(3), 620-638.

    Abbring, J.H. & Klein, T.J. (2020). "Dynamic Discrete Choice."
    Teaching package: https://github.com/jabbring/dynamic-discrete-choice
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


N_PROFIT_BINS = 10
N_INCUMBENT_STATUS = 2
N_STATES = N_PROFIT_BINS * N_INCUMBENT_STATUS
N_ACTIONS = 2
N_FEATURES = 4

PROFIT_LABELS = [f"profit_{i}" for i in range(N_PROFIT_BINS)]
STATUS_LABELS = ["inactive", "active"]


def state_to_components(state: int) -> tuple[int, int]:
    """Convert flat state index to (profit_bin, incumbent_status).

    incumbent_status: 0 = was inactive last period, 1 = was active.
    """
    return state // N_INCUMBENT_STATUS, state % N_INCUMBENT_STATUS


def components_to_state(profit_bin: int, incumbent_status: int) -> int:
    """Convert (profit_bin, incumbent_status) to flat state index."""
    return profit_bin * N_INCUMBENT_STATUS + incumbent_status


def _build_ar1_transition(n_bins: int, persistence: float) -> np.ndarray:
    """Build a discretized AR(1) transition matrix using Tauchen method.

    Creates a symmetric Markov chain on n_bins grid points that
    approximates an AR(1) process with the given persistence parameter.
    Higher persistence means the profit state is more sticky.
    """
    trans = np.zeros((n_bins, n_bins), dtype=np.float32)
    for i in range(n_bins):
        for j in range(n_bins):
            distance = abs(i - j)
            if distance == 0:
                trans[i, j] = persistence
            else:
                trans[i, j] = (1.0 - persistence) * (0.5 ** distance)
        trans[i] /= trans[i].sum()
    return trans


class EntryExitEnvironment(DDCEnvironment):
    """Dixit entry/exit environment for firm dynamics DDC.

    A firm observes market profitability and decides whether to be
    active or inactive. Entering requires paying a sunk entry cost.
    Exiting requires paying a sunk exit cost. Remaining active incurs
    a fixed operating cost but earns profit proportional to the market
    state. These sunk costs create a band of inaction (hysteresis):
    once active, a firm stays active even when profits dip below the
    entry threshold.

    State space:
        20 states = 10 profit bins x 2 incumbent status.

    Action space:
        Inactive (0) or Active (1).

    Example:
        >>> env = EntryExitEnvironment()
        >>> obs, info = env.reset()
        >>> pb, status = state_to_components(obs)
        >>> print(f"Profit: {PROFIT_LABELS[pb]}, Status: {STATUS_LABELS[status]}")
    """

    def __init__(
        self,
        profit_slope: float = 1.0,
        entry_cost: float = -2.0,
        exit_cost: float = -0.5,
        operating_cost: float = -0.5,
        persistence: float = 0.7,
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._profit_slope = profit_slope
        self._entry_cost = entry_cost
        self._exit_cost = exit_cost
        self._operating_cost = operating_cost
        self._persistence = persistence

        # Profit grid: evenly spaced from -1 to +1
        self._profit_grid = np.linspace(-1.0, 1.0, N_PROFIT_BINS)
        self._profit_transition = _build_ar1_transition(N_PROFIT_BINS, persistence)

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self._transition_matrices = self._build_transition_matrices()
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
            "profit_slope": self._profit_slope,
            "entry_cost": self._entry_cost,
            "exit_cost": self._exit_cost,
            "operating_cost": self._operating_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["profit_slope", "entry_cost", "exit_cost", "operating_cost"]

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition matrices for inactive and active actions.

        The profit bin transitions follow the AR(1) Markov chain
        regardless of the action. The incumbent status transitions
        deterministically: choosing active sets status to 1, choosing
        inactive sets status to 0.
        """
        transitions = np.zeros((N_ACTIONS, N_STATES, N_STATES), dtype=np.float32)

        for s in range(N_STATES):
            pb, _status = state_to_components(s)

            for next_pb in range(N_PROFIT_BINS):
                prob = self._profit_transition[pb, next_pb]

                # Action 0 (inactive): next status = 0
                ns_inactive = components_to_state(next_pb, 0)
                transitions[0, s, ns_inactive] = prob

                # Action 1 (active): next status = 1
                ns_active = components_to_state(next_pb, 1)
                transitions[1, s, ns_active] = prob

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility.

        Four features:
        - profit_flow: profit_level * active_indicator
        - entry_cost_indicator: 1 if entering (was inactive, choosing active)
        - exit_cost_indicator: 1 if exiting (was active, choosing inactive)
        - operating_cost_indicator: 1 if choosing active
        """
        features = np.zeros((N_STATES, N_ACTIONS, N_FEATURES), dtype=np.float32)

        for s in range(N_STATES):
            pb, status = state_to_components(s)
            profit_level = self._profit_grid[pb]

            # Action 1 (active)
            features[s, 1, 0] = profit_level  # profit flow
            if status == 0:
                features[s, 1, 1] = 1.0  # entry cost (entering)
            features[s, 1, 3] = 1.0  # operating cost

            # Action 0 (inactive)
            if status == 1:
                features[s, 0, 2] = 1.0  # exit cost (exiting)

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """New firms start inactive with profit drawn from stationary dist."""
        dist = np.zeros(N_STATES)
        # Start inactive, uniform over middle profit bins
        for pb in range(2, N_PROFIT_BINS - 2):
            s = components_to_state(pb, 0)
            dist[s] = 1.0
        dist /= dist.sum()
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        pb, status = state_to_components(state)
        profit_level = self._profit_grid[pb]

        if action == 1:  # active
            u = self._profit_slope * profit_level + self._operating_cost
            if status == 0:  # entering
                u += self._entry_cost
            return u
        else:  # inactive
            if status == 1:  # exiting
                return self._exit_cost
            return 0.0

    def _sample_next_state(self, state: int, action: int) -> int:
        pb, _status = state_to_components(state)
        next_pb = self._np_random.choice(
            N_PROFIT_BINS, p=self._profit_transition[pb]
        )
        next_status = 1 if action == 1 else 0
        return components_to_state(next_pb, next_status)

    def _state_to_record(self, state: int, action: int) -> dict:
        pb, status = state_to_components(state)
        return {
            "profit_bin": pb,
            "incumbent_status": status,
            "profit_label": PROFIT_LABELS[pb],
            "status_label": STATUS_LABELS[status],
            "is_active": action == 1,
            "entered": status == 0 and action == 1,
            "exited": status == 1 and action == 0,
        }

    @classmethod
    def info(cls) -> dict:
        return {
            "name": "Dixit Entry/Exit (Synthetic)",
            "description": (
                "Synthetic firm entry/exit DDC from the Dixit (1989) model. "
                "20 states (profit bin x incumbent status), 2 actions "
                "(inactive/active). Sunk entry and exit costs create hysteresis."
            ),
            "source": "Synthetic (Abbring-Klein teaching package)",
            "n_states": N_STATES,
            "n_actions": N_ACTIONS,
            "n_features": N_FEATURES,
            "state_description": "Profit bin x incumbent status",
            "action_description": "Inactive (0) / Active (1)",
            "ground_truth": True,
            "use_case": "Firm dynamics, entry/exit hysteresis, industrial organization",
        }

    def describe(self) -> str:
        return f"""Dixit Entry/Exit Environment
{'=' * 40}
States: {N_STATES} ({N_PROFIT_BINS} profit bins x {N_INCUMBENT_STATUS} incumbent status)
Actions: Inactive (0), Active (1)

True Parameters:
  Profit slope:    {self._profit_slope}
  Entry cost:      {self._entry_cost}
  Exit cost:       {self._exit_cost}
  Operating cost:  {self._operating_cost}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
  Persistence (rho):       {self._persistence}
"""
