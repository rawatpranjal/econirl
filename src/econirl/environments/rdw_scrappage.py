"""RDW vehicle scrappage environment.

This module implements a dynamic discrete choice model for vehicle scrappage
using Dutch RDW (Rijksdienst voor het Wegverkeer) inspection data. The decision
problem is an optimal stopping model where a vehicle owner observes the car's
age and defect severity from the mandatory annual APK inspection and decides
whether to keep the car or scrap it.

The model:
- State: Joint (age_bin, defect_level) encoded as a flat index
- Actions: Keep operating (0) or Scrap/replace (1)
- Utility: Operating cost increases with age and defect severity;
    scrappage has a fixed replacement cost
- Transitions: Age increments deterministically; defect severity
    transitions stochastically with age-dependent probabilities

The default parameters are calibrated to produce an annual scrappage rate
of roughly 5 to 8 percent, consistent with Dutch CBS statistics for
passenger vehicles aged 5 to 20 years.

Reference:
    RDW Open Data: https://opendata.rdw.nl
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines: An Empirical
    Model of Harold Zurcher." Econometrica, 55(5), 999-1033.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


def state_to_components(state: int, num_defect_levels: int = 3) -> tuple[int, int]:
    """Convert flat state index to (age_bin, defect_level)."""
    age_bin = state // num_defect_levels
    defect_level = state % num_defect_levels
    return age_bin, defect_level


def components_to_state(
    age_bin: int, defect_level: int, num_defect_levels: int = 3
) -> int:
    """Convert (age_bin, defect_level) to flat state index."""
    return age_bin * num_defect_levels + defect_level


class RDWScrapageEnvironment(DDCEnvironment):
    """RDW vehicle scrappage environment.

    A vehicle owner observes the car's age and defect severity from the
    annual APK inspection and decides whether to keep operating or scrap
    the vehicle. Scrapping incurs a replacement cost but resets the state
    to a new car with no defects. Continued operation incurs costs that
    grow with age and defect severity.

    State space:
        Joint (age_bin, defect_level) encoded as flat index.
        age_bin in {0, 1, ..., num_age_bins - 1} represents vehicle age
        in years. defect_level in {0, 1, 2} represents APK inspection
        outcome: 0 = pass (no defects), 1 = minor defects, 2 = major
        defects (rejection). Total states = num_age_bins * num_defect_levels.

    Action space:
        0 = Keep: Continue operating, pay repair costs if defects found
        1 = Scrap: Scrap/export vehicle, pay replacement cost for new car

    Utility specification:
        U(s, keep) = -age_cost * age - minor_defect_cost * I(minor)
                     - major_defect_cost * I(major) + epsilon_keep
        U(s, scrap) = -replacement_cost + epsilon_scrap

    where epsilon are i.i.d. Type I Extreme Value shocks.

    Transition dynamics:
        If keep: age increments by 1 (capped at max). Defect level
            transitions stochastically with age-dependent probabilities.
            Older cars are more likely to develop defects.
        If scrap: state resets to (age=0, defect=0), representing a
            new replacement vehicle entering the fleet.

    Example:
        >>> env = RDWScrapageEnvironment()
        >>> obs, info = env.reset()
        >>> print(f"Initial state: age={obs // 3}, defect={obs % 3}")
    """

    KEEP = 0
    SCRAP = 1

    def __init__(
        self,
        age_cost: float = 0.15,
        minor_defect_cost: float = 0.5,
        major_defect_cost: float = 1.5,
        replacement_cost: float = 3.0,
        num_age_bins: int = 25,
        num_defect_levels: int = 3,
        defect_age_sensitivity: float = 0.02,
        discount_factor: float = 0.95,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the RDW vehicle scrappage environment.

        Args:
            age_cost: Per-year cost of operating an aging vehicle.
            minor_defect_cost: Cost penalty for minor APK defects.
            major_defect_cost: Cost penalty for major APK defects.
            replacement_cost: Fixed cost of scrapping and replacing.
            num_age_bins: Number of age discretization bins (years).
            num_defect_levels: Number of defect severity levels.
            defect_age_sensitivity: How much age increases the probability
                of transitioning to a higher defect level.
            discount_factor: Annual time discount factor beta.
            scale_parameter: Logit scale parameter sigma.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        self._age_cost = age_cost
        self._minor_defect_cost = minor_defect_cost
        self._major_defect_cost = major_defect_cost
        self._replacement_cost = replacement_cost
        self._num_age_bins = num_age_bins
        self._num_defect_levels = num_defect_levels
        self._defect_age_sensitivity = defect_age_sensitivity

        n_states = num_age_bins * num_defect_levels
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(2)

        self._transition_matrices = self._build_transition_matrices()
        self._feature_matrix = self._build_feature_matrix()

    @property
    def num_states(self) -> int:
        return self._num_age_bins * self._num_defect_levels

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix

    @property
    def true_parameters(self) -> dict[str, float]:
        return {
            "age_cost": self._age_cost,
            "minor_defect_cost": self._minor_defect_cost,
            "major_defect_cost": self._major_defect_cost,
            "replacement_cost": self._replacement_cost,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["age_cost", "minor_defect_cost", "major_defect_cost", "replacement_cost"]

    def _defect_transition_probs(self, age: int) -> np.ndarray:
        """Compute defect level transition probabilities given age.

        Returns a (num_defect_levels, num_defect_levels) matrix where
        entry [d, d'] = P(defect' = d' | defect = d, age).

        Older vehicles are more likely to stay at or move to higher
        defect levels. The sensitivity parameter controls how quickly
        defect risk increases with age.
        """
        n_d = self._num_defect_levels
        s = self._defect_age_sensitivity
        trans = np.zeros((n_d, n_d), dtype=np.float64)

        for d in range(n_d):
            # Probability of staying at current defect level decreases with age
            p_stay = max(0.4, 0.85 - s * age)
            # Probability of improving (going down) is small and only from d > 0
            p_improve = 0.05 if d > 0 else 0.0
            # Remaining probability goes to worsening
            p_worsen = 1.0 - p_stay - p_improve

            if d == 0:
                # From no defects: stay clean or develop defects
                trans[d, 0] = p_stay
                if n_d > 1:
                    trans[d, 1] = p_worsen * 0.7
                if n_d > 2:
                    trans[d, 2] = p_worsen * 0.3
            elif d == n_d - 1:
                # From worst defect level: can improve or stay
                trans[d, d] = p_stay + p_worsen  # can't get worse
                if d > 0:
                    trans[d, d - 1] = p_improve
            else:
                # Middle defect levels
                trans[d, d] = p_stay
                trans[d, d - 1] = p_improve
                remaining = p_worsen
                # Split worsening between +1 and +2 if possible
                if d + 2 < n_d:
                    trans[d, d + 1] = remaining * 0.7
                    trans[d, d + 2] = remaining * 0.3
                else:
                    trans[d, d + 1] = remaining

        # Normalize rows to ensure valid probability distributions
        for d in range(n_d):
            row_sum = trans[d].sum()
            if row_sum > 0:
                trans[d] /= row_sum

        return trans

    def _build_transition_matrices(self) -> jnp.ndarray:
        """Build transition probability matrices P(s'|s,a).

        Returns:
            Tensor of shape (2, num_states, num_states)
            - transitions[0, s, s'] = P(s' | s, keep)
            - transitions[1, s, s'] = P(s' | s, scrap)
        """
        n = self.num_states
        n_a = self._num_age_bins
        n_d = self._num_defect_levels
        transitions = np.zeros((2, n, n), dtype=np.float64)

        # Keep action: age +1 (deterministic), defect transitions (stochastic)
        for age in range(n_a):
            next_age = min(age + 1, n_a - 1)
            defect_trans = self._defect_transition_probs(age)

            for d in range(n_d):
                s = components_to_state(age, d, n_d)
                for d_next in range(n_d):
                    s_next = components_to_state(next_age, d_next, n_d)
                    transitions[self.KEEP, s, s_next] += defect_trans[d, d_next]

        # Scrap action: reset to (age=0, defect=0)
        s_new = components_to_state(0, 0, n_d)
        transitions[self.SCRAP, :, s_new] = 1.0

        return jnp.array(transitions)

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for utility computation.

        Features are structured so that U(s, a) = theta dot phi(s, a)
        where theta = [age_cost, minor_defect_cost, major_defect_cost,
        replacement_cost].

        Returns:
            Tensor of shape (num_states, num_actions, num_features)
        """
        n = self.num_states
        n_d = self._num_defect_levels
        features = np.zeros((n, 2, 4), dtype=np.float32)

        for s in range(n):
            age, defect = state_to_components(s, n_d)

            # Keep action (a=0): -age_cost * age - defect costs
            features[s, self.KEEP, 0] = -float(age)
            features[s, self.KEEP, 1] = -1.0 if defect == 1 else 0.0
            features[s, self.KEEP, 2] = -1.0 if defect == 2 else 0.0
            features[s, self.KEEP, 3] = 0.0

            # Scrap action (a=1): -replacement_cost
            features[s, self.SCRAP, 0] = 0.0
            features[s, self.SCRAP, 1] = 0.0
            features[s, self.SCRAP, 2] = 0.0
            features[s, self.SCRAP, 3] = -1.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        """Return initial state distribution (start as new car, no defects)."""
        dist = np.zeros(self.num_states)
        dist[components_to_state(0, 0, self._num_defect_levels)] = 1.0
        return dist

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility for state-action pair."""
        age, defect = state_to_components(state, self._num_defect_levels)

        if action == self.KEEP:
            u = -self._age_cost * age
            if defect == 1:
                u -= self._minor_defect_cost
            elif defect == 2:
                u -= self._major_defect_cost
            return u
        else:
            return -self._replacement_cost

    def _sample_next_state(self, state: int, action: int) -> int:
        """Sample next state given current state and action."""
        n_d = self._num_defect_levels

        if action == self.SCRAP:
            return components_to_state(0, 0, n_d)

        age, defect = state_to_components(state, n_d)
        next_age = min(age + 1, self._num_age_bins - 1)

        # Sample defect transition
        defect_trans = self._defect_transition_probs(age)
        next_defect = self._np_random.choice(n_d, p=defect_trans[defect])

        return components_to_state(next_age, int(next_defect), n_d)

    def _state_to_record(self, state: int, action: int) -> dict[str, Any]:
        """Convert state-action pair to human-readable record fields."""
        age, defect = state_to_components(state, self._num_defect_levels)
        return {
            "age_bin": age,
            "defect_level": defect,
            "scrapped": int(action == self.SCRAP),
        }

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        n_d = self._num_defect_levels
        return f"""RDW Vehicle Scrappage Environment
====================================
States: {self.num_states} ({self._num_age_bins} age bins x {n_d} defect levels)
Actions: Keep (0), Scrap (1)

True Parameters:
  Age cost:           {self._age_cost}
  Minor defect cost:  {self._minor_defect_cost}
  Major defect cost:  {self._major_defect_cost}
  Replacement cost:   {self._replacement_cost}

Structural Parameters:
  Discount factor (beta):         {self._discount_factor}
  Scale parameter (sigma):        {self._scale_parameter}
  Defect age sensitivity:         {self._defect_age_sensitivity}
"""

    @classmethod
    def info(cls) -> dict:
        """Return metadata about this environment."""
        return {
            "name": "RDW Vehicle Scrappage",
            "description": (
                "Vehicle scrappage decision based on Dutch RDW inspection data. "
                "Owner observes car age and APK defect severity, decides to keep "
                "or scrap."
            ),
            "source": "RDW Open Data (opendata.rdw.nl)",
            "n_states": 75,
            "n_actions": 2,
            "parameters": [
                "age_cost",
                "minor_defect_cost",
                "major_defect_cost",
                "replacement_cost",
            ],
            "reference": (
                "RDW Open Data, Dutch APK inspection system. "
                "Rust (1987) structural framework applied to vehicle scrappage."
            ),
        }
