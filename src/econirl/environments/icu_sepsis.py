"""ICU sepsis treatment environment from real clinical data.

This module wraps the ICU-Sepsis benchmark MDP (Oberst and Sontag 2019,
Killian et al. 2024) as a DDCEnvironment for structural estimation. The
MDP was derived from MIMIC-III patient records by clustering patient
physiology into 716 discrete states and discretizing IV fluid and
vasopressor doses into a 5x5 action grid.

State space:
    716 states. States 0 through 712 are clinical states representing
    clusters of patient physiology. State 713 is death. State 714 is
    survival. State 715 is an absorbing terminal state that both death
    and survival transition into.

Action space:
    25 actions arranged as a 5x5 grid of IV fluid dose (0 to 4) and
    vasopressor dose (0 to 4). Action index = fluid_level * 5 + vaso_level.

Reward:
    The MDP assigns reward +1 upon entering the survival state (714) and
    reward 0 everywhere else. This is a terminal reward. Discount factor
    should be set to 1.0 or close to it since episodes terminate.

Transitions:
    Estimated from MIMIC-III data using the Komorowski et al. (2018)
    AI Clinician pipeline. The transition threshold was 20 (state-action
    pairs observed fewer than 20 times were redistributed).

Features:
    Four features per state-action pair for linear utility estimation.
    Feature 0 is the normalized SOFA score (illness severity). Feature 1
    is the normalized IV fluid dose. Feature 2 is the normalized
    vasopressor dose. Feature 3 is the absorbing state indicator.

    These features support a basic linear reward model. For richer models
    use the 47-dimensional cluster centers available via load_icu_sepsis_mdp().

Reference:
    Killian, T.W., Shan, J., Krishnamurthy, K., Joshi, P., Srinivasan,
    A., Lam, J., & Celi, L.A. (2024). "ICU-Sepsis: A Benchmark MDP
    Built from Real Medical Data." NeurIPS Workshop on Datasets and
    Benchmarks.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.environments.base import DDCEnvironment


# Default path to bundled NPZ data
_DATA_PATH = Path(__file__).parent.parent / "datasets" / "icu_sepsis_mdp.npz"

# Special state indices
DEATH_STATE = 713
SURVIVAL_STATE = 714
ABSORBING_STATE = 715
N_STATES = 716
N_ACTIONS = 25
N_FLUID_LEVELS = 5
N_VASO_LEVELS = 5


def action_to_doses(action: int) -> tuple[int, int]:
    """Convert flat action index to (fluid_level, vaso_level) pair."""
    return action // N_VASO_LEVELS, action % N_VASO_LEVELS


def doses_to_action(fluid_level: int, vaso_level: int) -> int:
    """Convert (fluid_level, vaso_level) pair to flat action index."""
    return fluid_level * N_VASO_LEVELS + vaso_level


class ICUSepsisEnvironment(DDCEnvironment):
    """ICU sepsis treatment MDP from real MIMIC-III clinical data.

    This environment wraps the ICU-Sepsis benchmark MDP for use with
    econirl estimators. The transition matrices come from real patient
    data, not simulation. The feature matrix encodes clinical severity
    (SOFA score) and treatment intensity (fluid and vasopressor doses)
    for linear utility estimation.

    Unlike synthetic environments, this MDP has no known "true parameters"
    since the data comes from real clinician behavior. The true_parameters
    property returns rough clinical priors that can serve as initialization
    points for IRL estimation.

    State space:
        716 states. States 0-712 are patient physiology clusters.
        State 713 is death. State 714 is survival (reward +1).
        State 715 is the absorbing terminal.

    Action space:
        25 actions = 5 IV fluid levels x 5 vasopressor dose levels.
        Action index = fluid_level * 5 + vaso_level.

    Example:
        >>> env = ICUSepsisEnvironment()
        >>> obs, info = env.reset()
        >>> print(f"Initial state: {obs}, SOFA: {env.sofa_scores[obs]:.1f}")
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        discount_factor: float = 0.99,
        scale_parameter: float = 1.0,
        seed: int | None = None,
    ):
        """Initialize the ICU sepsis environment.

        Args:
            data_path: Path to the NPZ file with MDP data. If None, uses
                the bundled data file.
            discount_factor: Time discount factor. Set close to 1.0 since
                episodes terminate upon death or survival.
            scale_parameter: Logit scale parameter sigma > 0.
            seed: Random seed for reproducibility.
        """
        super().__init__(
            discount_factor=discount_factor,
            scale_parameter=scale_parameter,
            seed=seed,
        )

        path = Path(data_path) if data_path is not None else _DATA_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"ICU-Sepsis MDP data not found at {path}. "
                "The bundled NPZ file should be at "
                "src/econirl/datasets/icu_sepsis_mdp.npz"
            )

        data = np.load(path)
        self._transitions_np = data["transitions"]  # (25, 716, 716)
        self._rewards = data["rewards"]  # (716,)
        self._initial_dist = data["initial_distribution"]  # (716,)
        self._expert_policy = data["expert_policy"]  # (716, 25)
        self._sofa_scores = data["sofa_scores"]  # (716,)

        self.observation_space = spaces.Discrete(N_STATES)
        self.action_space = spaces.Discrete(N_ACTIONS)

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
            "sofa_weight": -0.1,
            "fluid_weight": -0.02,
            "vaso_weight": -0.02,
            "absorbing_weight": 0.0,
        }

    @property
    def parameter_names(self) -> list[str]:
        return ["sofa_weight", "fluid_weight", "vaso_weight", "absorbing_weight"]

    @property
    def sofa_scores(self) -> np.ndarray:
        """SOFA scores per state (0 to 18 scale)."""
        return self._sofa_scores

    @property
    def expert_policy(self) -> np.ndarray:
        """Clinician behavior policy from MIMIC-III, shape (716, 25)."""
        return self._expert_policy

    @property
    def rewards(self) -> np.ndarray:
        """State reward vector, shape (716,). Only state 714 is nonzero."""
        return self._rewards

    def _build_feature_matrix(self) -> jnp.ndarray:
        """Build feature matrix for linear utility estimation.

        Four features per state-action pair:
        - sofa: Normalized SOFA score (0 to 1). Higher means sicker.
        - fluid: Normalized IV fluid dose level (0 to 1).
        - vaso: Normalized vasopressor dose level (0 to 1).
        - absorbing: Indicator for the absorbing terminal state.

        Returns:
            Tensor of shape (716, 25, 4).
        """
        features = np.zeros((N_STATES, N_ACTIONS, 4), dtype=np.float32)

        max_sofa = self._sofa_scores.max()
        if max_sofa > 0:
            norm_sofa = self._sofa_scores / max_sofa
        else:
            norm_sofa = self._sofa_scores

        for a in range(N_ACTIONS):
            fluid_level, vaso_level = action_to_doses(a)
            norm_fluid = fluid_level / (N_FLUID_LEVELS - 1)
            norm_vaso = vaso_level / (N_VASO_LEVELS - 1)

            for s in range(N_STATES):
                features[s, a, 0] = norm_sofa[s]
                features[s, a, 1] = norm_fluid
                features[s, a, 2] = norm_vaso

        features[ABSORBING_STATE, :, 3] = 1.0

        return jnp.array(features)

    def _get_initial_state_distribution(self) -> np.ndarray:
        return self._initial_dist.copy()

    def _compute_flow_utility(self, state: int, action: int) -> float:
        """Compute flow utility using the true (prior) parameters."""
        params = self.true_parameters
        features = np.array(self._feature_matrix[state, action])
        names = self.parameter_names
        return sum(params[n] * float(features[i]) for i, n in enumerate(names))

    def _sample_next_state(self, state: int, action: int) -> int:
        probs = self._transitions_np[action, state, :]
        return int(self._np_random.choice(N_STATES, p=probs))

    def describe(self) -> str:
        """Return a human-readable description of the environment."""
        return f"""ICU-Sepsis Environment (Komorowski et al. 2018)
{'=' * 50}
States: {N_STATES} (713 clinical + death + survival + absorbing)
Actions: {N_ACTIONS} (5 IV fluid levels x 5 vasopressor levels)
Death state: {DEATH_STATE}
Survival state: {SURVIVAL_STATE} (reward +1)
Absorbing state: {ABSORBING_STATE}

SOFA score range: {self._sofa_scores.min():.1f} to {self._sofa_scores.max():.1f}

Prior Parameters (not ground truth):
  SOFA weight:      {self.true_parameters['sofa_weight']}
  Fluid weight:     {self.true_parameters['fluid_weight']}
  Vaso weight:      {self.true_parameters['vaso_weight']}
  Absorbing weight: {self.true_parameters['absorbing_weight']}

Structural Parameters:
  Discount factor (beta): {self._discount_factor}
  Scale parameter (sigma): {self._scale_parameter}
"""
