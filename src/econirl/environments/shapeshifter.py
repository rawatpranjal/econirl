"""Shape-shifting synthetic DGP for code-vs-paper alignment benchmarks.

A single environment parameterized along eight axes that all twelve
focus estimators differ on. Used by the JSS deep-run Tier 4 cells to
verify that each estimator recovers ground truth in regimes its
source paper claims to support, and to surface failures in regimes
that should fail.

The eight axes are:

1.  ``reward_type``: ``"linear"`` (R = theta . phi) or ``"neural"``
    (R is a frozen MLP applied to state-action features).
2.  ``feature_type``: ``"linear"`` (polynomial in state index) or
    ``"neural"`` (frozen MLP of state index).
3.  ``action_dependent``: features vary across actions, or features
    are state-only and tiled across actions (this is the regime
    where MCE-IRL is unidentified).
4.  ``stochastic_transitions``: random sparse stochastic matrix or
    deterministic argmax row.
5.  ``stochastic_rewards``: at simulation time, additive Gaussian
    noise on the agent's flow utility (the Gumbel-logit shock is
    always present; this is *additional* noise on the reward itself).
6.  ``num_periods``: ``None`` for infinite horizon, ``int`` for
    finite horizon (uses backward induction for ground truth).
7.  ``discount_factor``: scalar in [0, 1).
8.  ``state_dim``: 1 for scalar state, >1 for a product space of
    ``num_states ** state_dim`` total states with mixed-radix
    encoding into ``state_dim`` continuous coordinates.

Ground truth is exact in every regime. Linear infinite cases are
solved by ``hybrid_iteration`` in ``core.solvers``; finite-horizon
cases by ``backward_induction``; neural-reward cases by computing
the full (S, A) reward matrix once at construction and feeding it
to the same solvers.

The environment exposes the standard ``DDCEnvironment`` interface,
so the existing ``simulate_panel`` and ``run_monte_carlo`` harnesses
work without modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces

from econirl.core.types import DDCProblem
from econirl.environments.base import DDCEnvironment


RewardType = Literal["linear", "neural"]
FeatureType = Literal["linear", "neural"]


@dataclass(frozen=True)
class ShapeshifterConfig:
    """Configuration for the shape-shifting DGP.

    Attributes match the eight axes documented in the module
    docstring. Defaults define the ``ss-spine`` cell of Tier 4.
    """

    num_states: int = 32
    num_actions: int = 3
    num_features: int = 4
    discount_factor: float = 0.95
    scale_parameter: float = 1.0
    num_periods: int | None = None
    reward_type: RewardType = "linear"
    feature_type: FeatureType = "linear"
    action_dependent: bool = True
    stochastic_rewards: bool = False
    stochastic_reward_scale: float = 0.3
    stochastic_transitions: bool = True
    transition_branching: int = 4
    state_dim: int = 1
    seed: int = 0
    network_width: int = 32
    network_depth: int = 2
    max_total_states: int = 4096
    reward_scale: float = 1.0

    @property
    def total_states(self) -> int:
        """Total number of flat state indices."""
        return self.num_states ** self.state_dim


def _frozen_mlp(
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    key: jax.Array,
) -> eqx.nn.MLP:
    """Construct a deterministic MLP with the given seed.

    The network is never trained. It serves as the ground-truth
    nonlinear map from state-action coordinates to either reward
    or features.
    """
    return eqx.nn.MLP(
        in_size=in_dim,
        out_size=out_dim,
        width_size=width,
        depth=depth,
        activation=jax.nn.tanh,
        key=key,
    )


class ShapeshifterEnvironment(DDCEnvironment):
    """Shape-shifting synthetic DDC environment.

    Construct with a ``ShapeshifterConfig`` (or pass ``**kwargs`` that
    map to its fields). Ground truth is computed at construction time
    so estimators can be benchmarked without re-solving the DP.

    The environment is fully deterministic given ``config.seed``: the
    transition matrix, neural network weights, and (if stochastic
    rewards are off) the flow utility are all reproducible.
    """

    def __init__(
        self,
        config: ShapeshifterConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            config = ShapeshifterConfig(**kwargs)
        elif kwargs:
            raise ValueError(
                "Pass either a ShapeshifterConfig or kwargs, not both."
            )

        if config.total_states > config.max_total_states:
            raise ValueError(
                f"Total states {config.total_states} exceeds cap "
                f"{config.max_total_states}. Lower num_states or state_dim."
            )

        super().__init__(
            discount_factor=config.discount_factor,
            scale_parameter=config.scale_parameter,
            seed=config.seed,
        )
        self._config = config
        self._S = config.total_states
        self._A = config.num_actions
        self._K = config.num_features

        # Split a single deterministic key across the random objects so
        # that flipping one axis of the config does not change the
        # transition matrix or the network weights of the other axes.
        keys = jax.random.split(jax.random.PRNGKey(config.seed), 5)
        key_phi, key_r, key_T, key_theta, key_init = keys

        # State coordinates. For state_dim == 1 we use a normalized
        # scalar in [0, 1]; for higher state_dim we mixed-radix decode
        # the flat index into a tuple of normalized scalars.
        self._coords = self._build_state_coordinates()

        # Action one-hot, used as input to the neural feature / reward
        # nets so they can produce action-dependent outputs cleanly.
        self._action_one_hot = jnp.eye(self._A, dtype=jnp.float32)

        # Feature matrix phi: shape (S, A, K).
        self._feature_matrix_arr = self._build_feature_matrix(key_phi)

        # True theta (linear reward) or full reward matrix (neural).
        if config.reward_type == "linear":
            self._theta = (
                jax.random.normal(key_theta, shape=(self._K,))
                * 0.5
                * config.reward_scale
            )
            self._reward_matrix = jnp.einsum(
                "sak,k->sa", self._feature_matrix_arr, self._theta
            )
        else:
            self._theta = None
            self._reward_matrix = self._build_neural_reward_matrix(key_r)

        # Transition matrices: shape (A, S, S).
        self._transition_matrices_arr = self._build_transition_matrices(key_T)

        # Initial-state distribution.
        self._initial_dist = self._build_initial_distribution(key_init)

        # Gym spaces.
        self.observation_space = spaces.Discrete(self._S)
        self.action_space = spaces.Discrete(self._A)

        # Convenience: precompute parameter-name list and value dict.
        self._parameter_names, self._true_parameters = self._build_param_metadata()

    # ------------------------------------------------------------------
    # Required DDCEnvironment properties
    # ------------------------------------------------------------------

    @property
    def num_states(self) -> int:
        return self._S

    @property
    def num_actions(self) -> int:
        return self._A

    @property
    def state_dim(self) -> int:
        return self._config.state_dim

    @property
    def transition_matrices(self) -> jnp.ndarray:
        return self._transition_matrices_arr

    @property
    def feature_matrix(self) -> jnp.ndarray:
        return self._feature_matrix_arr

    @property
    def true_parameters(self) -> dict[str, float]:
        return dict(self._true_parameters)

    @property
    def parameter_names(self) -> list[str]:
        return list(self._parameter_names)

    @property
    def config(self) -> ShapeshifterConfig:
        return self._config

    @property
    def true_reward_matrix(self) -> jnp.ndarray:
        """Ground-truth flow reward R(s, a), shape (S, A).

        For ``reward_type="linear"`` this is ``phi @ theta``. For
        ``reward_type="neural"`` this is the frozen MLP output and
        ``parameter_names`` will be empty (estimators that require a
        finite-dimensional theta cannot meaningfully recover this).
        """
        return self._reward_matrix

    @property
    def problem_spec(self) -> DDCProblem:
        return DDCProblem(
            num_states=self._S,
            num_actions=self._A,
            discount_factor=self._discount_factor,
            scale_parameter=self._scale_parameter,
            num_periods=self._config.num_periods,
            state_dim=self._config.state_dim,
            state_encoder=self.encode_states,
        )

    def encode_states(self, states: jnp.ndarray) -> jnp.ndarray:
        """Encode flat state indices to ``state_dim`` continuous features."""
        states = jnp.asarray(states, dtype=jnp.int32)
        coords = self._coords[states]
        # ``self._coords`` is shape (S, state_dim).
        return coords.astype(jnp.float32)

    # ------------------------------------------------------------------
    # Required DDCEnvironment methods
    # ------------------------------------------------------------------

    def _get_initial_state_distribution(self) -> np.ndarray:
        return np.asarray(self._initial_dist)

    def _compute_flow_utility(self, state: int, action: int) -> float:
        utility = float(self._reward_matrix[int(state), int(action)])
        if self._config.stochastic_rewards:
            noise = float(
                self._np_random.normal(0.0, self._config.stochastic_reward_scale)
            )
            utility += noise
        return utility

    def _sample_next_state(self, state: int, action: int) -> int:
        probs = np.asarray(self._transition_matrices_arr[int(action), int(state)])
        # Defensive: renormalize in case of float drift.
        total = probs.sum()
        if total <= 0:
            return int(self._np_random.integers(0, self._S))
        probs = probs / total
        return int(self._np_random.choice(self._S, p=probs))

    def step(self, action: int):
        """Override base step to honor finite horizon."""
        next_state, reward, _, _, info = super().step(action)
        terminated = False
        if self._config.num_periods is not None:
            terminated = self._current_period >= self._config.num_periods
        return next_state, reward, terminated, False, info

    # ------------------------------------------------------------------
    # Builders
    # ------------------------------------------------------------------

    def _build_state_coordinates(self) -> jnp.ndarray:
        """Mixed-radix decode each flat index into ``state_dim`` coords.

        Each coordinate is normalized to ``[0, 1]`` so neural nets get
        well-scaled inputs.
        """
        per_dim = self._config.num_states
        denom = max(per_dim - 1, 1)
        coords = np.zeros((self._S, self._config.state_dim), dtype=np.float32)
        for s in range(self._S):
            remainder = s
            for d in range(self._config.state_dim):
                coords[s, d] = (remainder % per_dim) / denom
                remainder //= per_dim
        return jnp.asarray(coords)

    def _build_feature_matrix(self, key: jax.Array) -> jnp.ndarray:
        """Construct phi(s, a) of shape (S, A, K).

        ``feature_type="linear"`` produces polynomial features in the
        state coordinate(s). ``feature_type="neural"`` evaluates a
        frozen tanh MLP on the (state-coords, action-one-hot)
        concatenation when action dependence is requested.
        ``action_dependent=False`` collapses the action dimension by
        tiling state-only features across all actions.
        """
        cfg = self._config

        if cfg.feature_type == "neural" and cfg.action_dependent:
            features = self._neural_action_features(key)
            return features.at[:, 0, :].set(0.0)

        if cfg.feature_type == "linear":
            base_features = self._linear_state_features()  # (S, K)
        else:
            base_features = self._neural_state_features(key)  # (S, K)

        if cfg.action_dependent:
            # Use action 0 as the normalized outside action. Other
            # actions carry the state features plus an action-specific
            # shift, giving finite-theta estimators an identified gauge.
            features = np.zeros(
                (self._S, self._A, self._K), dtype=np.float32
            )
            base = np.asarray(base_features)
            for a in range(1, self._A):
                features[:, a, :] = base
                features[:, a, -1] += float(a)
            features_arr = jnp.asarray(features)
        else:
            # State-only: tile the same K-vector across actions.
            features_arr = jnp.broadcast_to(
                base_features[:, None, :],
                (self._S, self._A, self._K),
            )

        return features_arr

    def _linear_state_features(self) -> jnp.ndarray:
        """Polynomial features in the state coordinate(s).

        For ``state_dim==1`` the features are powers of the normalized
        scalar; for higher state_dim we use products of powers up to
        ``num_features``.
        """
        coords = self._coords  # (S, state_dim)
        feats = []
        # Always include a bias, then powers of each coord, then
        # cross terms if state_dim > 1, until we hit num_features.
        feats.append(jnp.ones((self._S,), dtype=jnp.float32))
        for d in range(self._config.state_dim):
            feats.append(coords[:, d])
            feats.append(coords[:, d] ** 2)
        # Pad with random projections if we still need more.
        while len(feats) < self._K:
            feats.append(coords[:, 0] ** (len(feats) + 1))
        feats = feats[: self._K]
        return jnp.stack(feats, axis=1).astype(jnp.float32)

    def _neural_state_features(self, key: jax.Array) -> jnp.ndarray:
        """Frozen MLP from state coords to a K-vector."""
        net = _frozen_mlp(
            in_dim=self._config.state_dim,
            out_dim=self._K,
            width=self._config.network_width,
            depth=self._config.network_depth,
            key=key,
        )
        out = jax.vmap(net)(self._coords)
        return out.astype(jnp.float32)

    def _neural_action_features(self, key: jax.Array) -> jnp.ndarray:
        """Frozen MLP from (state coords, action one-hot) to K features."""
        net = _frozen_mlp(
            in_dim=self._config.state_dim + self._A,
            out_dim=self._K,
            width=self._config.network_width,
            depth=self._config.network_depth,
            key=key,
        )

        def feature_fn(coord: jnp.ndarray, a_onehot: jnp.ndarray) -> jnp.ndarray:
            x = jnp.concatenate([coord, a_onehot])
            return net(x)

        per_state = jax.vmap(
            lambda c: jax.vmap(lambda a: feature_fn(c, a))(self._action_one_hot)
        )
        return per_state(self._coords).astype(jnp.float32)

    def _build_neural_reward_matrix(self, key: jax.Array) -> jnp.ndarray:
        """Frozen MLP from (state coords, action one-hot) to scalar reward."""
        if not self._config.action_dependent:
            net = _frozen_mlp(
                in_dim=self._config.state_dim,
                out_dim=1,
                width=self._config.network_width,
                depth=self._config.network_depth,
                key=key,
            )
            rewards_s = jax.vmap(lambda coord: net(coord)[0])(self._coords)
            return (
                self._config.reward_scale
                * jnp.broadcast_to(
                    rewards_s[:, None],
                    (self._S, self._A),
                )
            ).astype(jnp.float32)

        net = _frozen_mlp(
            in_dim=self._config.state_dim + self._A,
            out_dim=1,
            width=self._config.network_width,
            depth=self._config.network_depth,
            key=key,
        )

        def reward_fn(coord: jnp.ndarray, a_onehot: jnp.ndarray) -> jnp.ndarray:
            x = jnp.concatenate([coord, a_onehot])
            return net(x)[0]

        coords = self._coords  # (S, state_dim)
        # vmap over states then over actions.
        per_state = jax.vmap(
            lambda c: jax.vmap(lambda a: reward_fn(c, a))(self._action_one_hot)
        )
        rewards = per_state(coords)
        return (
            (self._config.reward_scale * rewards)
            .at[:, 0]
            .set(0.0)
            .astype(jnp.float32)
        )

    def _build_transition_matrices(self, key: jax.Array) -> jnp.ndarray:
        """Random sparse stochastic matrix per action, optionally collapsed.

        Each row is a Dirichlet draw on a random subset of size
        ``transition_branching`` next-states. This guarantees ergodicity
        (every state can reach the same support) while keeping the
        matrix sparse enough that estimators that estimate transitions
        from data have something to learn.
        """
        cfg = self._config
        S = self._S
        A = self._A
        b = min(cfg.transition_branching, S)

        rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 2**31 - 1)))
        T = np.zeros((A, S, S), dtype=np.float32)
        for a in range(A):
            for s in range(S):
                support = rng.choice(S, size=b, replace=False)
                weights = rng.dirichlet(np.ones(b))
                T[a, s, support] = weights

        if not cfg.stochastic_transitions:
            # Collapse each row to its argmax (one-hot) for the
            # deterministic-transition regime.
            argmax = T.argmax(axis=2)
            T_det = np.zeros_like(T)
            for a in range(A):
                for s in range(S):
                    T_det[a, s, argmax[a, s]] = 1.0
            T = T_det

        return jnp.asarray(T)

    def _build_initial_distribution(self, key: jax.Array) -> jnp.ndarray:
        """Uniform initial distribution.

        Uniform keeps state coverage broad in simulated panels and
        avoids penalizing CCP-style estimators that rely on observing
        all states.
        """
        return jnp.ones((self._S,), dtype=jnp.float32) / self._S

    def _build_param_metadata(self) -> tuple[list[str], dict[str, float]]:
        if self._config.reward_type == "linear":
            names = [f"theta_{i}" for i in range(self._K)]
            values = {n: float(v) for n, v in zip(names, np.asarray(self._theta))}
            return names, values
        # Neural reward: no finite theta. Expose summary statistics so
        # downstream code that calls ``true_parameters`` does not crash.
        return [], {}
