"""MCEIRLNeural: Neural Maximum Causal Entropy IRL.

Supports two reward parameterizations:
- ``reward_type="state_action"`` (default): learns R(s,a) via a neural
  network that takes [state_features, action_onehot] as input.  This is
  more general and correctly handles environments with action-dependent
  rewards (e.g., gridworlds where moving has a cost but staying is free).
- ``reward_type="state"``: learns R(s) only, broadcasting the same reward
  to all actions (original behaviour).

Training loop (MCE-IRL objective, Ziebart 2010):
    for epoch in range(max_epochs):
        1. Compute reward matrix R(s,a) for all (state, action) pairs
        2. Solve soft Bellman with this reward (transitions required)
        3. Compute state visitation frequencies via forward pass
        4. Loss = -E_expert[R] + E_policy[R]  (feature matching)
        5. Backprop through reward network

After training, implied rewards are projected onto features via
least-squares to extract interpretable theta (same as NeuralGLADIUS).

Reference:
    Ziebart, B. D. (2010). Modeling purposeful adaptive behavior with the
        principle of maximum causal entropy. PhD thesis, CMU.
    Wulfmeier, M., Ondruska, P., & Posner, I. (2015). Maximum entropy
        deep inverse reinforcement learning. arXiv:1507.04888.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.occupancy import compute_state_action_visitation
from econirl.core.reward_spec import RewardSpec
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimators.neural_base import NeuralEstimatorMixin


# ---------------------------------------------------------------------------
# Internal network modules (Equinox)
# ---------------------------------------------------------------------------


class _StateRewardNetwork(eqx.Module):
    """R(s) reward network.

    Input: state features of shape (state_dim,).
    Output: scalar reward.
    """

    layers: list
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        keys = jax.random.split(key, num_layers + 1)
        layers = []
        in_dim = state_dim
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = eqx.nn.Linear(in_dim, 1, key=keys[-1])

    def __call__(self, state_feat: jax.Array) -> jax.Array:
        """Compute R(s) for a single state.

        Parameters
        ----------
        state_feat : jax.Array
            State features of shape (state_dim,).

        Returns
        -------
        jax.Array
            Scalar reward.
        """
        x = state_feat
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return self.output_layer(x).squeeze(-1)


class _StateActionRewardNetwork(eqx.Module):
    """R(s,a) reward network.

    Input: concatenation of state features (state_dim,) and action
    one-hot encoding (n_actions,).
    Output: scalar reward.
    """

    layers: list
    output_layer: eqx.nn.Linear
    _n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self._n_actions = n_actions
        input_dim = state_dim + n_actions
        keys = jax.random.split(key, num_layers + 1)
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(eqx.nn.Linear(in_dim, hidden_dim, key=keys[i]))
            in_dim = hidden_dim
        self.layers = layers
        self.output_layer = eqx.nn.Linear(in_dim, 1, key=keys[-1])

    def __call__(
        self, state_feat: jax.Array, action_onehot: jax.Array
    ) -> jax.Array:
        """Compute R(s,a) for a single (state, action) pair.

        Parameters
        ----------
        state_feat : jax.Array
            State features of shape (state_dim,).
        action_onehot : jax.Array
            One-hot action encoding of shape (n_actions,).

        Returns
        -------
        jax.Array
            Scalar reward.
        """
        x = jnp.concatenate([state_feat, action_onehot])
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
        return self.output_layer(x).squeeze(-1)

    def all_actions(self, state_feat: jax.Array) -> jax.Array:
        """Compute R(s,a) for all actions at every state.

        Parameters
        ----------
        state_feat : jax.Array
            State features of shape (S, state_dim).

        Returns
        -------
        jax.Array
            Reward matrix of shape (S, A).
        """
        S = state_feat.shape[0]
        A = self._n_actions
        eye = jnp.eye(A)
        # Expand state features: (S, state_dim) -> (S*A, state_dim)
        sf_expanded = jnp.repeat(state_feat, A, axis=0)
        # Tile action one-hots: (A, A) -> (S*A, A)
        act_expanded = jnp.tile(eye, (S, 1))
        # Apply network to all (state, action) pairs in one vmap call
        rewards = jax.vmap(self)(sf_expanded, act_expanded)
        return rewards.reshape(S, A)


# ---------------------------------------------------------------------------
# MCEIRLNeural estimator
# ---------------------------------------------------------------------------


class MCEIRLNeural(NeuralEstimatorMixin):
    """Neural Maximum Causal Entropy IRL.

    Learns a neural reward function using the MCE-IRL objective:
    maximize E_expert[R] - log Z(R)

    where Z(R) is the partition function (soft value at initial state).

    Supports two reward types:
    - ``reward_type="state_action"`` (default): R(s,a) via a network that
      takes [state_features, action_onehot].  This is more general and
      correctly handles action-dependent rewards.
    - ``reward_type="state"``: R(s) broadcast to all actions (original).

    For v1, transitions must be available so that exact soft value iteration
    and state visitation frequencies can be computed.

    Parameters
    ----------
    n_states : int, optional
        Number of discrete states.  Inferred from data if None.
    n_actions : int, optional
        Number of discrete actions.  Inferred from data if None.
    discount : float, default=0.95
        Time discount factor beta.
    reward_type : str, default="state_action"
        Type of reward function: ``"state_action"`` for R(s,a) or
        ``"state"`` for R(s) broadcast to all actions.
    reward_hidden_dim : int, default=64
        Hidden dimension for the reward MLP.
    reward_num_layers : int, default=2
        Number of hidden layers in the reward MLP.
    max_epochs : int, default=200
        Maximum number of training epochs.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    inner_solver : str, default="hybrid"
        Solver for soft value iteration: "hybrid" or "value".
    inner_tol : float, default=1e-8
        Convergence tolerance for inner solver.
    inner_max_iter : int, default=5000
        Maximum iterations for inner solver.
    state_encoder : callable, optional
        Function mapping state indices (int array) to feature vectors.
        Receives shape (B,) and should return shape (B, state_dim).
        If None, a default normalizing encoder is created.
    state_dim : int, optional
        Dimension of state features.  Required if state_encoder is provided.
    feature_names : list of str, optional
        Names for features when projecting rewards onto linear features.
    anchor_action : int, optional
        Action whose reward is fixed to zero. This is useful for identified
        action-dependent IRL designs with a normalized outside/exit action.
    absorbing_state : int, optional
        State whose reward row is fixed to zero.
    seed : int, default=0
        Random seed for network initialization.
    verbose : bool, default=False
        Whether to print progress during training.

    Attributes
    ----------
    params_ : dict or None
        Projected structural parameters after fitting.  None if no
        features were provided for projection.
    se_ : dict or None
        Pseudo standard errors from the projection regression.
    pvalues_ : dict or None
        P-values from Wald t-test on pseudo SEs.
    coef_ : numpy.ndarray or None
        Coefficient array (same values as params_ in array form).
    policy_ : numpy.ndarray or None
        Estimated choice probabilities P(a|s) of shape (n_states, n_actions).
    value_ : numpy.ndarray or None
        Estimated value function V(s) of shape (n_states,).
    reward_ : numpy.ndarray or None
        Neural reward.  Shape (n_states,) for ``reward_type="state"``
        or (n_states, n_actions) for ``reward_type="state_action"``.
    projection_r2_ : float or None
        R-squared of the feature projection.
    converged_ : bool or None
        Whether training converged.
    n_epochs_ : int or None
        Number of training epochs completed.

    Examples
    --------
    >>> from econirl.estimators import MCEIRLNeural
    >>> import numpy as np
    >>>
    >>> # R(s,a) -- default, more general
    >>> model = MCEIRLNeural(n_states=25, n_actions=4, discount=0.95)
    >>> model.fit(data=df, state="state", action="action", id="agent_id",
    ...           transitions=T)
    >>> print(model.reward_.shape)  # (25, 4)
    >>> print(model.policy_.shape)  # (25, 4)
    >>>
    >>> # R(s) -- state-only, backward compatible
    >>> model = MCEIRLNeural(n_states=25, n_actions=4, reward_type="state")
    >>> model.fit(...)
    >>> print(model.reward_.shape)  # (25,)
    """

    def __init__(
        self,
        n_states: int | None = None,
        n_actions: int | None = None,
        discount: float = 0.95,
        # Reward type
        reward_type: str = "state_action",
        # Network
        reward_hidden_dim: int = 64,
        reward_num_layers: int = 2,
        # Training
        max_epochs: int = 200,
        lr: float = 1e-3,
        # Inner solver
        inner_solver: str = "hybrid",
        inner_tol: float = 1e-8,
        inner_max_iter: int = 5000,
        # Encoders
        state_encoder: Callable | None = None,
        state_dim: int | None = None,
        # Projection
        feature_names: list[str] | None = None,
        anchor_action: int | None = None,
        absorbing_state: int | None = None,
        seed: int = 0,
        verbose: bool = False,
    ):
        if reward_type not in ("state", "state_action"):
            raise ValueError(
                f"reward_type must be 'state' or 'state_action', "
                f"got '{reward_type}'"
            )
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount
        self.reward_type = reward_type
        self.reward_hidden_dim = reward_hidden_dim
        self.reward_num_layers = reward_num_layers
        self.max_epochs = max_epochs
        self.lr = lr
        self.inner_solver = inner_solver
        self.inner_tol = inner_tol
        self.inner_max_iter = inner_max_iter
        self.state_encoder = state_encoder
        self.state_dim = state_dim
        self.feature_names = feature_names
        self.anchor_action = anchor_action
        self.absorbing_state = absorbing_state
        self.seed = seed
        self.verbose = verbose

        # Fitted attributes (set after fit())
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.reward_: np.ndarray | None = None
        self.projection_r2_: float | None = None
        self.converged_: bool | None = None
        self.n_epochs_: int | None = None
        self.feature_difference_: float | None = None
        self.occupancy_moment_residual_: float | None = None

        # Internal state
        self._reward_net = None
        self._state_encoder: Callable | None = None
        self._state_dim: int | None = None
        self._n_states: int | None = None
        self._n_actions: int | None = None
        self._empirical_sa: jnp.ndarray | None = None
        self._initial_distribution: jnp.ndarray | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        features: RewardSpec | np.ndarray | None = None,
        transitions: np.ndarray | None = None,
        context: object = None,
    ) -> "MCEIRLNeural":
        """Fit the MCEIRLNeural estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with demonstrations.  When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
        state : str, optional
            Column name for the state variable (required for DataFrame).
        action : str, optional
            Column name for the action variable (required for DataFrame).
        id : str, optional
            Column name for the individual identifier (required for DataFrame).
        features : RewardSpec or numpy.ndarray, optional
            Feature specification for parameter projection.  If provided,
            the neural reward is projected onto these features to extract
            interpretable theta.
        transitions : numpy.ndarray
            Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
            Required for v1 (exact soft value iteration).
        context : ignored
            Accepted for API compatibility but not used.

        Returns
        -------
        self : MCEIRLNeural
            Returns self for method chaining.
        """
        if transitions is None:
            raise ValueError(
                "MCEIRLNeural v1 requires transitions. "
                "Pass transitions as (n_actions, n_states, n_states) array."
            )

        # --- Step 1: Extract arrays from data ---
        panel, all_states, all_actions, all_next = self._extract_data(
            data, state, action, id
        )

        n_states = self.n_states or int(all_states.max()) + 1
        n_actions = self.n_actions or int(all_actions.max()) + 1
        if self.anchor_action is not None and not 0 <= self.anchor_action < n_actions:
            raise ValueError(
                f"anchor_action must be in [0, {n_actions}), got {self.anchor_action}"
            )
        if self.absorbing_state is not None and not 0 <= self.absorbing_state < n_states:
            raise ValueError(
                f"absorbing_state must be in [0, {n_states}), got {self.absorbing_state}"
            )
        self._n_states = n_states
        self._n_actions = n_actions

        # Convert transitions to JAX
        transitions_jax = jnp.asarray(transitions, dtype=jnp.float32)

        # --- Step 2: Build encoder ---
        self._build_encoder(n_states)

        # --- Step 3: Compute empirical state-action occupancy ---
        empirical_sa = self._compute_empirical_occupancy(
            panel, n_states, n_actions, discount=self.discount
        )
        self._empirical_sa = empirical_sa
        self._initial_distribution = self._compute_initial_distribution(
            panel, n_states
        )

        # --- Step 4: Build reward network ---
        key = jax.random.PRNGKey(self.seed)
        if self.reward_type == "state_action":
            self._reward_net = _StateActionRewardNetwork(
                self._state_dim,
                n_actions,
                self.reward_hidden_dim,
                self.reward_num_layers,
                key=key,
            )
        else:
            self._reward_net = _StateRewardNetwork(
                self._state_dim,
                self.reward_hidden_dim,
                self.reward_num_layers,
                key=key,
            )

        # --- Step 5: Training loop ---
        self._train_mce(
            transitions_jax, empirical_sa, n_states, n_actions,
        )

        # --- Step 6: Extract policy, value, and reward ---
        self._extract_final(transitions_jax, n_states, n_actions)

        # --- Step 7: Feature projection ---
        if features is not None:
            self._project_onto_features(features, n_states, n_actions)
        else:
            self.params_ = None
            self.se_ = None
            self.pvalues_ = None
            self.projection_r2_ = None
            self.coef_ = None

        return self

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
    ) -> tuple[TrajectoryPanel, np.ndarray, np.ndarray, np.ndarray]:
        """Extract state/action/next_state arrays from input data."""
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
            all_states = np.asarray(panel.all_states, dtype=np.int64)
            all_actions = np.asarray(panel.all_actions, dtype=np.int64)
            all_next = np.asarray(panel.all_next_states, dtype=np.int64)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            panel = TrajectoryPanel.from_panel(data)
            all_states = np.asarray(panel.get_all_states(), dtype=np.int64)
            all_actions = np.asarray(panel.get_all_actions(), dtype=np.int64)
            all_next = np.asarray(panel.get_all_next_states(), dtype=np.int64)
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, "
                f"got {type(data)}"
            )

        return panel, all_states, all_actions, all_next

    # ------------------------------------------------------------------
    # Encoder setup
    # ------------------------------------------------------------------

    def _build_encoder(self, n_states: int) -> None:
        """Build default state encoder if not provided."""
        if self.state_encoder is not None:
            self._state_encoder = self.state_encoder
            self._state_dim = self.state_dim or 1
        else:
            max_s = max(n_states - 1, 1)

            def _default_encoder(s, _ms=max_s):
                s_float = jnp.asarray(s, dtype=jnp.float32)
                return (s_float / _ms).reshape(-1, 1)

            self._state_encoder = _default_encoder
            self._state_dim = 1

    # ------------------------------------------------------------------
    # Empirical occupancy
    # ------------------------------------------------------------------

    def _compute_empirical_occupancy(
        self,
        panel: TrajectoryPanel,
        n_states: int,
        n_actions: int,
        discount: float = 1.0,
    ) -> jnp.ndarray:
        """Compute empirical state-action occupancy from demonstrations.

        Returns
        -------
        jnp.ndarray
            State-action occupancy of shape (n_states, n_actions).
            Normalized to sum to 1.
        """
        sa_counts = np.zeros((n_states, n_actions), dtype=np.float32)
        total = 0.0
        for traj in panel.trajectories:
            states = np.asarray(traj.states, dtype=np.int64)
            actions = np.asarray(traj.actions, dtype=np.int64)
            if len(states) == 0:
                continue
            if discount == 1.0:
                weights = np.ones(len(states), dtype=np.float32)
            else:
                weights = np.power(float(discount), np.arange(len(states))).astype(
                    np.float32
                )
            flat_idx = states * n_actions + actions
            np.add.at(sa_counts.ravel(), flat_idx, weights)
            total += float(weights.sum())
        if total > 0:
            sa_counts = sa_counts / total
        return jnp.array(sa_counts)

    def _compute_initial_distribution(
        self,
        panel: TrajectoryPanel,
        n_states: int,
    ) -> jnp.ndarray:
        """Compute the empirical initial-state distribution."""
        counts = np.zeros(n_states, dtype=np.float32)
        for traj in panel.trajectories:
            if len(traj.states):
                counts[int(traj.states[0])] += 1.0
        total = counts.sum()
        if total > 0:
            counts = counts / total
        else:
            counts[:] = 1.0 / n_states
        return jnp.asarray(counts)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _compute_reward_matrix(
        self,
        reward_net,
        state_feat: jax.Array,
        n_states: int,
        n_actions: int,
    ) -> jax.Array:
        """Compute R(s,a) for all states and actions."""
        if self.reward_type == "state_action":
            rewards = reward_net.all_actions(state_feat)
        else:
            rewards_s = jax.vmap(reward_net)(state_feat)
            rewards = jnp.broadcast_to(
                rewards_s[:, None], (n_states, n_actions)
            )
        if self.absorbing_state is not None:
            rewards = rewards.at[int(self.absorbing_state), :].set(0.0)
        if self.anchor_action is not None:
            rewards = rewards.at[:, int(self.anchor_action)].set(0.0)
        return rewards

    def _train_mce(
        self,
        transitions: jnp.ndarray,
        empirical_sa: jnp.ndarray,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Run MCE-IRL training with neural reward network.

        Training loop:
        1. Forward: compute R(s,a) for all states and actions
        2. Solve soft Bellman: V, policy = soft_value_iteration(R, transitions)
        3. Compute state visitation: D(s) = forward_pass(policy, transitions)
        4. Expected occupancy: E_policy[sa] = D(s) * pi(a|s)
        5. Gradient: grad_R = policy_sa - empirical_sa
        6. Backprop through reward network via surrogate loss
        """
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=self.lr, weight_decay=1e-5),
        )
        opt_state = optimizer.init(eqx.filter(self._reward_net, eqx.is_array))

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        bellman = SoftBellmanOperator(problem=problem, transitions=transitions)

        best_loss = float("inf")
        best_net = self._reward_net
        patience_counter = 0
        patience = 100

        all_state_indices = jnp.arange(n_states)
        reward_net = self._reward_net

        from tqdm import tqdm
        pbar = tqdm(
            range(self.max_epochs),
            desc="MCE-IRL-NN",
            disable=not self.verbose,
            leave=True,
        )
        for epoch in pbar:
            # 1. Compute reward matrix R(s,a) (no gradient tracking needed here)
            state_feat = self._state_encoder(all_state_indices)
            reward_matrix = self._compute_reward_matrix(
                reward_net, state_feat, n_states, n_actions
            )

            # 2. Solve soft Bellman (no gradient through VI)
            if self.inner_solver == "hybrid":
                result = hybrid_iteration(
                    bellman,
                    reward_matrix,
                    tol=self.inner_tol,
                    max_iter=self.inner_max_iter,
                )
            else:
                result = value_iteration(
                    bellman,
                    reward_matrix,
                    tol=self.inner_tol,
                    max_iter=self.inner_max_iter,
                )
            policy = result.policy

            # 3. Compute state-action occupancy via discounted forward pass
            policy_sa = self._forward_pass(
                policy, transitions, n_states, self.discount
            )

            # 4. Feature matching gradient w.r.t. R(s,a)
            grad_r = policy_sa - empirical_sa

            # 5. Compute network parameter gradients via surrogate loss.
            #    The surrogate loss L = sum(R * grad_r) has gradient
            #    dL/d_params = sum(grad_r * dR/d_params), which is exactly
            #    the chain rule for the MCE-IRL objective.
            def surrogate_loss(net):
                R = self._compute_reward_matrix(
                    net, state_feat, n_states, n_actions
                )
                return jnp.sum(R * grad_r)

            loss_val_jax, grads = eqx.filter_value_and_grad(surrogate_loss)(
                reward_net
            )

            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(reward_net, eqx.is_array)
            )
            reward_net = eqx.apply_updates(reward_net, updates)

            # Monitor feature matching residual
            loss_val = float(jnp.sum(grad_r ** 2))
            feature_diff = float(jnp.linalg.norm(empirical_sa - policy_sa))

            pbar.set_postfix({
                "loss": f"{loss_val:.4f}",
                "fdiff": f"{feature_diff:.4f}",
                "best": f"{best_loss:.4f}",
                "no_imp": patience_counter,
            })

            # Early stopping with best model checkpoint
            if loss_val < best_loss - 1e-5:
                best_loss = loss_val
                best_net = reward_net
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        self._reward_net = best_net
        self.converged_ = patience_counter >= patience or epoch == self.max_epochs - 1
        self.n_epochs_ = epoch + 1
        self.feature_difference_ = float(np.sqrt(best_loss))

    def _forward_pass(
        self,
        policy: jnp.ndarray,
        transitions: jnp.ndarray,
        n_states: int,
        discount: float,
    ) -> jnp.ndarray:
        """Compute normalized discounted state-action visitation.

        Parameters
        ----------
        policy : jnp.ndarray
            Policy pi(a|s), shape (n_states, n_actions).
        transitions : jnp.ndarray
            Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
        n_states : int
            Number of states.
        discount : float
            Discount factor.

        Returns
        -------
        jnp.ndarray
            State-action visitation frequencies, shape (n_states, n_actions).
        """
        problem = DDCProblem(
            num_states=n_states,
            num_actions=policy.shape[1],
            discount_factor=discount,
            scale_parameter=1.0,
        )
        return compute_state_action_visitation(
            policy,
            transitions,
            problem,
            self._initial_distribution,
        )

    # ------------------------------------------------------------------
    # Post-training extraction
    # ------------------------------------------------------------------

    def _extract_final(
        self,
        transitions: jnp.ndarray,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Extract policy, value function, and reward from trained network."""
        all_state_indices = jnp.arange(n_states)
        state_feat = self._state_encoder(all_state_indices)

        reward_matrix = self._compute_reward_matrix(
            self._reward_net, state_feat, n_states, n_actions
        )

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        bellman = SoftBellmanOperator(
            problem=problem, transitions=transitions
        )

        if self.inner_solver == "hybrid":
            result = hybrid_iteration(
                bellman,
                reward_matrix,
                tol=self.inner_tol,
                max_iter=self.inner_max_iter,
            )
        else:
            result = value_iteration(
                bellman,
                reward_matrix,
                tol=self.inner_tol,
                max_iter=self.inner_max_iter,
            )

        self.policy_ = np.asarray(result.policy)
        self.value_ = np.asarray(result.V)
        if self.reward_type == "state_action":
            self.reward_ = np.asarray(reward_matrix)
        else:
            rewards_s = jax.vmap(self._reward_net)(state_feat)
            self.reward_ = np.asarray(rewards_s)
        if self._empirical_sa is not None:
            policy_sa = self._forward_pass(
                result.policy,
                transitions,
                n_states,
                self.discount,
            )
            residual = self._empirical_sa - policy_sa
            self.feature_difference_ = float(jnp.linalg.norm(residual))
            self.occupancy_moment_residual_ = float(jnp.max(jnp.abs(residual)))

    def _project_onto_features(
        self,
        features: RewardSpec | np.ndarray,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Project neural rewards onto features for interpretable theta.

        For ``reward_type="state_action"``, R(s,a) is projected onto
        (S*A, K) features:
            theta = argmin ||Phi_flat @ theta - R_flat||^2

        For ``reward_type="state"``, R(s) is projected onto (S, K) state
        features (original behaviour).

        Parameters
        ----------
        features : RewardSpec or numpy.ndarray
            Feature specification.  RewardSpec provides (S, A, K) matrix.
            An array of shape (S, K) or (S, A, K) is also accepted.
        n_states : int
            Number of states.
        n_actions : int
            Number of actions.
        """
        # Extract feature matrix and names
        if isinstance(features, RewardSpec):
            feat_3d = np.asarray(features.feature_matrix)
            names = features.parameter_names
        else:
            feat_arr = np.asarray(features)
            if feat_arr.ndim == 3:
                feat_3d = feat_arr
            elif feat_arr.ndim == 2:
                # (S, K) -> broadcast to (S, A, K)
                feat_3d = np.broadcast_to(
                    feat_arr[:, None, :],
                    (feat_arr.shape[0], n_actions, feat_arr.shape[1]),
                ).copy()
            else:
                raise ValueError(
                    f"features must be 2D (S, K) or 3D (S, A, K), "
                    f"got {feat_arr.ndim}D"
                )
            names = self.feature_names or [
                f"f{i}" for i in range(feat_3d.shape[-1])
            ]

        rewards = self.reward_.astype(np.float32)

        if self.reward_type == "state_action":
            phi = feat_3d.reshape(-1, feat_3d.shape[-1]).astype(np.float32)
            r_flat = rewards.reshape(-1)
        else:
            phi = feat_3d[:, 0, :].astype(np.float32)
            r_flat = rewards

        theta, se, r2 = self._project_parameters(phi, r_flat)

        self.params_ = {n: float(v) for n, v in zip(names, theta)}
        self.se_ = {n: float(v) for n, v in zip(names, se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = r2
        self.coef_ = theta

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Reward matrix R(s,a) of shape (n_states, n_actions).

        For ``reward_type="state_action"``, ``self.reward_`` already has
        shape (n_states, n_actions) and is returned directly.  For
        ``reward_type="state"``, the state-only reward is broadcast to all
        actions.
        """
        if self.reward_ is None:
            return None
        if self.reward_.ndim == 2:
            return self.reward_
        # State-only reward: broadcast to all actions
        n_actions = self._n_actions or self.n_actions
        return np.tile(self.reward_[:, np.newaxis], (1, n_actions))

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        """Predict choice probabilities for given states.

        Parameters
        ----------
        states : numpy.ndarray
            Array of state indices.

        Returns
        -------
        numpy.ndarray
            Choice probabilities of shape (len(states), n_actions).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states = np.asarray(states, dtype=np.int64)
        return self.policy_[states]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for projected parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.  Returns (1 - alpha) confidence intervals.

        Returns
        -------
        dict
            ``{param_name: (lower, upper)}`` confidence intervals.

        Raises
        ------
        RuntimeError
            If no projected parameters are available.
        """
        if self.params_ is None or self.se_ is None:
            raise RuntimeError(
                "No projected parameters available. "
                "Call fit() with features= to extract structural parameters."
            )
        z = scipy_norm.ppf(1 - alpha / 2)
        intervals: dict[str, tuple[float, float]] = {}
        for name in self.params_:
            est = self.params_[name]
            se = self.se_[name]
            if np.isfinite(se):
                intervals[name] = (est - z * se, est + z * se)
            else:
                intervals[name] = (float("nan"), float("nan"))
        return intervals

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a formatted summary of estimation results.

        Returns
        -------
        str
            Human-readable summary including neural reward info,
            parameter estimates, and projection R-squared.
        """
        if self.policy_ is None:
            return "MCEIRLNeural: Not fitted yet. Call fit() first."

        return self._format_neural_summary(
            method_name="MCEIRLNeural (Deep MCE-IRL)",
            params=self.params_,
            se=self.se_,
            pvalues=self.pvalues_,
            projection_r2=self.projection_r2_,
            n_observations=self._n_states,
            n_epochs=self.n_epochs_,
            converged=self.converged_,
            discount=self.discount,
            extra_lines=[
                f"Reward type: {self.reward_type}",
                f"Reward network: {self.reward_num_layers} layers x {self.reward_hidden_dim} hidden",
                f"Inner solver: {self.inner_solver}",
                f"Anchor action: {self.anchor_action}",
                f"Absorbing state: {self.absorbing_state}",
            ],
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"MCEIRLNeural(n_states={self._n_states or self.n_states}, "
            f"n_actions={self._n_actions or self.n_actions}, "
            f"discount={self.discount}, "
            f"fitted={fitted})"
        )
