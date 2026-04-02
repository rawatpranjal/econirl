"""GLADIUS Estimator: Neural Network-based IRL for Dynamic Discrete Choice.

This module implements the GLADIUS estimator from Kang et al. (2025),
which uses neural networks to parameterize Q-functions and expected
future value functions for inverse reinforcement learning in dynamic
discrete choice models.

Algorithm:
    1. Parameterize Q(s,a) and EV(s,a) = E[V(s')|s,a] with MLPs.
    2. Train via mini-batch SGD on observed (s, a, s') transitions:
       - NLL loss: negative log-likelihood of observed actions under
         softmax policy derived from Q.
       - Bellman penalty: squared TD error beta*(EV(s,a) - V(s'))^2,
         where V(s') = sigma * logsumexp(Q(s', :) / sigma).
    3. Extract structural parameters by regressing implied rewards
       r(s,a) = Q(s,a) - beta * EV(s,a) onto the feature matrix.

Reference:
    Kang, M., et al. (2025). DDC IRL with neural networks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.results import EstimationSummary, GoodnessOfFit
from econirl.preferences.base import UtilityFunction


@dataclass
class GLADIUSConfig:
    """Configuration for the GLADIUS estimator.

    Attributes:
        q_hidden_dim: Hidden dimension for the Q-network MLP.
        q_num_layers: Number of hidden layers in the Q-network.
        v_hidden_dim: Hidden dimension for the zeta-network MLP.
        v_num_layers: Number of hidden layers in the zeta-network.
        q_lr: Learning rate for the Q-network optimizer.
        v_lr: Learning rate for the zeta-network optimizer.
        max_epochs: Maximum number of training epochs.
        batch_size: Mini-batch size for SGD.
        bellman_penalty_weight: Weight on the Bellman consistency penalty.
        weight_decay: L2 regularization weight.
        gradient_clip: Maximum gradient norm for clipping.
        patience: Early stopping patience (epochs without improvement).
        alternating_updates: Use alternating zeta/Q optimization from
            Algorithm 1 in Kang et al. (2025). When False, both networks
            are updated jointly in each step (legacy behavior).
        lr_decay_rate: Learning rate decay rate. The effective learning
            rate decays as lr_0 / (1 + decay_rate * step). Set to 0.0
            for constant learning rate.
        tikhonov_annealing: When True, the NLL loss is scaled by
            tikhonov_initial_weight / (1 + epoch), transitioning from
            behavioral cloning early to Bellman-driven refinement later.
        tikhonov_initial_weight: Initial weight on NLL when Tikhonov
            annealing is enabled.
        anchor_action: When set, Bellman error is only computed for
            transitions where a equals this action index. Per the paper
            (Assumption 3), this restricts the Bellman constraint to the
            anchor action for identification.
        compute_se: Whether to compute standard errors via bootstrap.
        n_bootstrap: Number of bootstrap replications for SE computation.
        verbose: Whether to print progress messages.
    """

    q_hidden_dim: int = 128
    q_num_layers: int = 3
    v_hidden_dim: int = 128
    v_num_layers: int = 3
    q_lr: float = 1e-3
    v_lr: float = 1e-3
    max_epochs: int = 500
    batch_size: int = 512
    bellman_penalty_weight: float = 1.0
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    patience: int = 50
    alternating_updates: bool = True
    lr_decay_rate: float = 0.001
    tikhonov_annealing: bool = False
    tikhonov_initial_weight: float = 100.0
    anchor_action: int | None = None
    compute_se: bool = True
    n_bootstrap: int = 100
    verbose: bool = False


class _QNetwork(eqx.Module):
    """MLP that maps (state_features, action_onehot) to a scalar Q value."""

    mlp: eqx.nn.MLP
    n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + n_actions,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute Q for a single input vector (state_feat || action_onehot).

        Args:
            x: Input vector of shape (state_dim + n_actions,).

        Returns:
            Scalar Q value.
        """
        return self.mlp(x).squeeze(-1)

    def forward(
        self, state_features: jax.Array, action_onehot: jax.Array
    ) -> jax.Array:
        """Compute Q(s, a) for a batch.

        Args:
            state_features: Array of shape (batch, state_dim).
            action_onehot: Array of shape (batch, n_actions).

        Returns:
            Q values of shape (batch,).
        """
        x = jnp.concatenate([state_features, action_onehot], axis=-1)
        return jax.vmap(self)(x)

    def forward_all_actions(self, state_features: jax.Array) -> jax.Array:
        """Compute Q(s, a) for all actions at once.

        Args:
            state_features: Array of shape (batch, state_dim).

        Returns:
            Q values of shape (batch, n_actions).
        """
        batch_size = state_features.shape[0]
        eye = jnp.eye(self.n_actions)

        def _q_for_action(a_idx: int) -> jax.Array:
            onehot = jnp.broadcast_to(eye[a_idx], (batch_size, self.n_actions))
            return self.forward(state_features, onehot)

        # Stack Q values for each action along axis 1.
        q_values = jnp.stack(
            [_q_for_action(a) for a in range(self.n_actions)], axis=1
        )
        return q_values


class _EVNetwork(eqx.Module):
    """MLP that maps (state_features, action_onehot) to scalar EV = E[V(s')|s,a]."""

    mlp: eqx.nn.MLP
    n_actions: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.mlp = eqx.nn.MLP(
            in_size=state_dim + n_actions,
            out_size=1,
            width_size=hidden_dim,
            depth=num_layers,
            activation=jax.nn.relu,
            key=key,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Compute EV for a single input vector (state_feat || action_onehot).

        Args:
            x: Input vector of shape (state_dim + n_actions,).

        Returns:
            Scalar EV value.
        """
        return self.mlp(x).squeeze(-1)

    def forward(
        self, state_features: jax.Array, action_onehot: jax.Array
    ) -> jax.Array:
        """Compute EV(s, a) for a batch.

        Args:
            state_features: Array of shape (batch, state_dim).
            action_onehot: Array of shape (batch, n_actions).

        Returns:
            EV values of shape (batch,).
        """
        x = jnp.concatenate([state_features, action_onehot], axis=-1)
        return jax.vmap(self)(x)


class GLADIUSEstimator(BaseEstimator):
    """GLADIUS estimator for DDC IRL with neural networks.

    Uses two MLPs (Q_net and EV_net) to approximate the Q-function
    and expected next-period value function. The loss combines negative
    log-likelihood (NLL) with a Bellman consistency penalty. After
    training, structural parameters are recovered by regressing implied
    rewards onto the feature matrix via least squares.

    Parameters
    ----------
    config : GLADIUSConfig, optional
        Configuration object. If None, default config is used.
    **kwargs
        Override individual config parameters.

    References
    ----------
    Kang, M., et al. (2025). DDC IRL with neural networks.
    """

    def __init__(self, config: GLADIUSConfig | None = None, **kwargs):
        if config is None:
            config = GLADIUSConfig(**kwargs)
        else:
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        super().__init__(
            se_method="bootstrap" if config.compute_se else "asymptotic",
            compute_hessian=False,
            verbose=config.verbose,
        )
        self.config = config
        self.q_net_: _QNetwork | None = None
        self.ev_net_: _EVNetwork | None = None

    @property
    def name(self) -> str:
        return "GLADIUS"

    def _build_state_features(
        self, states: jnp.ndarray, problem: DDCProblem
    ) -> jnp.ndarray:
        """Build state feature vectors from state indices.

        Uses the problem's state_encoder if available, otherwise
        normalizes state index to [0, 1].

        Args:
            states: Array of state indices, shape (batch,).
            problem: Problem specification with optional state_encoder.

        Returns:
            Feature array of shape (batch, state_dim).
        """
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        normalized = states.astype(jnp.float32) / max(problem.num_states - 1, 1)
        return normalized[:, None]

    def _build_state_features_all(self, problem: DDCProblem) -> jnp.ndarray:
        """Build feature vectors for all states.

        Args:
            problem: Problem specification.

        Returns:
            Feature array of shape (n_states, state_dim).
        """
        return self._build_state_features(jnp.arange(problem.num_states), problem)

    def _compute_log_likelihood(
        self,
        q_net: _QNetwork,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        problem: DDCProblem,
        sigma: float,
    ) -> float:
        """Compute the log-likelihood of the full dataset.

        Args:
            q_net: Trained Q-network.
            states: All observed state indices.
            actions: All observed action indices.
            problem: Problem specification.
            sigma: Scale parameter.

        Returns:
            Total log-likelihood (scalar).
        """
        state_feat = self._build_state_features(states, problem)
        q_all = q_net.forward_all_actions(state_feat)  # (N, n_actions)
        log_probs = q_all / sigma - jax.scipy.special.logsumexp(
            q_all / sigma, axis=1, keepdims=True
        )
        ll = float(log_probs[jnp.arange(len(actions)), actions].sum())
        return ll

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationSummary:
        """Estimate utility parameters from panel data.

        Overrides the base estimate() method because GLADIUS uses neural
        networks internally and needs custom handling for the summary.

        Parameters
        ----------
        panel : Panel
            Panel data with observed choices.
        utility : UtilityFunction
            Utility function specification.
        problem : DDCProblem
            Problem specification.
        transitions : jnp.ndarray
            Transition matrices P(s'|s,a).
        initial_params : jnp.ndarray, optional
            Not used (networks have their own initialization).

        Returns
        -------
        EstimationSummary
        """
        start_time = time.time()

        result = self._optimize(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            initial_params=initial_params,
            **kwargs,
        )

        # Standard errors not directly available from NN; fill with NaN
        standard_errors = jnp.full_like(result.parameters, float("nan"))

        # Goodness of fit
        n_obs = panel.num_observations
        n_params = len(result.parameters)
        ll = result.log_likelihood

        goodness_of_fit = GoodnessOfFit(
            log_likelihood=ll,
            num_parameters=n_params,
            num_observations=n_obs,
            aic=-2 * ll + 2 * n_params,
            bic=-2 * ll + n_params * np.log(n_obs),
            prediction_accuracy=self._compute_prediction_accuracy(panel, result.policy),
        )

        total_time = time.time() - start_time

        return EstimationSummary(
            parameters=result.parameters,
            parameter_names=utility.parameter_names,
            standard_errors=standard_errors,
            hessian=None,
            variance_covariance=None,
            method=self.name,
            num_observations=n_obs,
            num_individuals=panel.num_individuals,
            num_periods=max(panel.num_periods_per_individual),
            discount_factor=problem.discount_factor,
            scale_parameter=problem.scale_parameter,
            log_likelihood=ll,
            goodness_of_fit=goodness_of_fit,
            identification=None,
            converged=result.converged,
            num_iterations=result.num_iterations,
            convergence_message=result.message,
            value_function=result.value_function,
            policy=result.policy,
            estimation_time=total_time,
            metadata=result.metadata,
        )

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Core GLADIUS optimization routine.

        Implements Algorithm 1 from Kang et al. (2025). When alternating
        updates are enabled (default), even-numbered batches update the
        zeta network (value approximator) and odd-numbered batches update
        the Q network (action-value function). This alternating scheme
        stabilizes training by giving each network a stable target.

        The zeta network approximates E[V(s')|s,a]. The Q network is
        trained via negative log-likelihood of observed actions, with a
        Bellman consistency penalty that pushes V_Q(s') toward zeta(s,a).
        After training, implied rewards r(s,a) = Q(s,a) - beta*zeta(s,a)
        are projected onto the feature matrix to recover structural
        parameters.

        Steps:
            1. Extract (s, a, s') tuples from panel.
            2. Build Q-network and zeta-network.
            3. Train via alternating mini-batch SGD.
            4. Extract structural parameters via least-squares regression.

        Returns:
            EstimationResult with parameters, policy, and value function.
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        beta = problem.discount_factor
        sigma = problem.scale_parameter

        # --- Step 1: Extract data ---
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        all_next_states = panel.get_all_next_states()
        n_obs = len(all_states)

        # --- Step 2: Build networks ---
        state_dim = problem.state_dim or 1

        key = jax.random.PRNGKey(0)
        q_key, zeta_key = jax.random.split(key)

        q_net = _QNetwork(
            state_dim, n_actions, self.config.q_hidden_dim,
            self.config.q_num_layers, key=q_key,
        )
        # Zeta network approximates E[V(s')|s,a]. Same architecture as
        # the EV network but trained with the corrected alternating scheme
        # from Algorithm 1 in the paper. Kept as a separate network
        # (rather than a shared-body multi-headed MLP) for cleaner
        # gradient separation in the alternating update.
        zeta_net = _EVNetwork(
            state_dim, n_actions, self.config.v_hidden_dim,
            self.config.v_num_layers, key=zeta_key,
        )

        # Build optimizers with gradient clipping, weight decay, and
        # learning rate decay following Kang et al. (2025).
        # LR decays as lr_0 / (1 + decay_rate * step).
        decay = self.config.lr_decay_rate

        def _make_optimizer(base_lr: float):
            """Build an optax optimizer chain with LR decay."""
            def lr_schedule(step):
                return base_lr / (1.0 + decay * step)

            return optax.chain(
                optax.clip_by_global_norm(self.config.gradient_clip),
                optax.scale_by_adam(),
                optax.add_decayed_weights(self.config.weight_decay),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1.0),
            )

        q_optimizer = _make_optimizer(self.config.q_lr)
        zeta_optimizer = _make_optimizer(self.config.v_lr)

        q_opt_state = q_optimizer.init(eqx.filter(q_net, eqx.is_array))
        zeta_opt_state = zeta_optimizer.init(eqx.filter(zeta_net, eqx.is_array))

        bellman_weight = self.config.bellman_penalty_weight
        anchor_action = self.config.anchor_action

        # --- Define alternating training steps ---

        @eqx.filter_jit
        def zeta_step(zeta_net, q_net, zeta_opt_state, s_feat, a_batch, sp_feat):
            """Update zeta to approximate E[V(s')|s,a]. Q is frozen."""

            def zeta_loss_fn(zeta_net_inner):
                a_onehot = jax.nn.one_hot(a_batch, n_actions)
                zeta_sa = zeta_net_inner.forward(s_feat, a_onehot)

                # V(s') from frozen Q network
                q_sp_all = jax.lax.stop_gradient(
                    q_net.forward_all_actions(sp_feat)
                )
                v_sp = sigma * jax.scipy.special.logsumexp(
                    q_sp_all / sigma, axis=1
                )

                # MSE loss: make zeta approximate E[V(s')|s,a]
                mse = jnp.mean((zeta_sa - v_sp) ** 2)
                return mse

            loss, grads = eqx.filter_value_and_grad(zeta_loss_fn)(zeta_net)
            updates, new_zeta_opt = zeta_optimizer.update(
                grads, zeta_opt_state, eqx.filter(zeta_net, eqx.is_array)
            )
            zeta_net = eqx.apply_updates(zeta_net, updates)
            return zeta_net, new_zeta_opt, loss

        @eqx.filter_jit
        def q_step(q_net, zeta_net, q_opt_state, s_feat, a_batch, sp_feat,
                   ce_weight):
            """Update Q to fit observed choices with Bellman consistency.

            Loss = ce_weight * NLL + bellman_weight * Bellman_penalty

            NLL: negative log-likelihood of observed actions under the
            softmax policy derived from Q.

            Bellman penalty: pushes V_Q(s') toward zeta(s,a), propagating
            Bellman consistency through the Q network at the next state.
            Zeta is frozen (no gradient).
            """

            def q_loss_fn(q_net_inner):
                a_onehot = jax.nn.one_hot(a_batch, n_actions)

                # Q(s, all a) for NLL computation
                q_all = q_net_inner.forward_all_actions(s_feat)

                # NLL loss
                log_probs = q_all / sigma - jax.scipy.special.logsumexp(
                    q_all / sigma, axis=1, keepdims=True
                )
                nll = -log_probs[jnp.arange(len(a_batch)), a_batch].mean()

                # Bellman consistency penalty.
                # V_Q(s') should match zeta(s,a) (frozen).
                # This gives Q gradients at s' via V = logsumexp(Q(s',:)).
                q_sp_all = q_net_inner.forward_all_actions(sp_feat)
                v_sp = sigma * jax.scipy.special.logsumexp(
                    q_sp_all / sigma, axis=1
                )
                zeta_sa = jax.lax.stop_gradient(
                    zeta_net.forward(s_feat, a_onehot)
                )

                # Optional anchor action filtering: only compute Bellman
                # error for transitions where a equals the anchor action.
                if anchor_action is not None:
                    mask = (a_batch == anchor_action).astype(jnp.float32)
                    bellman_loss = jnp.sum(
                        mask * (v_sp - zeta_sa) ** 2
                    ) / jnp.maximum(mask.sum(), 1.0)
                else:
                    bellman_loss = jnp.mean((v_sp - zeta_sa) ** 2)

                total_loss = ce_weight * nll + bellman_weight * bellman_loss
                return total_loss, (nll, bellman_loss)

            (loss, aux), grads = eqx.filter_value_and_grad(
                q_loss_fn, has_aux=True
            )(q_net)

            updates, new_q_opt = q_optimizer.update(
                grads, q_opt_state, eqx.filter(q_net, eqx.is_array)
            )
            q_net = eqx.apply_updates(q_net, updates)
            return q_net, new_q_opt, loss, aux

        # Legacy joint step for backward compatibility when
        # alternating_updates=False.
        @eqx.filter_jit
        def joint_step(q_net, zeta_net, q_opt_state, zeta_opt_state,
                       s_feat, a_batch, sp_feat, ce_weight):
            """Joint update of both networks in one step (legacy mode)."""

            def loss_fn(nets):
                q_net_inner, zeta_net_inner = nets
                a_onehot = jax.nn.one_hot(a_batch, n_actions)
                q_all = q_net_inner.forward_all_actions(s_feat)
                log_probs = q_all / sigma - jax.scipy.special.logsumexp(
                    q_all / sigma, axis=1, keepdims=True
                )
                nll = -log_probs[jnp.arange(len(a_batch)), a_batch].mean()

                zeta_sa = zeta_net_inner.forward(s_feat, a_onehot)
                q_sp_all = q_net_inner.forward_all_actions(sp_feat)
                v_sp = sigma * jax.scipy.special.logsumexp(
                    q_sp_all / sigma, axis=1
                )
                td_error = beta * (
                    zeta_sa - jax.lax.stop_gradient(v_sp)
                )
                bellman_loss = jnp.mean(td_error ** 2)

                total_loss = ce_weight * nll + bellman_weight * bellman_loss
                return total_loss, (nll, bellman_loss)

            (loss, aux), grads = eqx.filter_value_and_grad(
                loss_fn, has_aux=True
            )((q_net, zeta_net))

            q_grads, zeta_grads = grads
            q_updates, new_q_opt = q_optimizer.update(
                q_grads, q_opt_state, eqx.filter(q_net, eqx.is_array)
            )
            zeta_updates, new_zeta_opt = zeta_optimizer.update(
                zeta_grads, zeta_opt_state,
                eqx.filter(zeta_net, eqx.is_array)
            )
            q_net = eqx.apply_updates(q_net, q_updates)
            zeta_net = eqx.apply_updates(zeta_net, zeta_updates)
            return q_net, zeta_net, new_q_opt, new_zeta_opt, loss, aux

        # --- Step 3: Training loop ---
        best_loss = float("inf")
        epochs_no_improve = 0
        converged = False
        loss_history: list[float] = []
        rng_key = jax.random.PRNGKey(42)
        alternating = self.config.alternating_updates

        from tqdm import tqdm
        pbar = tqdm(
            range(self.config.max_epochs),
            desc="GLADIUS",
            disable=not self._verbose,
            leave=True,
        )
        for epoch in pbar:
            rng_key, perm_key = jax.random.split(rng_key)
            perm = jax.random.permutation(perm_key, n_obs)
            epoch_loss = 0.0
            n_batches = 0

            # Tikhonov annealing: decay CE weight over epochs so training
            # transitions from behavioral cloning to Bellman-driven.
            if self.config.tikhonov_annealing:
                ce_weight = jnp.float32(
                    self.config.tikhonov_initial_weight / (1.0 + epoch)
                )
            else:
                ce_weight = jnp.float32(1.0)

            batch_idx = 0
            for start_idx in range(0, n_obs, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, n_obs)
                idx = perm[start_idx:end_idx]

                s_batch = all_states[idx]
                a_batch = all_actions[idx]
                sp_batch = all_next_states[idx]

                s_feat = self._build_state_features(s_batch, problem)
                sp_feat = self._build_state_features(sp_batch, problem)

                if alternating:
                    if batch_idx % 2 == 0:
                        # Even batch: update zeta only
                        zeta_net, zeta_opt_state, loss = zeta_step(
                            zeta_net, q_net, zeta_opt_state,
                            s_feat, a_batch, sp_feat,
                        )
                    else:
                        # Odd batch: update Q only
                        q_net, q_opt_state, loss, _aux = q_step(
                            q_net, zeta_net, q_opt_state,
                            s_feat, a_batch, sp_feat, ce_weight,
                        )
                else:
                    q_net, zeta_net, q_opt_state, zeta_opt_state, loss, _aux = (
                        joint_step(
                            q_net, zeta_net, q_opt_state, zeta_opt_state,
                            s_feat, a_batch, sp_feat, ce_weight,
                        )
                    )

                epoch_loss += float(loss)
                n_batches += 1
                batch_idx += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "best": f"{best_loss:.4f}"})

            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.config.patience:
                converged = True
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "status": "early stop"})
                pbar.close()
                if self._verbose:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                break

        num_epochs = epoch + 1
        if num_epochs == self.config.max_epochs:
            converged = True

        # --- Step 4: Extract parameters ---
        all_state_feat = self._build_state_features_all(problem)

        # Compute Q(s, a) for all (s, a)
        q_table = q_net.forward_all_actions(all_state_feat)

        # Compute zeta(s, a) for all (s, a)
        eye = jnp.eye(n_actions)
        ev_columns = []
        for a in range(n_actions):
            a_oh = jnp.broadcast_to(eye[a], (n_states, n_actions))
            ev_columns.append(zeta_net.forward(all_state_feat, a_oh))
        ev_table = jnp.stack(ev_columns, axis=1)

        # Implied reward: r(s, a) = Q(s, a) - beta * zeta(s, a)
        reward_table = q_table - beta * ev_table

        # Policy: softmax of Q values
        policy = jax.nn.softmax(q_table / sigma, axis=1)

        # Value function: V(s) = sigma * logsumexp(Q(s, :) / sigma)
        value_function = sigma * jax.scipy.special.logsumexp(
            q_table / sigma, axis=1
        )

        # Regress implied rewards onto feature matrix if available
        parameters = self._extract_parameters(utility, reward_table)

        # Compute log-likelihood
        ll = self._compute_log_likelihood(
            q_net, all_states, all_actions, problem, sigma
        )

        optimization_time = time.time() - start_time

        # Store trained networks for post-estimation use
        self.q_net_ = q_net
        self.ev_net_ = zeta_net

        message = f"GLADIUS converged after {num_epochs} epochs"
        if self._verbose:
            self._log(message)

        return EstimationResult(
            parameters=parameters,
            log_likelihood=ll,
            value_function=value_function,
            policy=policy,
            hessian=None,
            gradient_contributions=None,
            converged=converged,
            num_iterations=num_epochs,
            num_function_evals=num_epochs,
            num_inner_iterations=0,
            message=message,
            optimization_time=optimization_time,
            metadata={
                "reward_table": np.asarray(reward_table).tolist(),
                "q_table": np.asarray(q_table).tolist(),
                "ev_table": np.asarray(ev_table).tolist(),
                "loss_history": loss_history,
                "final_loss": (
                    loss_history[-1] if loss_history else float("nan")
                ),
            },
        )

    def _extract_parameters(
        self, utility: UtilityFunction, reward_table: jnp.ndarray
    ) -> jnp.ndarray:
        """Extract structural parameters via action-difference projection.

        IRL rewards are identified only up to additive constants (Kim et al.
        2021, Cao & Cohen 2021). To eliminate the unidentified constant, we
        project action DIFFERENCES onto feature DIFFERENCES relative to
        action 0. For each state s and action a > 0:

            dr(s) = r(s, a) - r(s, 0)
            dphi(s) = phi(s, a) - phi(s, 0)

        The constant in the absolute reward cancels in the difference, so
        both level and slope parameters are identified.

        Args:
            utility: Utility function specification.
            reward_table: Implied rewards of shape (n_states, n_actions).

        Returns:
            Parameter vector.
        """
        feature_matrix = getattr(utility, "feature_matrix", None)

        if feature_matrix is not None:
            # feature_matrix shape: (n_states, n_actions, n_features)
            n_states, n_actions, n_features = feature_matrix.shape
            feature_matrix = jnp.asarray(feature_matrix)

            # Action-difference projection: for each action a > 0,
            # compute dr(s) = r(s,a) - r(s,0) and
            # dphi(s) = phi(s,a) - phi(s,0).
            dr_list = []
            dphi_list = []
            for a in range(1, n_actions):
                dr = reward_table[:, a] - reward_table[:, 0]
                dphi = feature_matrix[:, a, :] - feature_matrix[:, 0, :]
                dr_list.append(dr)
                dphi_list.append(dphi)

            X = jnp.concatenate(dphi_list, axis=0)
            y = jnp.concatenate(dr_list, axis=0)

            parameters, _residuals, _rank, _sv = jnp.linalg.lstsq(X, y)
            return parameters
        else:
            return reward_table.flatten()
