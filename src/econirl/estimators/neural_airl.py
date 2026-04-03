"""NeuralAIRL: Context-aware Adversarial Inverse Reinforcement Learning.

Learns a disentangled reward r(s,a,ctx) and shaping potential h(s,ctx) via
adversarial training against a learned policy network, then extracts
structural parameters by projecting implied rewards onto features.

No transition matrix is needed. Supports context conditioning through
pluggable state and context encoders.

Reference:
    Fu, J., Luo, K., & Levine, S. (2018). Learning robust rewards with
    adversarial inverse reinforcement learning. ICLR.
"""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pandas as pd
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, TrajectoryPanel
from econirl.estimators.neural_base import NeuralEstimatorMixin

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _is_torch_tensor(values: object) -> bool:
    return torch is not None and isinstance(values, torch.Tensor)


def _to_numpy(values: object) -> np.ndarray:
    if _is_torch_tensor(values):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _to_jax_float(values: object) -> jax.Array:
    if _is_torch_tensor(values):
        return jnp.asarray(values.detach().cpu().numpy(), dtype=jnp.float32)
    return jnp.asarray(values, dtype=jnp.float32)


def _to_jax_int(values: object) -> jax.Array:
    if _is_torch_tensor(values):
        return jnp.asarray(values.detach().cpu().numpy(), dtype=jnp.int32)
    return jnp.asarray(values, dtype=jnp.int32)


def _return_like(values: jax.Array, *templates: object) -> object:
    if any(_is_torch_tensor(template) for template in templates):
        if torch is None:  # pragma: no cover
            raise RuntimeError("Torch is required for torch tensor outputs.")
        return torch.tensor(np.asarray(values).copy())
    return values


def _bce_with_logits(logits: jax.Array, targets: jax.Array) -> jax.Array:
    return jnp.maximum(logits, 0.0) - logits * targets + jax.nn.softplus(-jnp.abs(logits))


def _sample_actions(policy_probs: jax.Array, key: jax.Array) -> jax.Array:
    probs = jnp.clip(policy_probs, 1e-8, 1.0)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    keys = jr.split(key, probs.shape[0])
    return jax.vmap(lambda k, p: jr.categorical(k, jnp.log(p)))(keys, probs).astype(jnp.int32)


class _MLP(eqx.Module):
    layers: tuple[eqx.nn.Linear, ...]
    output_layer: eqx.nn.Linear

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        n_hidden = max(num_layers, 0)
        keys = jr.split(key, n_hidden + 1)
        layers: list[eqx.nn.Linear] = []
        current_dim = in_dim
        for idx in range(n_hidden):
            layers.append(eqx.nn.Linear(current_dim, hidden_dim, key=keys[idx]))
            current_dim = hidden_dim
        self.layers = tuple(layers)
        self.output_layer = eqx.nn.Linear(current_dim, out_dim, key=keys[-1])

    def _forward_single(self, x: jax.Array) -> jax.Array:
        h = x
        for layer in self.layers:
            h = jax.nn.relu(layer(h))
        return self.output_layer(h)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = jnp.asarray(x, dtype=jnp.float32)
        if x.ndim == 1:
            return self._forward_single(x)
        return jax.vmap(self._forward_single)(x)

    def eval(self) -> _MLP:
        return self


class _RewardNetwork(eqx.Module):
    n_actions: int = eqx.field(static=True)
    net: _MLP

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.net = _MLP(
            state_dim + context_dim + n_actions,
            1,
            hidden_dim,
            num_layers,
            key=key,
        )

    def __call__(
        self,
        state_feat: object,
        ctx_feat: object,
        action_onehot: object,
    ) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        ao = _to_jax_float(action_onehot)
        x = jnp.concatenate([sf, cf, ao], axis=-1)
        out = jnp.squeeze(self.net(x), axis=-1)
        return _return_like(out, state_feat, ctx_feat, action_onehot)

    def all_actions(
        self,
        state_feat: object,
        ctx_feat: object,
        n_actions: int,
    ) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        actions = jnp.eye(n_actions, dtype=jnp.float32)
        sf_exp = jnp.repeat(sf[:, None, :], n_actions, axis=1)
        cf_exp = jnp.repeat(cf[:, None, :], n_actions, axis=1)
        a_exp = jnp.repeat(actions[None, :, :], sf.shape[0], axis=0)
        x = jnp.concatenate([sf_exp, cf_exp, a_exp], axis=-1)
        out = jnp.squeeze(jax.vmap(self.net)(x), axis=-1)
        return _return_like(out, state_feat, ctx_feat)

    def eval(self) -> _RewardNetwork:
        return self


class _ShapingNetwork(eqx.Module):
    net: _MLP

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.net = _MLP(
            state_dim + context_dim,
            1,
            hidden_dim,
            num_layers,
            key=key,
        )

    def __call__(self, state_feat: object, ctx_feat: object) -> object:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        x = jnp.concatenate([sf, cf], axis=-1)
        out = jnp.squeeze(self.net(x), axis=-1)
        return _return_like(out, state_feat, ctx_feat)

    def eval(self) -> _ShapingNetwork:
        return self


class _PolicyNetwork(eqx.Module):
    n_actions: int = eqx.field(static=True)
    net: _MLP

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: jax.Array,
    ):
        self.n_actions = n_actions
        self.net = _MLP(
            state_dim + context_dim,
            n_actions,
            hidden_dim,
            num_layers,
            key=key,
        )

    def logits(self, state_feat: object, ctx_feat: object) -> jax.Array:
        sf = _to_jax_float(state_feat)
        cf = _to_jax_float(ctx_feat)
        x = jnp.concatenate([sf, cf], axis=-1)
        return jnp.asarray(self.net(x), dtype=jnp.float32)

    def __call__(self, state_feat: object, ctx_feat: object) -> object:
        logits = self.logits(state_feat, ctx_feat)
        probs = jax.nn.softmax(logits, axis=-1)
        return _return_like(probs, state_feat, ctx_feat)

    def log_prob(
        self,
        state_feat: object,
        ctx_feat: object,
        actions: object,
    ) -> object:
        logits = self.logits(state_feat, ctx_feat)
        actions_j = _to_jax_int(actions)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        out = log_probs[jnp.arange(actions_j.shape[0]), actions_j]
        return _return_like(out, state_feat, ctx_feat, actions)

    def eval(self) -> _PolicyNetwork:
        return self


class _DiscriminatorBundle(eqx.Module):
    reward_net: _RewardNetwork
    shaping_net: _ShapingNetwork


class NeuralAIRL(NeuralEstimatorMixin):
    """Context-aware AIRL estimator with sklearn-style API."""

    def __init__(
        self,
        n_actions: int = 8,
        discount: float = 0.95,
        reward_hidden_dim: int = 128,
        reward_num_layers: int = 3,
        shaping_hidden_dim: int = 128,
        shaping_num_layers: int = 3,
        policy_hidden_dim: int = 128,
        policy_num_layers: int = 3,
        batch_size: int = 512,
        max_epochs: int = 500,
        disc_lr: float = 1e-3,
        policy_lr: float = 1e-3,
        disc_steps: int = 5,
        gradient_clip: float = 1.0,
        patience: int = 50,
        label_smoothing: float = 0.0,
        state_encoder: Callable[[object], object] | None = None,
        context_encoder: Callable[[object], object] | None = None,
        state_dim: int | None = None,
        context_dim: int = 0,
        feature_names: list[str] | None = None,
        verbose: bool = False,
    ):
        self.n_actions = n_actions
        self.discount = discount
        self.reward_hidden_dim = reward_hidden_dim
        self.reward_num_layers = reward_num_layers
        self.shaping_hidden_dim = shaping_hidden_dim
        self.shaping_num_layers = shaping_num_layers
        self.policy_hidden_dim = policy_hidden_dim
        self.policy_num_layers = policy_num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.disc_lr = disc_lr
        self.policy_lr = policy_lr
        self.disc_steps = disc_steps
        self.gradient_clip = gradient_clip
        self.patience = patience
        self.label_smoothing = label_smoothing
        self.state_encoder = state_encoder
        self.context_encoder = context_encoder
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.feature_names = feature_names
        self.verbose = verbose

        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.projection_r2_: float | None = None
        self.converged_: bool | None = None
        self.n_epochs_: int | None = None

        self._reward_net: _RewardNetwork | None = None
        self._shaping_net: _ShapingNetwork | None = None
        self._policy_net: _PolicyNetwork | None = None
        self._state_encoder: Callable[[object], jax.Array] | None = None
        self._context_encoder: Callable[[object], jax.Array] | None = None
        self._state_dim: int | None = None
        self._context_dim: int | None = None
        self._n_states: int | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        context: str | object | None = None,
        features: RewardSpec | object | None = None,
        transitions: object = None,
    ) -> NeuralAIRL:
        all_states, all_actions, all_next, all_contexts = self._extract_data(
            data, state, action, id, context
        )

        n_states = int(np.asarray(all_states).max()) + 1
        self._n_states = n_states

        self._build_encoders(all_states, all_contexts, n_states)

        key = jr.PRNGKey(np.random.randint(0, 2**31 - 1))
        reward_key, shaping_key, policy_key = jr.split(key, 3)
        self._reward_net = _RewardNetwork(
            self._state_dim,
            self._context_dim,
            self.n_actions,
            self.reward_hidden_dim,
            self.reward_num_layers,
            key=reward_key,
        )
        self._shaping_net = _ShapingNetwork(
            self._state_dim,
            self._context_dim,
            self.shaping_hidden_dim,
            self.shaping_num_layers,
            key=shaping_key,
        )
        self._policy_net = _PolicyNetwork(
            self._state_dim,
            self._context_dim,
            self.n_actions,
            self.policy_hidden_dim,
            self.policy_num_layers,
            key=policy_key,
        )

        self._train(all_states, all_actions, all_next, all_contexts)
        self._extract_policy_and_value(all_states, all_contexts, n_states)

        if features is not None:
            self._project_onto_features(
                features, all_states, all_actions, all_contexts
            )
        else:
            self.params_ = None
            self.se_ = None
            self.pvalues_ = None
            self.projection_r2_ = None
            self.coef_ = None

        return self

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        context: str | object | None,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
            all_states = jnp.asarray(panel.all_states, dtype=jnp.int32)
            all_actions = jnp.asarray(panel.all_actions, dtype=jnp.int32)
            all_next = jnp.asarray(panel.all_next_states, dtype=jnp.int32)

            if isinstance(context, str):
                all_contexts = self._extract_context_from_df(data, id, context, panel)
            elif context is not None:
                all_contexts = _to_jax_int(context)
            else:
                all_contexts = jnp.zeros(len(all_states), dtype=jnp.int32)
        elif isinstance(data, (Panel, TrajectoryPanel)):
            all_states = jnp.asarray(data.get_all_states(), dtype=jnp.int32)
            all_actions = jnp.asarray(data.get_all_actions(), dtype=jnp.int32)
            all_next = jnp.asarray(data.get_all_next_states(), dtype=jnp.int32)
            if context is not None:
                all_contexts = _to_jax_int(context)
            else:
                all_contexts = jnp.zeros(len(all_states), dtype=jnp.int32)
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, got {type(data)}"
            )

        return all_states, all_actions, all_next, all_contexts

    def _extract_context_from_df(
        self,
        df: pd.DataFrame,
        id_col: str,
        context_col: str,
        panel: TrajectoryPanel,
    ) -> jax.Array:
        contexts: list[int] = []
        for _, group in df.groupby(id_col, sort=True):
            group = group.sort_index()
            contexts.extend(group[context_col].values.tolist())
        return jnp.asarray(contexts, dtype=jnp.int32)

    def _call_encoder(self, encoder: Callable[[object], object], values: object) -> jax.Array:
        try:
            encoded = encoder(values)
        except Exception as err:
            if torch is None:
                raise err
            torch_values = torch.tensor(_to_numpy(values).copy(), dtype=torch.long)
            encoded = encoder(torch_values)
        return _to_jax_float(encoded)

    def _build_encoders(
        self,
        all_states: jax.Array,
        all_contexts: jax.Array,
        n_states: int,
    ) -> None:
        if self.state_encoder is not None:
            self._state_encoder = lambda s: self._call_encoder(self.state_encoder, s)
            self._state_dim = self.state_dim or 1
        else:
            max_s = max(n_states - 1, 1)
            self._state_encoder = lambda s, _ms=max_s: (
                _to_jax_float(s) / float(_ms)
            ).reshape(-1, 1)
            self._state_dim = 1

        if self.context_encoder is not None:
            self._context_encoder = lambda c: self._call_encoder(self.context_encoder, c)
            self._context_dim = self.context_dim or 1
        else:
            n_ctx = max(int(np.asarray(all_contexts).max()), 1) if len(all_contexts) else 1
            self._context_encoder = lambda c, _mc=n_ctx: (
                _to_jax_float(c) / float(_mc)
            ).reshape(-1, 1)
            self._context_dim = 1

    def _train(
        self,
        states: jax.Array,
        actions: jax.Array,
        next_states: jax.Array,
        contexts: jax.Array,
    ) -> None:
        disc_model = _DiscriminatorBundle(self._reward_net, self._shaping_net)
        policy_net = self._policy_net

        disc_transforms = []
        policy_transforms = []
        if self.gradient_clip > 0:
            disc_transforms.append(optax.clip_by_global_norm(self.gradient_clip))
            policy_transforms.append(optax.clip_by_global_norm(self.gradient_clip))
        disc_transforms.append(optax.adam(self.disc_lr))
        policy_transforms.append(optax.adam(self.policy_lr))

        disc_optimizer = optax.chain(*disc_transforms)
        policy_optimizer = optax.chain(*policy_transforms)

        disc_opt_state = disc_optimizer.init(eqx.filter(disc_model, eqx.is_inexact_array))
        policy_opt_state = policy_optimizer.init(eqx.filter(policy_net, eqx.is_inexact_array))

        N = len(states)
        best_loss = float("inf")
        patience_counter = 0
        expert_label = 1.0 - self.label_smoothing
        policy_label = 0.0 + self.label_smoothing

        def compute_disc_logits(
            disc: _DiscriminatorBundle,
            policy: _PolicyNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            action_idx: jax.Array,
            ns_feat: jax.Array,
        ) -> jax.Array:
            action_oh = jax.nn.one_hot(action_idx, self.n_actions, dtype=jnp.float32)
            g = disc.reward_net(s_feat, ctx_feat, action_oh)
            h_s = disc.shaping_net(s_feat, ctx_feat)
            h_ns = disc.shaping_net(ns_feat, ctx_feat)
            log_pi = policy.log_prob(s_feat, ctx_feat, action_idx)
            return g + self.discount * h_ns - h_s - log_pi

        @eqx.filter_value_and_grad
        def disc_loss_fn(
            disc: _DiscriminatorBundle,
            policy: _PolicyNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            action_idx: jax.Array,
            ns_feat: jax.Array,
            key: jax.Array,
        ) -> jax.Array:
            expert_logits = compute_disc_logits(disc, policy, s_feat, ctx_feat, action_idx, ns_feat)
            policy_probs = policy(s_feat, ctx_feat)
            policy_actions = _sample_actions(policy_probs, key)
            policy_logits = compute_disc_logits(disc, policy, s_feat, ctx_feat, policy_actions, ns_feat)
            expert_targets = jnp.full_like(expert_logits, expert_label)
            policy_targets = jnp.full_like(policy_logits, policy_label)
            loss = _bce_with_logits(expert_logits, expert_targets).mean()
            loss = loss + _bce_with_logits(policy_logits, policy_targets).mean()
            return loss

        @eqx.filter_value_and_grad
        def policy_loss_fn(
            policy: _PolicyNetwork,
            disc: _DiscriminatorBundle,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            ns_feat: jax.Array,
            key: jax.Array,
        ) -> jax.Array:
            policy_probs = policy(s_feat, ctx_feat)
            policy_actions = _sample_actions(policy_probs, key)
            log_pi = policy.log_prob(s_feat, ctx_feat, policy_actions)
            disc_logits = compute_disc_logits(disc, policy, s_feat, ctx_feat, policy_actions, ns_feat)
            disc_reward = -jax.nn.softplus(-disc_logits)
            entropy = -(policy_probs * jnp.log(policy_probs + 1e-10)).sum(axis=-1)
            return -(log_pi * jax.lax.stop_gradient(disc_reward)).mean() - 0.01 * entropy.mean()

        @eqx.filter_jit
        def disc_step(
            disc: _DiscriminatorBundle,
            disc_state: optax.OptState,
            policy: _PolicyNetwork,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            action_idx: jax.Array,
            ns_feat: jax.Array,
            key: jax.Array,
        ) -> tuple[_DiscriminatorBundle, optax.OptState, jax.Array]:
            loss, grads = disc_loss_fn(disc, policy, s_feat, ctx_feat, action_idx, ns_feat, key)
            updates, disc_state = disc_optimizer.update(grads, disc_state, disc)
            disc = eqx.apply_updates(disc, updates)
            return disc, disc_state, loss

        @eqx.filter_jit
        def policy_step(
            policy: _PolicyNetwork,
            policy_state: optax.OptState,
            disc: _DiscriminatorBundle,
            s_feat: jax.Array,
            ctx_feat: jax.Array,
            ns_feat: jax.Array,
            key: jax.Array,
        ) -> tuple[_PolicyNetwork, optax.OptState, jax.Array]:
            loss, grads = policy_loss_fn(policy, disc, s_feat, ctx_feat, ns_feat, key)
            updates, policy_state = policy_optimizer.update(grads, policy_state, policy)
            policy = eqx.apply_updates(policy, updates)
            return policy, policy_state, loss

        best_disc_model = disc_model
        best_policy_net = policy_net

        for epoch in range(self.max_epochs):
            perm = np.random.permutation(N)
            epoch_disc_loss = 0.0
            epoch_policy_loss = 0.0
            n_batches = 0

            for start in range(0, N, self.batch_size):
                idx = perm[start : start + self.batch_size]
                s = states[idx]
                a = actions[idx]
                ns = next_states[idx]
                ctx = contexts[idx]

                s_feat = self._state_encoder(s)
                ns_feat = self._state_encoder(ns)
                ctx_feat = self._context_encoder(ctx)

                last_disc_loss = 0.0
                for _ in range(self.disc_steps):
                    disc_key = jr.PRNGKey(np.random.randint(0, 2**31 - 1))
                    disc_model, disc_opt_state, disc_loss = disc_step(
                        disc_model,
                        disc_opt_state,
                        policy_net,
                        s_feat,
                        ctx_feat,
                        a,
                        ns_feat,
                        disc_key,
                    )
                    last_disc_loss = float(disc_loss)

                policy_key = jr.PRNGKey(np.random.randint(0, 2**31 - 1))
                policy_net, policy_opt_state, policy_loss = policy_step(
                    policy_net,
                    policy_opt_state,
                    disc_model,
                    s_feat,
                    ctx_feat,
                    ns_feat,
                    policy_key,
                )

                epoch_disc_loss += last_disc_loss
                epoch_policy_loss += float(policy_loss)
                n_batches += 1

            avg_disc_loss = epoch_disc_loss / max(n_batches, 1)
            avg_policy_loss = epoch_policy_loss / max(n_batches, 1)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}: disc_loss={avg_disc_loss:.4f} "
                    f"policy_loss={avg_policy_loss:.4f}"
                )

            if avg_disc_loss < best_loss - 1e-4:
                best_loss = avg_disc_loss
                patience_counter = 0
                best_disc_model = disc_model
                best_policy_net = policy_net
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        self._reward_net = best_disc_model.reward_net
        self._shaping_net = best_disc_model.shaping_net
        self._policy_net = best_policy_net
        self.converged_ = patience_counter >= self.patience or epoch == self.max_epochs - 1
        self.n_epochs_ = epoch + 1

    def _extract_policy_and_value(
        self,
        all_states: jax.Array,
        all_contexts: jax.Array,
        n_states: int,
    ) -> None:
        unique_states = jnp.arange(n_states, dtype=jnp.int32)
        ctx_default = jnp.zeros(n_states, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(ctx_default)
        policy = self._policy_net(s_feat, ctx_feat)
        value = self._shaping_net(s_feat, ctx_feat)
        self.policy_ = np.asarray(policy)
        self.value_ = np.asarray(value)

    def _project_onto_features(
        self,
        features: RewardSpec | object,
        states: jax.Array,
        actions: jax.Array,
        contexts: jax.Array,
    ) -> None:
        if isinstance(features, RewardSpec):
            feat_matrix = features.feature_matrix
            names = features.parameter_names
        else:
            feat_matrix = features
            names = self.feature_names or [f"f{i}" for i in range(np.asarray(features).shape[-1])]

        states_j = _to_jax_int(states)
        actions_j = _to_jax_int(actions)
        contexts_j = _to_jax_int(contexts)
        s_feat = self._state_encoder(states_j)
        ctx_feat = self._context_encoder(contexts_j)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        rewards = jnp.asarray(self._reward_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)

        feat_np = _to_numpy(feat_matrix)
        phi = feat_np[np.asarray(states_j), np.asarray(actions_j), :]

        theta, se, r2 = self._project_parameters(phi, rewards)
        self.params_ = {n: float(v) for n, v in zip(names, theta)}
        self.se_ = {n: float(v) for n, v in zip(names, se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = r2
        self.coef_ = np.asarray(theta)

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        if self._reward_net is None or self._n_states is None:
            return None

        unique_states = jnp.arange(self._n_states, dtype=jnp.int32)
        ctx_default = jnp.zeros(self._n_states, dtype=jnp.int32)
        s_feat = self._state_encoder(unique_states)
        ctx_feat = self._context_encoder(ctx_default)
        r_all = self._reward_net.all_actions(s_feat, ctx_feat, self.n_actions)
        return np.asarray(r_all)

    def predict_proba(self, states: np.ndarray) -> np.ndarray:
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states = np.asarray(states, dtype=np.int64)
        return self.policy_[states]

    def predict_proba_from_features(
        self,
        state_features: object,
        contexts: object | None = None,
    ) -> np.ndarray:
        if self._policy_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        s_feat = _to_jax_float(state_features)
        if s_feat.ndim == 1:
            s_feat = s_feat[None, :]
        if contexts is None:
            contexts = jnp.zeros(s_feat.shape[0], dtype=jnp.int32)
        ctx_feat = self._context_encoder(contexts)
        probs = self._policy_net(s_feat, ctx_feat)
        return np.asarray(probs)

    def predict_reward_from_features(
        self,
        state_features: object,
        actions: object,
        contexts: object | None = None,
    ) -> np.ndarray:
        if self._reward_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        s_feat = _to_jax_float(state_features)
        if s_feat.ndim == 1:
            s_feat = s_feat[None, :]
        actions_j = _to_jax_int(actions)
        if actions_j.ndim == 0:
            actions_j = actions_j[None]
        if contexts is None:
            contexts = jnp.zeros(s_feat.shape[0], dtype=jnp.int32)
        ctx_feat = self._context_encoder(contexts)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        rewards = self._reward_net(s_feat, ctx_feat, a_oh)
        return np.asarray(rewards)

    def predict_reward(
        self,
        states: object,
        actions: object,
        contexts: object | None = None,
    ) -> object:
        if self._reward_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        states_j = _to_jax_int(states)
        actions_j = _to_jax_int(actions)
        if contexts is None:
            contexts_j = jnp.zeros(states_j.shape[0], dtype=jnp.int32)
        else:
            contexts_j = _to_jax_int(contexts)

        s_feat = self._state_encoder(states_j)
        ctx_feat = self._context_encoder(contexts_j)
        a_oh = jax.nn.one_hot(actions_j, self.n_actions, dtype=jnp.float32)
        rewards = jnp.asarray(self._reward_net(s_feat, ctx_feat, a_oh), dtype=jnp.float32)
        return _return_like(rewards, states, actions, contexts)

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
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

    def summary(self) -> str:
        if self.policy_ is None:
            return "NeuralAIRL: Not fitted yet. Call fit() first."

        n_obs = None
        if self._n_states is not None and self.policy_ is not None:
            n_obs = self._n_states

        return self._format_neural_summary(
            method_name="NeuralAIRL",
            params=self.params_,
            se=self.se_,
            pvalues=self.pvalues_,
            projection_r2=self.projection_r2_,
            n_observations=n_obs,
            n_epochs=self.n_epochs_,
            converged=self.converged_,
            discount=self.discount,
            context_dim=self._context_dim,
            extra_lines=[
                f"Reward network: {self.reward_num_layers} layers x {self.reward_hidden_dim} hidden",
                f"Shaping network: {self.shaping_num_layers} layers x {self.shaping_hidden_dim} hidden",
                f"Policy network: {self.policy_num_layers} layers x {self.policy_hidden_dim} hidden",
            ],
        )

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"NeuralAIRL(n_actions={self.n_actions}, "
            f"discount={self.discount}, "
            f"fitted={fitted})"
        )
