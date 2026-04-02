"""NeuralGLADIUS: Context-aware Q-learning with Bellman consistency penalty.

Learns Q(s,a,ctx) and EV(s,a,ctx) via mini-batch training, then extracts
structural parameters by projecting implied rewards onto features.

No transition matrix needed. Supports context conditioning (destination,
time-of-day, user segment, etc.) through pluggable encoders.

Algorithm:
    1. Parameterize Q(s,a,ctx) and EV(s,a,ctx) with MLPs.
    2. Train via mini-batch SGD on observed (s, a, s', ctx) tuples:
       - NLL loss: negative log-likelihood of observed actions under
         softmax policy derived from Q.
       - Bellman penalty: (EV(s,a,ctx) - V(s',ctx))^2 where
         V(s',ctx) = sigma * logsumexp(Q(s', :, ctx) / sigma).
    3. After training, implied rewards r(s,a,ctx) = Q(s,a,ctx) - beta*EV(s,a,ctx)
       are projected onto features via least-squares to recover theta.

Reference:
    Kang, M., et al. (2025). DDC IRL with neural networks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm

from econirl.core.reward_spec import RewardSpec
from econirl.core.types import Panel, TrajectoryPanel
from econirl.estimators.neural_base import NeuralEstimatorMixin


# ---------------------------------------------------------------------------
# Internal network modules
# ---------------------------------------------------------------------------


class _ContextQNetwork(nn.Module):
    """Q(s, a, ctx) network.

    Input: concatenation of [state_features, context_features, action_onehot].
    Output: scalar Q value.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self.n_actions = n_actions
        input_dim = state_dim + context_dim + n_actions
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        action_onehot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Q(s, a, ctx).

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).
        ctx_feat : torch.Tensor
            Context features of shape (B, context_dim).
        action_onehot : torch.Tensor
            One-hot action of shape (B, n_actions).

        Returns
        -------
        torch.Tensor
            Q values of shape (B,).
        """
        x = torch.cat([state_feat, ctx_feat, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)

    def all_actions(
        self,
        state_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        n_actions: int,
    ) -> torch.Tensor:
        """Compute Q for all actions at once.

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).
        ctx_feat : torch.Tensor
            Context features of shape (B, context_dim).
        n_actions : int
            Number of actions.

        Returns
        -------
        torch.Tensor
            Q values of shape (B, n_actions).
        """
        B = state_feat.shape[0]
        actions = torch.eye(n_actions, device=state_feat.device)  # (A, A)
        actions = actions.unsqueeze(0).expand(B, -1, -1)  # (B, A, A)
        sf = state_feat.unsqueeze(1).expand(-1, n_actions, -1)  # (B, A, d_s)
        cf = ctx_feat.unsqueeze(1).expand(-1, n_actions, -1)  # (B, A, d_c)
        x = torch.cat([sf, cf, actions], dim=-1)  # (B, A, input_dim)
        return self.net(x.reshape(B * n_actions, -1)).reshape(B, n_actions)


# ---------------------------------------------------------------------------
# NeuralGLADIUS estimator
# ---------------------------------------------------------------------------


class NeuralGLADIUS(NeuralEstimatorMixin):
    """Context-aware GLADIUS estimator with sklearn-style API.

    Trains Q-network and EV-network via NLL + Bellman penalty on mini-batches
    of (s, a, s', ctx). After training, implied rewards r = Q - beta*EV are
    projected onto features via least-squares to get interpretable theta.

    No transition matrix required. Supports context conditioning through
    pluggable state/context encoders.

    Parameters
    ----------
    n_actions : int, default=8
        Number of discrete actions.
    discount : float, default=0.95
        Time discount factor beta.
    scale : float, default=1.0
        Logit scale parameter sigma.
    q_hidden_dim : int, default=128
        Hidden dimension for the Q-network MLP.
    q_num_layers : int, default=3
        Number of hidden layers in the Q-network.
    ev_hidden_dim : int, default=128
        Hidden dimension for the EV-network MLP.
    ev_num_layers : int, default=3
        Number of hidden layers in the EV-network.
    batch_size : int, default=512
        Mini-batch size for SGD.
    max_epochs : int, default=500
        Maximum number of training epochs.
    lr : float, default=1e-3
        Learning rate for Adam optimizer.
    bellman_weight : float, default=1.0
        Weight on the Bellman consistency penalty.
    gradient_clip : float, default=1.0
        Maximum gradient norm for clipping. 0 disables clipping.
    patience : int, default=50
        Early stopping patience (epochs without improvement).
    alternating_updates : bool, default=True
        Use alternating zeta/Q optimization from Algorithm 1 in
        Kang et al. (2025). When False, both networks are updated
        jointly in each step (legacy behavior).
    lr_decay_rate : float, default=0.001
        Learning rate decay rate. Effective LR decays as
        lr_0 / (1 + decay_rate * step). Set to 0 for constant LR.
    tikhonov_annealing : bool, default=False
        When True, NLL loss is scaled by tikhonov_initial_weight /
        (1 + epoch), transitioning from behavioral cloning early to
        Bellman-driven refinement later.
    tikhonov_initial_weight : float, default=100.0
        Initial weight on NLL when Tikhonov annealing is enabled.
    anchor_action : int or None, default=None
        When set, Bellman error is only computed for transitions
        where a equals this action index.
    state_encoder : callable, optional
        Function mapping state indices (long tensor) to feature vectors.
        Receives shape (B,) and should return shape (B, state_dim).
        If None, a default normalizing encoder is created.
    context_encoder : callable, optional
        Function mapping context indices (long tensor) to feature vectors.
        Receives shape (B,) and should return shape (B, context_dim).
        If None, a default normalizing encoder is created.
    state_dim : int, optional
        Dimension of state features. Required if state_encoder is provided.
    context_dim : int, default=0
        Dimension of context features. Required if context_encoder is provided.
    feature_names : list of str, optional
        Names for features when using raw tensor features for projection.
    verbose : bool, default=False
        Whether to print progress during training.

    Attributes
    ----------
    params_ : dict or None
        Projected structural parameters after fitting. None if no features
        were provided.
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
    projection_r2_ : float or None
        R-squared of the feature projection.
    converged_ : bool or None
        Whether training converged (early stopping or max epochs).
    n_epochs_ : int or None
        Number of training epochs completed.

    Examples
    --------
    >>> from econirl.estimators import NeuralGLADIUS
    >>> import pandas as pd
    >>>
    >>> model = NeuralGLADIUS(n_actions=3, discount=0.95, max_epochs=200)
    >>> model.fit(data=df, state="state", action="action", id="agent_id")
    >>> print(model.policy_.shape)  # (n_states, n_actions)
    >>>
    >>> # With context and feature projection
    >>> model.fit(data=df, state="state", action="action", id="agent_id",
    ...           context="destination", features=reward_spec)
    >>> print(model.params_)
    >>> print(model.projection_r2_)
    """

    def __init__(
        self,
        n_actions: int = 8,
        discount: float = 0.95,
        scale: float = 1.0,
        q_hidden_dim: int = 128,
        q_num_layers: int = 3,
        ev_hidden_dim: int = 128,
        ev_num_layers: int = 3,
        batch_size: int = 512,
        max_epochs: int = 500,
        lr: float = 1e-3,
        bellman_weight: float = 1.0,
        gradient_clip: float = 1.0,
        patience: int = 50,
        alternating_updates: bool = True,
        lr_decay_rate: float = 0.001,
        tikhonov_annealing: bool = False,
        tikhonov_initial_weight: float = 100.0,
        anchor_action: int | None = None,
        state_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        context_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        state_dim: int | None = None,
        context_dim: int = 0,
        feature_names: list[str] | None = None,
        verbose: bool = False,
    ):
        self.n_actions = n_actions
        self.discount = discount
        self.scale = scale
        self.q_hidden_dim = q_hidden_dim
        self.q_num_layers = q_num_layers
        self.ev_hidden_dim = ev_hidden_dim
        self.ev_num_layers = ev_num_layers
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.lr = lr
        self.bellman_weight = bellman_weight
        self.gradient_clip = gradient_clip
        self.patience = patience
        self.alternating_updates = alternating_updates
        self.lr_decay_rate = lr_decay_rate
        self.tikhonov_annealing = tikhonov_annealing
        self.tikhonov_initial_weight = tikhonov_initial_weight
        self.anchor_action = anchor_action
        self.state_encoder = state_encoder
        self.context_encoder = context_encoder
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.feature_names = feature_names
        self.verbose = verbose

        # Fitted attributes (set after fit())
        self.params_: dict[str, float] | None = None
        self.se_: dict[str, float] | None = None
        self.pvalues_: dict[str, float] | None = None
        self.coef_: np.ndarray | None = None
        self.policy_: np.ndarray | None = None
        self.value_: np.ndarray | None = None
        self.projection_r2_: float | None = None
        self.converged_: bool | None = None
        self.n_epochs_: int | None = None

        # Internal state
        self._q_net: _ContextQNetwork | None = None
        self._ev_net: _ContextQNetwork | None = None
        self._state_encoder: Callable | None = None
        self._context_encoder: Callable | None = None
        self._state_dim: int | None = None
        self._context_dim: int | None = None
        self._n_states: int | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        context: str | torch.Tensor | None = None,
        features: RewardSpec | torch.Tensor | None = None,
        transitions: object = None,
    ) -> "NeuralGLADIUS":
        """Fit the NeuralGLADIUS estimator to data.

        Parameters
        ----------
        data : pandas.DataFrame or Panel or TrajectoryPanel
            Panel data with observations. When a DataFrame is passed,
            ``state``, ``action``, and ``id`` column names are required.
        state : str, optional
            Column name for the state variable (required for DataFrame input).
        action : str, optional
            Column name for the action variable (required for DataFrame input).
        id : str, optional
            Column name for the individual identifier (required for DataFrame
            input).
        context : str or torch.Tensor, optional
            Context information. If a string, it is treated as a column name
            in the DataFrame. If a Tensor, it should have shape (N,) with
            context indices aligned with the panel observations. If None,
            no context conditioning is used.
        features : RewardSpec or torch.Tensor, optional
            Feature specification for parameter projection. If a RewardSpec,
            uses its feature_matrix (S, A, K) and parameter_names. If a
            Tensor, should have shape (S, A, K). If None, no projection is
            done and params_ will be None.
        transitions : ignored
            Accepted for API compatibility but not used. NeuralGLADIUS does
            not require a transition matrix.

        Returns
        -------
        self : NeuralGLADIUS
            Returns self for method chaining.
        """
        # --- Step 1: Extract tensors from data ---
        all_states, all_actions, all_next, all_contexts = self._extract_data(
            data, state, action, id, context
        )

        n_states = int(all_states.max().item()) + 1
        self._n_states = n_states

        # --- Step 2: Build encoders if not provided ---
        self._build_encoders(all_states, all_contexts, n_states)

        # --- Step 3: Build networks ---
        sd = self._state_dim
        cd = self._context_dim
        self._q_net = _ContextQNetwork(
            sd, cd, self.n_actions, self.q_hidden_dim, self.q_num_layers
        )
        self._ev_net = _ContextQNetwork(
            sd, cd, self.n_actions, self.ev_hidden_dim, self.ev_num_layers
        )

        # --- Step 4: Training loop ---
        self._train(all_states, all_actions, all_next, all_contexts)

        # --- Step 5: Extract policy and value ---
        self._extract_policy_and_value(all_states, all_contexts, n_states)

        # --- Step 6: Feature projection ---
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

    # ------------------------------------------------------------------
    # Data extraction
    # ------------------------------------------------------------------

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        context: str | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract state/action/next_state/context tensors from input data.

        Returns
        -------
        tuple of torch.Tensor
            (all_states, all_actions, all_next_states, all_contexts)
        """
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
            all_states = torch.tensor(np.asarray(panel.all_states), dtype=torch.long)
            all_actions = torch.tensor(np.asarray(panel.all_actions), dtype=torch.long)
            all_next = torch.tensor(np.asarray(panel.all_next_states), dtype=torch.long)

            if isinstance(context, str):
                all_contexts = self._extract_context_from_df(
                    data, id, context, panel
                )
            elif context is not None:
                all_contexts = (
                    context
                    if isinstance(context, torch.Tensor)
                    else torch.tensor(context, dtype=torch.long)
                )
            else:
                all_contexts = torch.zeros(len(all_states), dtype=torch.long)

        elif isinstance(data, (Panel, TrajectoryPanel)):
            all_states = torch.tensor(np.asarray(data.get_all_states()), dtype=torch.long)
            all_actions = torch.tensor(np.asarray(data.get_all_actions()), dtype=torch.long)
            all_next = torch.tensor(np.asarray(data.get_all_next_states()), dtype=torch.long)

            if context is not None and isinstance(context, torch.Tensor):
                all_contexts = context
            else:
                all_contexts = torch.zeros(len(all_states), dtype=torch.long)
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, "
                f"got {type(data)}"
            )

        return all_states, all_actions, all_next, all_contexts

    def _extract_context_from_df(
        self,
        df: pd.DataFrame,
        id_col: str,
        context_col: str,
        panel: TrajectoryPanel,
    ) -> torch.Tensor:
        """Extract context values from DataFrame aligned with panel observations.

        Parameters
        ----------
        df : pd.DataFrame
            Source DataFrame.
        id_col : str
            Column name for individual ID.
        context_col : str
            Column name for context variable.
        panel : TrajectoryPanel
            Panel built from the same DataFrame.

        Returns
        -------
        torch.Tensor
            Context tensor of shape (N,) aligned with panel observations.
        """
        # Build context aligned with how TrajectoryPanel.from_dataframe
        # processes groups: sorted by id, then by index within each group.
        contexts: list[int] = []
        for _, group in df.groupby(id_col, sort=True):
            group = group.sort_index()
            contexts.extend(group[context_col].values.tolist())
        return torch.tensor(contexts, dtype=torch.long)

    # ------------------------------------------------------------------
    # Encoder setup
    # ------------------------------------------------------------------

    def _build_encoders(
        self,
        all_states: torch.Tensor,
        all_contexts: torch.Tensor,
        n_states: int,
    ) -> None:
        """Build default encoders if not provided by the user."""
        if self.state_encoder is not None:
            self._state_encoder = self.state_encoder
            self._state_dim = self.state_dim or 1
        else:
            max_s = max(n_states - 1, 1)
            self._state_encoder = lambda s, _ms=max_s: (
                s.float() / _ms
            ).unsqueeze(-1)
            self._state_dim = 1

        if self.context_encoder is not None:
            self._context_encoder = self.context_encoder
            self._context_dim = self.context_dim or 1
        else:
            n_ctx = max(int(all_contexts.max().item()), 1)
            self._context_encoder = lambda c, _mc=n_ctx: (
                c.float() / _mc
            ).unsqueeze(-1)
            self._context_dim = 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        contexts: torch.Tensor,
    ) -> None:
        """Run the mini-batch training loop.

        When alternating_updates is True (default), follows Algorithm 1
        from Kang et al. (2025): even batches update the zeta (EV)
        network, odd batches update the Q network. This stabilizes
        training by giving each network a stable target.

        Learning rate decays as lr_0 / (1 + decay_rate * step). Optional
        Tikhonov annealing scales the NLL weight by 1/(1+epoch) to
        transition from behavioral cloning to Bellman-driven refinement.
        """
        # Separate optimizers for alternating updates.
        q_optimizer = torch.optim.Adam(
            self._q_net.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        zeta_optimizer = torch.optim.Adam(
            self._ev_net.parameters(), lr=self.lr, weight_decay=1e-4,
        )

        # LR decay: lr(t) = lr_0 / (1 + decay * t)
        decay = self.lr_decay_rate
        q_scheduler = torch.optim.lr_scheduler.LambdaLR(
            q_optimizer, lambda step: 1.0 / (1.0 + decay * step)
        )
        zeta_scheduler = torch.optim.lr_scheduler.LambdaLR(
            zeta_optimizer, lambda step: 1.0 / (1.0 + decay * step)
        )

        N = len(states)
        best_loss = float("inf")
        patience_counter = 0

        # Compute inverse-frequency class weights for the NLL loss.
        # This prevents rare actions (e.g., 0.2% replacement events)
        # from being drowned out by the majority action.
        action_counts = torch.bincount(actions.long(), minlength=self.n_actions).float()
        action_counts = action_counts.clamp(min=1.0)
        class_weights = N / (self.n_actions * action_counts)
        if self.verbose and class_weights.max() / class_weights.min() > 10:
            print(f"  Class weights: {class_weights.tolist()} "
                  f"(ratio {class_weights.max()/class_weights.min():.0f}x)")

        for epoch in range(self.max_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0

            # Tikhonov annealing: decay CE weight over epochs
            if self.tikhonov_annealing:
                ce_weight = self.tikhonov_initial_weight / (1.0 + epoch)
            else:
                ce_weight = 1.0

            batch_idx = 0
            for i in range(0, N, self.batch_size):
                idx = perm[i : i + self.batch_size]
                s = states[idx]
                a = actions[idx]
                ns = next_states[idx]
                ctx = contexts[idx]

                # Encode
                s_feat = self._state_encoder(s)
                ns_feat = self._state_encoder(ns)
                ctx_feat = self._context_encoder(ctx)
                a_oh = F.one_hot(a.long(), self.n_actions).float()

                if self.alternating_updates and batch_idx % 2 == 0:
                    # Even batch: update zeta (EV) network only.
                    # Minimize MSE between zeta(s,a) and V(s') from Q.
                    zeta_sa = self._ev_net(s_feat, ctx_feat, a_oh)
                    with torch.no_grad():
                        q_next_all = self._q_net.all_actions(
                            ns_feat, ctx_feat, self.n_actions
                        )
                        v_next = self.scale * torch.logsumexp(
                            q_next_all / self.scale, dim=1
                        )
                    loss = ((zeta_sa - v_next) ** 2).mean()

                    zeta_optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self._ev_net.parameters(), self.gradient_clip,
                        )
                    zeta_optimizer.step()
                    zeta_scheduler.step()

                elif self.alternating_updates and batch_idx % 2 == 1:
                    # Odd batch: update Q network only.
                    # In the IRL setting (no observed rewards), Q is
                    # trained only by NLL. Bellman consistency is enforced
                    # by the zeta step. Passing Bellman gradients through
                    # V_Q(s') = logsumexp(Q) causes Q-value explosion.
                    q_all = self._q_net.all_actions(
                        s_feat, ctx_feat, self.n_actions
                    )

                    # Weighted NLL loss
                    log_probs = torch.log_softmax(q_all / self.scale, dim=1)
                    per_obs_nll = -log_probs[torch.arange(len(a)), a.long()]
                    weights = class_weights[a.long()]
                    nll = (per_obs_nll * weights).mean()

                    loss = ce_weight * nll

                    q_optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            self._q_net.parameters(), self.gradient_clip,
                        )
                    q_optimizer.step()
                    q_scheduler.step()

                else:
                    # Legacy joint update (alternating_updates=False).
                    q_all = self._q_net.all_actions(
                        s_feat, ctx_feat, self.n_actions
                    )
                    log_probs = torch.log_softmax(q_all / self.scale, dim=1)
                    per_obs_nll = -log_probs[torch.arange(len(a)), a.long()]
                    weights = class_weights[a.long()]
                    nll = (per_obs_nll * weights).mean()

                    ev_sa = self._ev_net(s_feat, ctx_feat, a_oh)
                    q_next_all = self._q_net.all_actions(
                        ns_feat, ctx_feat, self.n_actions
                    )
                    v_next = self.scale * torch.logsumexp(
                        q_next_all / self.scale, dim=1
                    )
                    bellman = ((ev_sa - v_next.detach()) ** 2).mean()
                    loss = ce_weight * nll + self.bellman_weight * bellman

                    q_optimizer.zero_grad()
                    zeta_optimizer.zero_grad()
                    loss.backward()
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            list(self._q_net.parameters())
                            + list(self._ev_net.parameters()),
                            self.gradient_clip,
                        )
                    q_optimizer.step()
                    zeta_optimizer.step()
                    q_scheduler.step()
                    zeta_scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1
                batch_idx += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}: loss={avg_loss:.4f}"
                )

            # Early stopping
            if avg_loss < best_loss - 1e-4:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        self.converged_ = (
            patience_counter >= self.patience or epoch == self.max_epochs - 1
        )
        self.n_epochs_ = epoch + 1

    # ------------------------------------------------------------------
    # Post-training extraction
    # ------------------------------------------------------------------

    def _extract_policy_and_value(
        self,
        all_states: torch.Tensor,
        all_contexts: torch.Tensor,
        n_states: int,
    ) -> None:
        """Compute policy and value function for all states.

        Evaluates at context=0 for the policy/value matrices, which is
        appropriate for the default encoder. For multi-context models,
        use predict_reward() for context-specific predictions.
        """
        self._q_net.eval()
        self._ev_net.eval()

        with torch.no_grad():
            unique_states = torch.arange(n_states)
            ctx_default = torch.zeros(n_states, dtype=torch.long)

            s_feat = self._state_encoder(unique_states)
            ctx_feat = self._context_encoder(ctx_default)

            q_all = self._q_net.all_actions(s_feat, ctx_feat, self.n_actions)
            policy = torch.softmax(q_all / self.scale, dim=1)
            value = self.scale * torch.logsumexp(q_all / self.scale, dim=1)

            self.policy_ = policy.numpy()
            self.value_ = value.numpy()

    def _project_onto_features(
        self,
        features: RewardSpec | torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        contexts: torch.Tensor,
    ) -> None:
        """Project implied rewards onto features for interpretable theta.

        Parameters
        ----------
        features : RewardSpec or torch.Tensor
            Feature specification. RewardSpec provides (S, A, K) matrix and
            parameter names. Tensor should be (S, A, K).
        states : torch.Tensor
            Observed state indices.
        actions : torch.Tensor
            Observed action indices.
        contexts : torch.Tensor
            Observed context indices.
        """
        if isinstance(features, RewardSpec):
            feat_matrix = features.feature_matrix  # (S, A, K)
            names = features.parameter_names
        else:
            feat_matrix = features
            names = self.feature_names or [
                f"f{i}" for i in range(features.shape[-1])
            ]

        if not isinstance(feat_matrix, torch.Tensor):
            feat_matrix = torch.tensor(np.asarray(feat_matrix), dtype=torch.float32)

        # Compute implied rewards for ALL state-action pairs at the
        # state level, then project ACTION DIFFERENCES onto feature
        # differences. This eliminates the unidentified constant in
        # the absolute Q-level (Kim et al. 2021, Cao & Cohen 2021).
        n_s = self._n_states
        unique_states = torch.arange(n_s, dtype=torch.long)
        unique_ctx = torch.zeros(n_s, dtype=torch.long)

        with torch.no_grad():
            s_feat = self._state_encoder(unique_states)
            ctx_feat = self._context_encoder(unique_ctx)

            # Compute Q and EV for each action at each state
            q_all = self._q_net.all_actions(s_feat, ctx_feat, self.n_actions)
            r_all = torch.zeros(n_s, self.n_actions)
            for a_idx in range(self.n_actions):
                a_oh = F.one_hot(
                    torch.full((n_s,), a_idx, dtype=torch.long),
                    self.n_actions,
                ).float()
                ev_a = self._ev_net(s_feat, ctx_feat, a_oh)
                r_all[:, a_idx] = q_all[:, a_idx] - self.discount * ev_a

        # Project using action differences relative to action 0.
        # For each state s and action a>0:
        #   dr(s) = r(s, a) - r(s, 0)
        #   dphi(s) = phi(s, a) - phi(s, 0)
        # The constant in the absolute reward cancels in the difference,
        # so both level and slope parameters are identified.
        dr_list = []
        dphi_list = []
        for a_idx in range(1, self.n_actions):
            dr = r_all[:, a_idx] - r_all[:, 0]  # (S,)
            dphi = feat_matrix[:n_s, a_idx, :] - feat_matrix[:n_s, 0, :]  # (S, K)
            dr_list.append(dr)
            dphi_list.append(dphi)

        rewards = torch.cat(dr_list).float()
        phi = torch.cat(dphi_list).float()

        theta, se, r2 = self._project_parameters(phi, rewards)

        self.params_ = {n: float(v) for n, v in zip(names, theta)}
        self.se_ = {n: float(v) for n, v in zip(names, se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = r2
        self.coef_ = theta.numpy()

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

    @property
    def reward_matrix_(self) -> np.ndarray | None:
        """Structural reward matrix R(s,a) of shape (n_states, n_actions).

        Computes implied rewards r(s,a) = Q(s,a) - beta*EV(s,a) for all
        state-action pairs evaluated at context=0. Returns None if the
        model has not been fitted.
        """
        if self._q_net is None or self._ev_net is None or self._n_states is None:
            return None

        n_s = self._n_states
        self._q_net.eval()
        self._ev_net.eval()

        with torch.no_grad():
            unique_states = torch.arange(n_s, dtype=torch.long)
            ctx_default = torch.zeros(n_s, dtype=torch.long)

            s_feat = self._state_encoder(unique_states)
            ctx_feat = self._context_encoder(ctx_default)

            q_all = self._q_net.all_actions(s_feat, ctx_feat, self.n_actions)
            r_all = torch.zeros(n_s, self.n_actions)
            for a_idx in range(self.n_actions):
                a_oh = F.one_hot(
                    torch.full((n_s,), a_idx, dtype=torch.long),
                    self.n_actions,
                ).float()
                ev_a = self._ev_net(s_feat, ctx_feat, a_oh)
                r_all[:, a_idx] = q_all[:, a_idx] - self.discount * ev_a

        return r_all.numpy()

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
            Each row sums to 1.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self.policy_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        states = np.asarray(states, dtype=np.int64)
        return self.policy_[states]

    def predict_reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        contexts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict implied rewards r(s,a,ctx) = Q(s,a,ctx) - beta*EV(s,a,ctx).

        Parameters
        ----------
        states : torch.Tensor
            State indices of shape (N,).
        actions : torch.Tensor
            Action indices of shape (N,).
        contexts : torch.Tensor, optional
            Context indices of shape (N,). If None, uses zeros.

        Returns
        -------
        torch.Tensor
            Implied rewards of shape (N,).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        """
        if self._q_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if contexts is None:
            contexts = torch.zeros(len(states), dtype=torch.long)

        self._q_net.eval()
        self._ev_net.eval()

        with torch.no_grad():
            s_feat = self._state_encoder(states)
            ctx_feat = self._context_encoder(contexts)
            a_oh = F.one_hot(actions.long(), self.n_actions).float()

            q_vals = self._q_net(s_feat, ctx_feat, a_oh)
            ev_vals = self._ev_net(s_feat, ctx_feat, a_oh)
            return q_vals - self.discount * ev_vals

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def conf_int(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Compute confidence intervals for projected parameters.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level. Returns (1 - alpha) confidence intervals.

        Returns
        -------
        dict
            ``{param_name: (lower, upper)}`` confidence intervals.

        Raises
        ------
        RuntimeError
            If the model has not been fitted or no features were provided.
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
            Human-readable summary including parameter estimates,
            pseudo standard errors, and projection R-squared.
        """
        if self.policy_ is None:
            return "NeuralGLADIUS: Not fitted yet. Call fit() first."

        n_obs = None
        if self._n_states is not None and self.policy_ is not None:
            # Approximate from policy shape
            n_obs = self._n_states  # best we have without storing N

        return self._format_neural_summary(
            method_name="NeuralGLADIUS",
            params=self.params_,
            se=self.se_,
            pvalues=self.pvalues_,
            projection_r2=self.projection_r2_,
            n_observations=n_obs,
            n_epochs=self.n_epochs_,
            converged=self.converged_,
            discount=self.discount,
            scale=self.scale,
            context_dim=self._context_dim,
            extra_lines=[
                f"Q-network: {self.q_num_layers} layers x {self.q_hidden_dim} hidden",
                f"EV-network: {self.ev_num_layers} layers x {self.ev_hidden_dim} hidden",
            ],
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"NeuralGLADIUS(n_actions={self.n_actions}, "
            f"discount={self.discount}, "
            f"fitted={fitted})"
        )
