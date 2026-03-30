"""NeuralAIRL: Context-aware Adversarial Inverse Reinforcement Learning.

Learns a disentangled reward r(s,a,ctx) and shaping potential h(s,ctx) via
adversarial training against a learned policy network, then extracts
structural parameters by projecting implied rewards onto features.

No transition matrix needed. Supports context conditioning (destination,
time-of-day, user segment, etc.) through pluggable encoders.

Algorithm:
    1. Parameterize reward g(s,a,ctx), shaping h(s,ctx), policy pi(a|s,ctx)
       with MLPs.
    2. Construct discriminator:
       D(s,a,s',ctx) = sigmoid(f(s,a,s',ctx) - log pi(a|s,ctx))
       where f(s,a,s',ctx) = g(s,a,ctx) + gamma*h(s',ctx) - h(s,ctx).
    3. Train discriminator to classify expert vs policy-generated transitions
       using binary cross-entropy.
    4. Train policy to fool the discriminator (maximize log D under policy).
    5. After training, implied rewards r(s,a,ctx) = g(s,a,ctx) are projected
       onto features via least-squares to recover theta.

Reference:
    Fu, J., Luo, K., & Levine, S. (2018). Learning robust rewards with
    adversarial inverse reinforcement learning. ICLR.
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


class _RewardNetwork(nn.Module):
    """g(s, a, ctx) reward network.

    Input: concatenation of [state_features, context_features, action_onehot].
    Output: scalar reward value.
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
        """Compute g(s, a, ctx).

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
            Reward values of shape (B,).
        """
        x = torch.cat([state_feat, ctx_feat, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)

    def all_actions(
        self,
        state_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        n_actions: int,
    ) -> torch.Tensor:
        """Compute g for all actions at once.

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
            Reward values of shape (B, n_actions).
        """
        B = state_feat.shape[0]
        actions = torch.eye(n_actions, device=state_feat.device)  # (A, A)
        actions = actions.unsqueeze(0).expand(B, -1, -1)  # (B, A, A)
        sf = state_feat.unsqueeze(1).expand(-1, n_actions, -1)  # (B, A, d_s)
        cf = ctx_feat.unsqueeze(1).expand(-1, n_actions, -1)  # (B, A, d_c)
        x = torch.cat([sf, cf, actions], dim=-1)  # (B, A, input_dim)
        return self.net(x.reshape(B * n_actions, -1)).reshape(B, n_actions)


class _ShapingNetwork(nn.Module):
    """h(s, ctx) potential-based shaping network.

    Input: concatenation of [state_features, context_features].
    Output: scalar shaping potential.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        input_dim = state_dim + context_dim
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
    ) -> torch.Tensor:
        """Compute h(s, ctx).

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).
        ctx_feat : torch.Tensor
            Context features of shape (B, context_dim).

        Returns
        -------
        torch.Tensor
            Shaping values of shape (B,).
        """
        x = torch.cat([state_feat, ctx_feat], dim=-1)
        return self.net(x).squeeze(-1)


class _PolicyNetwork(nn.Module):
    """pi(a|s, ctx) policy network.

    Input: concatenation of [state_features, context_features].
    Output: softmax probability over n_actions.
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
        input_dim = state_dim + context_dim
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute pi(a|s, ctx) as softmax probabilities.

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).
        ctx_feat : torch.Tensor
            Context features of shape (B, context_dim).

        Returns
        -------
        torch.Tensor
            Action probabilities of shape (B, n_actions).
        """
        x = torch.cat([state_feat, ctx_feat], dim=-1)
        logits = self.net(x)
        return torch.softmax(logits, dim=-1)

    def log_prob(
        self,
        state_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log pi(a|s, ctx) for given actions.

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).
        ctx_feat : torch.Tensor
            Context features of shape (B, context_dim).
        actions : torch.Tensor
            Action indices of shape (B,).

        Returns
        -------
        torch.Tensor
            Log probabilities of shape (B,).
        """
        x = torch.cat([state_feat, ctx_feat], dim=-1)
        logits = self.net(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs[torch.arange(len(actions)), actions.long()]


# ---------------------------------------------------------------------------
# NeuralAIRL estimator
# ---------------------------------------------------------------------------


class NeuralAIRL(NeuralEstimatorMixin):
    """Context-aware AIRL estimator with sklearn-style API.

    Trains a reward network g(s,a,ctx), shaping network h(s,ctx), and
    policy network pi(a|s,ctx) via adversarial training. The discriminator
    classifies expert vs policy-generated transitions. After training,
    implied rewards g(s,a,ctx) are projected onto features via least-squares
    to get interpretable theta.

    No transition matrix required. Supports context conditioning through
    pluggable state/context encoders.

    Parameters
    ----------
    n_actions : int, default=8
        Number of discrete actions.
    discount : float, default=0.95
        Time discount factor gamma.
    reward_hidden_dim : int, default=128
        Hidden dimension for the reward network MLP.
    reward_num_layers : int, default=3
        Number of hidden layers in the reward network.
    shaping_hidden_dim : int, default=128
        Hidden dimension for the shaping network MLP.
    shaping_num_layers : int, default=3
        Number of hidden layers in the shaping network.
    policy_hidden_dim : int, default=128
        Hidden dimension for the policy network MLP.
    policy_num_layers : int, default=3
        Number of hidden layers in the policy network.
    batch_size : int, default=512
        Mini-batch size for SGD.
    max_epochs : int, default=500
        Maximum number of training epochs.
    disc_lr : float, default=1e-3
        Learning rate for discriminator (reward + shaping) optimizer.
    policy_lr : float, default=1e-3
        Learning rate for policy optimizer.
    disc_steps : int, default=5
        Discriminator updates per policy update.
    gradient_clip : float, default=1.0
        Maximum gradient norm for clipping. 0 disables clipping.
    patience : int, default=50
        Early stopping patience (epochs without improvement).
    label_smoothing : float, default=0.0
        Label smoothing for discriminator BCE loss. 0 means no smoothing.
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
    >>> from econirl.estimators import NeuralAIRL
    >>> import pandas as pd
    >>>
    >>> model = NeuralAIRL(n_actions=3, discount=0.95, max_epochs=200)
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
        state_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        context_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
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
        self._reward_net: _RewardNetwork | None = None
        self._shaping_net: _ShapingNetwork | None = None
        self._policy_net: _PolicyNetwork | None = None
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
    ) -> "NeuralAIRL":
        """Fit the NeuralAIRL estimator to data.

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
            Accepted for API compatibility but not used. NeuralAIRL does
            not require a transition matrix.

        Returns
        -------
        self : NeuralAIRL
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
        self._reward_net = _RewardNetwork(
            sd, cd, self.n_actions, self.reward_hidden_dim, self.reward_num_layers
        )
        self._shaping_net = _ShapingNetwork(
            sd, cd, self.shaping_hidden_dim, self.shaping_num_layers
        )
        self._policy_net = _PolicyNetwork(
            sd, cd, self.n_actions, self.policy_hidden_dim, self.policy_num_layers
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
    # Data extraction (same as NeuralGLADIUS)
    # ------------------------------------------------------------------

    def _extract_data(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None,
        action: str | None,
        id: str | None,
        context: str | torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract state/action/next_state/context tensors from input data."""
        if isinstance(data, pd.DataFrame):
            if state is None or action is None or id is None:
                raise ValueError(
                    "state, action, and id column names are required "
                    "when data is a DataFrame"
                )
            panel = TrajectoryPanel.from_dataframe(
                data, state=state, action=action, id=id
            )
            all_states = panel.all_states
            all_actions = panel.all_actions
            all_next = panel.all_next_states

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
            all_states = data.get_all_states()
            all_actions = data.get_all_actions()
            all_next = data.get_all_next_states()

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
        """Extract context values from DataFrame aligned with panel observations."""
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
        """Run the adversarial training loop.

        Alternates between:
        1. Discriminator update: classify expert vs policy transitions
        2. Policy update: fool the discriminator

        Uses early stopping on discriminator loss.
        """
        disc_optimizer = torch.optim.Adam(
            list(self._reward_net.parameters())
            + list(self._shaping_net.parameters()),
            lr=self.disc_lr,
            weight_decay=1e-4,
        )
        policy_optimizer = torch.optim.Adam(
            self._policy_net.parameters(),
            lr=self.policy_lr,
            weight_decay=1e-4,
        )

        N = len(states)
        best_loss = float("inf")
        patience_counter = 0

        # Labels with optional smoothing
        expert_label = 1.0 - self.label_smoothing
        policy_label = 0.0 + self.label_smoothing

        for epoch in range(self.max_epochs):
            perm = torch.randperm(N)
            epoch_disc_loss = 0.0
            epoch_policy_loss = 0.0
            n_batches = 0

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

                # --- Discriminator update ---
                for _ in range(self.disc_steps):
                    disc_optimizer.zero_grad()

                    # Expert transitions: compute discriminator logits
                    expert_logits = self._compute_disc_logits(
                        s_feat, ctx_feat, a_oh, a, ns_feat
                    )

                    # Policy transitions: sample actions from policy,
                    # use same (s, s') but with policy-sampled actions
                    with torch.no_grad():
                        policy_probs = self._policy_net(s_feat, ctx_feat)
                        policy_actions = torch.multinomial(
                            policy_probs, 1
                        ).squeeze(-1)
                        policy_a_oh = F.one_hot(
                            policy_actions.long(), self.n_actions
                        ).float()

                    policy_logits = self._compute_disc_logits(
                        s_feat, ctx_feat, policy_a_oh, policy_actions, ns_feat
                    )

                    # BCE loss: expert = 1, policy = 0
                    expert_targets = torch.full_like(
                        expert_logits, expert_label
                    )
                    policy_targets = torch.full_like(
                        policy_logits, policy_label
                    )
                    disc_loss = (
                        F.binary_cross_entropy_with_logits(
                            expert_logits, expert_targets
                        )
                        + F.binary_cross_entropy_with_logits(
                            policy_logits, policy_targets
                        )
                    )

                    disc_loss.backward()
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(
                            list(self._reward_net.parameters())
                            + list(self._shaping_net.parameters()),
                            self.gradient_clip,
                        )
                    disc_optimizer.step()

                # --- Policy update ---
                policy_optimizer.zero_grad()

                # Policy wants to maximize log D(s, a_pi, s')
                # which means maximizing the discriminator logits for
                # policy-sampled actions
                policy_probs = self._policy_net(s_feat, ctx_feat)
                policy_actions = torch.multinomial(
                    policy_probs, 1
                ).squeeze(-1)
                policy_a_oh = F.one_hot(
                    policy_actions.long(), self.n_actions
                ).float()

                # Use log pi for REINFORCE-style gradient
                log_pi = self._policy_net.log_prob(
                    s_feat, ctx_feat, policy_actions
                )

                # Reward signal from discriminator
                with torch.no_grad():
                    disc_logits = self._compute_disc_logits(
                        s_feat, ctx_feat, policy_a_oh, policy_actions, ns_feat
                    )
                    # log D(s,a,s') = logits - log(1 + exp(logits))
                    # = -softplus(-logits)
                    disc_reward = -F.softplus(-disc_logits)

                # Policy loss: -E[log pi(a|s) * reward]
                # Also add entropy bonus for exploration
                entropy = -(policy_probs * (policy_probs + 1e-10).log()).sum(
                    dim=-1
                )
                policy_loss = -(log_pi * disc_reward).mean() - 0.01 * entropy.mean()

                policy_loss.backward()
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(
                        self._policy_net.parameters(),
                        self.gradient_clip,
                    )
                policy_optimizer.step()

                epoch_disc_loss += disc_loss.item()
                epoch_policy_loss += policy_loss.item()
                n_batches += 1

            avg_disc_loss = epoch_disc_loss / max(n_batches, 1)
            avg_policy_loss = epoch_policy_loss / max(n_batches, 1)

            if self.verbose and (epoch + 1) % 50 == 0:
                print(
                    f"  Epoch {epoch + 1}: disc_loss={avg_disc_loss:.4f} "
                    f"policy_loss={avg_policy_loss:.4f}"
                )

            # Early stopping on discriminator loss
            if avg_disc_loss < best_loss - 1e-4:
                best_loss = avg_disc_loss
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

    def _compute_disc_logits(
        self,
        s_feat: torch.Tensor,
        ctx_feat: torch.Tensor,
        a_oh: torch.Tensor,
        actions: torch.Tensor,
        ns_feat: torch.Tensor,
    ) -> torch.Tensor:
        """Compute AIRL discriminator logits.

        logit = f(s,a,s') - log pi(a|s)
        where f(s,a,s') = g(s,a) + gamma*h(s') - h(s)

        Parameters
        ----------
        s_feat : torch.Tensor
            State features of shape (B, state_dim).
        ctx_feat : torch.Tensor
            Context features of shape (B, context_dim).
        a_oh : torch.Tensor
            One-hot actions of shape (B, n_actions).
        actions : torch.Tensor
            Action indices of shape (B,).
        ns_feat : torch.Tensor
            Next-state features of shape (B, state_dim).

        Returns
        -------
        torch.Tensor
            Discriminator logits of shape (B,).
        """
        g = self._reward_net(s_feat, ctx_feat, a_oh)
        h_s = self._shaping_net(s_feat, ctx_feat)
        h_ns = self._shaping_net(ns_feat, ctx_feat)
        f = g + self.discount * h_ns - h_s

        log_pi = self._policy_net.log_prob(s_feat, ctx_feat, actions)

        return f - log_pi

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

        Evaluates at context=0 for the policy/value matrices.
        """
        self._reward_net.eval()
        self._shaping_net.eval()
        self._policy_net.eval()

        with torch.no_grad():
            unique_states = torch.arange(n_states)
            ctx_default = torch.zeros(n_states, dtype=torch.long)

            s_feat = self._state_encoder(unique_states)
            ctx_feat = self._context_encoder(ctx_default)

            # Policy from the policy network
            policy = self._policy_net(s_feat, ctx_feat)

            # Value from shaping network
            value = self._shaping_net(s_feat, ctx_feat)

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

        with torch.no_grad():
            s_feat = self._state_encoder(states)
            ctx_feat = self._context_encoder(contexts)
            a_oh = F.one_hot(actions.long(), self.n_actions).float()

            # Implied rewards are the reward network output g(s,a,ctx)
            rewards = self._reward_net(s_feat, ctx_feat, a_oh)

        # Get features for observed (s, a) pairs
        phi = feat_matrix[states.long(), actions.long(), :]  # (N, K)

        # Use float32 for projection
        phi = phi.float()
        rewards = rewards.float()

        theta, se, r2 = self._project_parameters(phi, rewards)

        self.params_ = {n: float(v) for n, v in zip(names, theta)}
        self.se_ = {n: float(v) for n, v in zip(names, se)}
        self.pvalues_ = self._compute_pvalues(self.params_, self.se_)
        self.projection_r2_ = r2
        self.coef_ = theta.numpy()

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------

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
        """Predict implied rewards g(s,a,ctx) from the reward network.

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
        if self._reward_net is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        if contexts is None:
            contexts = torch.zeros(len(states), dtype=torch.long)

        self._reward_net.eval()

        with torch.no_grad():
            s_feat = self._state_encoder(states)
            ctx_feat = self._context_encoder(contexts)
            a_oh = F.one_hot(actions.long(), self.n_actions).float()

            return self._reward_net(s_feat, ctx_feat, a_oh)

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

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self.policy_ is not None
        return (
            f"NeuralAIRL(n_actions={self.n_actions}, "
            f"discount={self.discount}, "
            f"fitted={fitted})"
        )
