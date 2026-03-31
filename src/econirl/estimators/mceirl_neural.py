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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.reward_spec import RewardSpec
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel, TrajectoryPanel
from econirl.estimators.neural_base import NeuralEstimatorMixin


# ---------------------------------------------------------------------------
# Internal network modules
# ---------------------------------------------------------------------------


class _StateRewardNetwork(nn.Module):
    """R(s) reward network.

    Input: state features of shape (B, state_dim).
    Output: scalar reward of shape (B,).
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state_feat: torch.Tensor) -> torch.Tensor:
        """Compute R(s).

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).

        Returns
        -------
        torch.Tensor
            Rewards of shape (B,).
        """
        return self.net(state_feat).squeeze(-1)


class _StateActionRewardNetwork(nn.Module):
    """R(s,a) reward network.

    Input: concatenation of state features (B, state_dim) and action
    one-hot encoding (B, n_actions).
    Output: scalar reward of shape (B,).
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int,
        num_layers: int,
    ):
        super().__init__()
        self._n_actions = n_actions
        input_dim = state_dim + n_actions
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self, state_feat: torch.Tensor, action_onehot: torch.Tensor
    ) -> torch.Tensor:
        """Compute R(s,a).

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (B, state_dim).
        action_onehot : torch.Tensor
            One-hot action encoding of shape (B, n_actions).

        Returns
        -------
        torch.Tensor
            Rewards of shape (B,).
        """
        x = torch.cat([state_feat, action_onehot], dim=-1)
        return self.net(x).squeeze(-1)

    def all_actions(self, state_feat: torch.Tensor) -> torch.Tensor:
        """Compute R(s,a) for all actions at every state.

        Parameters
        ----------
        state_feat : torch.Tensor
            State features of shape (S, state_dim).

        Returns
        -------
        torch.Tensor
            Reward matrix of shape (S, A).
        """
        S = state_feat.shape[0]
        A = self._n_actions
        # One-hot action identities: (A, A)
        eye = torch.eye(A, device=state_feat.device, dtype=state_feat.dtype)
        # Expand state_feat to (S, A, state_dim)
        sf = state_feat.unsqueeze(1).expand(-1, A, -1)
        # Expand actions to (S, A, A)
        act = eye.unsqueeze(0).expand(S, -1, -1)
        # Concatenate along last dim -> (S, A, state_dim + A)
        x = torch.cat([sf, act], dim=-1)
        # Flatten, forward, reshape
        return self.net(x.reshape(S * A, -1)).reshape(S, A)


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
        Function mapping state indices (long tensor) to feature vectors.
        Receives shape (B,) and should return shape (B, state_dim).
        If None, a default normalizing encoder is created.
    state_dim : int, optional
        Dimension of state features.  Required if state_encoder is provided.
    feature_names : list of str, optional
        Names for features when projecting rewards onto linear features.
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
        state_encoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        state_dim: int | None = None,
        # Projection
        feature_names: list[str] | None = None,
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

        # Internal state
        self._reward_net: nn.Module | None = None
        self._state_encoder: Callable | None = None
        self._state_dim: int | None = None
        self._n_states: int | None = None
        self._n_actions: int | None = None

    def fit(
        self,
        data: pd.DataFrame | Panel | TrajectoryPanel,
        state: str | None = None,
        action: str | None = None,
        id: str | None = None,
        features: RewardSpec | torch.Tensor | None = None,
        transitions: torch.Tensor | np.ndarray | None = None,
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
        features : RewardSpec or torch.Tensor, optional
            Feature specification for parameter projection.  If provided,
            the neural reward is projected onto these features to extract
            interpretable theta.
        transitions : torch.Tensor or numpy.ndarray
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
                "Pass transitions as (n_actions, n_states, n_states) tensor."
            )

        # --- Step 1: Extract tensors from data ---
        all_states, all_actions, all_next = self._extract_data(
            data, state, action, id
        )

        n_states = self.n_states or int(all_states.max().item()) + 1
        n_actions = self.n_actions or int(all_actions.max().item()) + 1
        self._n_states = n_states
        self._n_actions = n_actions

        # Convert transitions to tensor
        if isinstance(transitions, np.ndarray):
            transitions_t = torch.tensor(transitions, dtype=torch.float32)
        else:
            transitions_t = transitions.float()

        # --- Step 2: Build encoder ---
        self._build_encoder(n_states)

        # --- Step 3: Compute empirical state-action occupancy ---
        empirical_sa = self._compute_empirical_occupancy(
            all_states, all_actions, n_states, n_actions
        )

        # --- Step 4: Build reward network ---
        if self.reward_type == "state_action":
            self._reward_net = _StateActionRewardNetwork(
                self._state_dim,
                n_actions,
                self.reward_hidden_dim,
                self.reward_num_layers,
            )
        else:
            self._reward_net = _StateRewardNetwork(
                self._state_dim,
                self.reward_hidden_dim,
                self.reward_num_layers,
            )

        # --- Step 5: Training loop ---
        self._train_mce(
            transitions_t, empirical_sa, n_states, n_actions,
        )

        # --- Step 6: Extract policy, value, and reward ---
        self._extract_final(transitions_t, n_states, n_actions)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract state/action/next_state tensors from input data."""
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
        elif isinstance(data, (Panel, TrajectoryPanel)):
            all_states = data.get_all_states()
            all_actions = data.get_all_actions()
            all_next = data.get_all_next_states()
        else:
            raise TypeError(
                f"data must be a DataFrame, Panel, or TrajectoryPanel, "
                f"got {type(data)}"
            )

        return all_states, all_actions, all_next

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
            self._state_encoder = lambda s, _ms=max_s: (
                s.float() / _ms
            ).unsqueeze(-1)
            self._state_dim = 1

    # ------------------------------------------------------------------
    # Empirical occupancy
    # ------------------------------------------------------------------

    def _compute_empirical_occupancy(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        n_states: int,
        n_actions: int,
    ) -> torch.Tensor:
        """Compute empirical state-action occupancy from demonstrations.

        Returns
        -------
        torch.Tensor
            State-action occupancy of shape (n_states, n_actions).
            Normalized to sum to 1.
        """
        sa_counts = torch.zeros(n_states, n_actions, dtype=torch.float32)
        for s, a in zip(states, actions):
            sa_counts[s.long(), a.long()] += 1
        # Normalize
        total = sa_counts.sum()
        if total > 0:
            sa_counts = sa_counts / total
        return sa_counts

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train_mce(
        self,
        transitions: torch.Tensor,
        empirical_sa: torch.Tensor,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Run MCE-IRL training with neural reward network.

        Training loop:
        1. Forward: compute R(s,a) for all states and actions
           - state_action: reward_net.all_actions(state_feat)
           - state: reward_net(state_feat), broadcast to all actions
        2. Solve soft Bellman: V, policy = soft_value_iteration(R, transitions)
        3. Compute state visitation: D(s) = forward_pass(policy, transitions)
        4. Expected occupancy: E_policy[sa] = D(s) * pi(a|s)
        5. Loss: -sum(empirical_sa * R_sa) + sum(policy_sa * R_sa)
        6. Backprop through reward network
        """
        optimizer = torch.optim.Adam(
            self._reward_net.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
        )
        # LR scheduler for training stability
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=50, factor=0.5
        )

        problem = DDCProblem(
            num_states=n_states,
            num_actions=n_actions,
            discount_factor=self.discount,
            scale_parameter=1.0,
        )
        bellman = SoftBellmanOperator(problem=problem, transitions=transitions)

        best_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        patience = 100

        all_state_indices = torch.arange(n_states)

        for epoch in range(self.max_epochs):
            # 1. Compute reward matrix R(s,a)
            state_feat = self._state_encoder(all_state_indices)

            if self.reward_type == "state_action":
                reward_matrix = self._reward_net.all_actions(
                    state_feat
                )  # (S, A)
            else:
                rewards_s = self._reward_net(state_feat)  # (S,)
                reward_matrix = rewards_s.unsqueeze(1).expand(
                    -1, n_actions
                )  # (S, A)

            # 2. Solve soft Bellman (detach -- no grad through VI)
            with torch.no_grad():
                if self.inner_solver == "hybrid":
                    result = hybrid_iteration(
                        bellman,
                        reward_matrix.detach(),
                        tol=self.inner_tol,
                        max_iter=self.inner_max_iter,
                    )
                else:
                    result = value_iteration(
                        bellman,
                        reward_matrix.detach(),
                        tol=self.inner_tol,
                        max_iter=self.inner_max_iter,
                    )
                policy = result.policy  # (S, A)

            # 3. Compute state visitation frequencies via forward pass
            with torch.no_grad():
                D = self._forward_pass(
                    policy, transitions, n_states, self.discount
                )

            # 4. State-action occupancy under current policy
            policy_sa = D.unsqueeze(1) * policy  # (S, A) = D(s) * pi(a|s)

            # 5. Feature matching loss on (s,a) occupancies:
            #    L = -sum_{s,a} mu_D(s,a) * R(s,a)
            #    Gradient w.r.t. R(s,a): policy_sa - empirical_sa
            #    Per Wulfmeier et al. (2016) Algorithm 1 Eq. 11:
            #    Apply this gradient directly to the reward network via
            #    backprop, rather than optimizing a scalar loss (which
            #    causes reward values to grow unboundedly).
            grad_r = policy_sa - empirical_sa  # (S, A)

            optimizer.zero_grad()
            reward_matrix.backward(gradient=grad_r)

            # Gradient clipping
            nn.utils.clip_grad_norm_(self._reward_net.parameters(), 1.0)

            optimizer.step()

            # Monitor feature matching residual as the "loss"
            loss_val = torch.sum(grad_r ** 2).item()
            scheduler.step(loss_val)

            # Monitor feature matching gap
            feature_diff = torch.norm(empirical_sa - policy_sa).item()

            if self.verbose and (epoch + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch {epoch + 1}: loss={loss_val:.4f}, "
                    f"feature_diff={feature_diff:.6f}, lr={current_lr:.2e}"
                )

            # Early stopping with best model checkpoint
            if loss_val < best_loss - 1e-5:
                best_loss = loss_val
                best_state_dict = {
                    k: v.clone() for k, v in self._reward_net.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Restore best model
        if best_state_dict is not None:
            self._reward_net.load_state_dict(best_state_dict)

        self.converged_ = patience_counter >= patience or epoch == self.max_epochs - 1
        self.n_epochs_ = epoch + 1

    def _forward_pass(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        n_states: int,
        discount: float,
    ) -> torch.Tensor:
        """Compute state visitation frequencies via forward message passing.

        D(s) = rho_0(s) + gamma * sum_{s',a} D(s') pi(a|s') P(s|s',a)

        Parameters
        ----------
        policy : torch.Tensor
            Policy pi(a|s), shape (n_states, n_actions).
        transitions : torch.Tensor
            Transition matrices P(s'|s,a), shape (n_actions, n_states, n_states).
        n_states : int
            Number of states.
        discount : float
            Discount factor.

        Returns
        -------
        torch.Tensor
            State visitation frequencies, shape (n_states,).
        """
        # Initial uniform distribution
        rho0 = torch.ones(n_states, dtype=torch.float32) / n_states
        D = rho0.clone()

        # Policy-weighted transition: P_pi(s'|s) = sum_a pi(a|s) P(s'|s,a)
        P_pi = torch.einsum("sa,ast->st", policy, transitions)

        for _ in range(500):
            D_new = rho0 + discount * (P_pi.T @ D)
            delta = torch.abs(D_new - D).max().item()
            D = D_new
            if delta < 1e-8:
                break

        # Normalize
        D = D / D.sum()
        return D

    # ------------------------------------------------------------------
    # Post-training extraction
    # ------------------------------------------------------------------

    def _extract_final(
        self,
        transitions: torch.Tensor,
        n_states: int,
        n_actions: int,
    ) -> None:
        """Extract policy, value function, and reward from trained network."""
        self._reward_net.eval()

        with torch.no_grad():
            all_state_indices = torch.arange(n_states)
            state_feat = self._state_encoder(all_state_indices)

            if self.reward_type == "state_action":
                reward_matrix = self._reward_net.all_actions(
                    state_feat
                )  # (S, A)
            else:
                rewards_s = self._reward_net(state_feat)  # (S,)
                reward_matrix = rewards_s.unsqueeze(1).expand(
                    -1, n_actions
                )  # (S, A)

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

            self.policy_ = result.policy.numpy()
            self.value_ = result.V.numpy()
            if self.reward_type == "state_action":
                self.reward_ = reward_matrix.numpy()  # (S, A)
            else:
                self.reward_ = rewards_s.numpy()  # (S,)

    def _project_onto_features(
        self,
        features: RewardSpec | torch.Tensor,
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
        features : RewardSpec or torch.Tensor
            Feature specification.  RewardSpec provides (S, A, K) matrix.
            A Tensor of shape (S, K) or (S, A, K) is also accepted.
        n_states : int
            Number of states.
        n_actions : int
            Number of actions.
        """
        # Extract feature matrix and names
        if isinstance(features, RewardSpec):
            feat_3d = features.feature_matrix  # (S, A, K)
            names = features.parameter_names
        elif features.ndim == 3:
            feat_3d = features  # (S, A, K)
            names = self.feature_names or [
                f"f{i}" for i in range(features.shape[-1])
            ]
        elif features.ndim == 2:
            # (S, K) -> broadcast to (S, A, K)
            feat_3d = features.unsqueeze(1).expand(-1, n_actions, -1)
            names = self.feature_names or [
                f"f{i}" for i in range(features.shape[-1])
            ]
        else:
            raise ValueError(
                f"features must be 2D (S, K) or 3D (S, A, K), "
                f"got {features.ndim}D"
            )

        rewards = torch.tensor(self.reward_, dtype=torch.float32)

        if self.reward_type == "state_action":
            # R(s,a) is (S, A) -- project onto flattened (S*A, K)
            phi = feat_3d.reshape(-1, feat_3d.shape[-1]).float()  # (S*A, K)
            r_flat = rewards.reshape(-1)  # (S*A,)
        else:
            # R(s) is (S,) -- project onto state features from action 0
            phi = feat_3d[:, 0, :].float()  # (S, K)
            r_flat = rewards  # (S,)

        theta, se, r2 = self._project_parameters(phi, r_flat)

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
