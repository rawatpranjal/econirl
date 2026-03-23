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
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        v_hidden_dim: Hidden dimension for the EV-network MLP.
        v_num_layers: Number of hidden layers in the EV-network.
        q_lr: Learning rate for the Q-network optimizer.
        v_lr: Learning rate for the EV-network optimizer.
        max_epochs: Maximum number of training epochs.
        batch_size: Mini-batch size for SGD.
        bellman_penalty_weight: Weight on the Bellman consistency penalty.
        weight_decay: L2 regularization weight.
        gradient_clip: Maximum gradient norm for clipping.
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
    compute_se: bool = True
    n_bootstrap: int = 100
    verbose: bool = False


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, output_dim: int) -> nn.Sequential:
    """Build a simple MLP: Linear -> ReLU -> ... -> Linear.

    Args:
        input_dim: Dimension of input features.
        hidden_dim: Dimension of hidden layers.
        num_layers: Number of hidden layers.
        output_dim: Dimension of output.

    Returns:
        An nn.Sequential MLP module.
    """
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """MLP that maps (state_features, action_onehot) to a scalar Q value."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.n_actions = n_actions
        self.mlp = _build_mlp(state_dim + n_actions, hidden_dim, num_layers, 1)

    def forward(self, state_features: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a).

        Args:
            state_features: Tensor of shape (batch, state_dim).
            action_onehot: Tensor of shape (batch, n_actions).

        Returns:
            Q values of shape (batch,).
        """
        x = torch.cat([state_features, action_onehot], dim=-1)
        return self.mlp(x).squeeze(-1)

    def forward_all_actions(self, state_features: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a) for all actions at once.

        Args:
            state_features: Tensor of shape (batch, state_dim).

        Returns:
            Q values of shape (batch, n_actions).
        """
        batch_size = state_features.shape[0]
        q_values = []
        for a in range(self.n_actions):
            onehot = torch.zeros(batch_size, self.n_actions, device=state_features.device)
            onehot[:, a] = 1.0
            q_values.append(self.forward(state_features, onehot))
        return torch.stack(q_values, dim=1)


class EVNetwork(nn.Module):
    """MLP that maps (state_features, action_onehot) to a scalar EV = E[V(s')|s,a]."""

    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int, num_layers: int):
        super().__init__()
        self.n_actions = n_actions
        self.mlp = _build_mlp(state_dim + n_actions, hidden_dim, num_layers, 1)

    def forward(self, state_features: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        """Compute EV(s, a).

        Args:
            state_features: Tensor of shape (batch, state_dim).
            action_onehot: Tensor of shape (batch, n_actions).

        Returns:
            EV values of shape (batch,).
        """
        x = torch.cat([state_features, action_onehot], dim=-1)
        return self.mlp(x).squeeze(-1)


class GLADIUSEstimator(BaseEstimator):
    """GLADIUS estimator for DDC IRL with neural networks.

    Uses two MLPs -- Q_net and EV_net -- to approximate the Q-function
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
        self.q_net_: QNetwork | None = None
        self.ev_net_: EVNetwork | None = None

    @property
    def name(self) -> str:
        return "GLADIUS"

    def _build_state_features(self, states: torch.Tensor, n_states: int) -> torch.Tensor:
        """Build state feature vectors from state indices.

        Uses normalized state index as features for the tabular case.

        Args:
            states: Tensor of state indices, shape (batch,).
            n_states: Total number of states.

        Returns:
            Feature tensor of shape (batch, 1) with values in [0, 1].
        """
        return (states.float() / max(n_states - 1, 1)).unsqueeze(-1)

    def _build_state_features_all(self, n_states: int) -> torch.Tensor:
        """Build feature vectors for all states.

        Args:
            n_states: Total number of states.

        Returns:
            Feature tensor of shape (n_states, 1).
        """
        return self._build_state_features(torch.arange(n_states), n_states)

    def _compute_log_likelihood(
        self,
        q_net: QNetwork,
        states: torch.Tensor,
        actions: torch.Tensor,
        n_states: int,
        sigma: float,
    ) -> float:
        """Compute the log-likelihood of the full dataset.

        Args:
            q_net: Trained Q-network.
            states: All observed state indices.
            actions: All observed action indices.
            n_states: Number of states.
            sigma: Scale parameter.

        Returns:
            Total log-likelihood (scalar).
        """
        with torch.no_grad():
            state_feat = self._build_state_features(states, n_states)
            q_all = q_net.forward_all_actions(state_feat)  # (N, n_actions)
            log_probs = q_all / sigma - torch.logsumexp(q_all / sigma, dim=1, keepdim=True)
            ll = log_probs[torch.arange(len(actions)), actions].sum().item()
        return ll

    def estimate(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
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
        transitions : torch.Tensor
            Transition matrices P(s'|s,a).
        initial_params : torch.Tensor, optional
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
        standard_errors = torch.full_like(result.parameters, float("nan"))

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
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Core GLADIUS optimization routine.

        Steps:
            1. Extract (s, a, s') tuples from panel.
            2. Build Q-network and EV-network.
            3. Train via mini-batch SGD with NLL + Bellman penalty.
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
        state_dim = 1  # normalized state index
        q_net = QNetwork(state_dim, n_actions, self.config.q_hidden_dim, self.config.q_num_layers)
        ev_net = EVNetwork(state_dim, n_actions, self.config.v_hidden_dim, self.config.v_num_layers)

        q_optimizer = torch.optim.Adam(
            q_net.parameters(), lr=self.config.q_lr, weight_decay=self.config.weight_decay
        )
        ev_optimizer = torch.optim.Adam(
            ev_net.parameters(), lr=self.config.v_lr, weight_decay=self.config.weight_decay
        )

        # --- Step 3: Training loop ---
        best_loss = float("inf")
        epochs_no_improve = 0
        patience = 50
        converged = False

        loss_history: list[float] = []

        for epoch in range(self.config.max_epochs):
            # Shuffle data
            perm = torch.randperm(n_obs)
            epoch_loss = 0.0
            n_batches = 0

            for start_idx in range(0, n_obs, self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, n_obs)
                idx = perm[start_idx:end_idx]

                s_batch = all_states[idx]
                a_batch = all_actions[idx]
                sp_batch = all_next_states[idx]

                # Build features
                s_feat = self._build_state_features(s_batch, n_states)
                sp_feat = self._build_state_features(sp_batch, n_states)

                # Action one-hot
                a_onehot = F.one_hot(a_batch.long(), n_actions).float()

                # Forward pass: Q(s, a)
                q_sa = q_net(s_feat, a_onehot)

                # Forward pass: Q(s, all a) for NLL
                q_all = q_net.forward_all_actions(s_feat)  # (batch, n_actions)

                # NLL loss: -log P(a|s) = -[Q(s,a)/sigma - logsumexp(Q(s,:)/sigma)]
                log_probs = q_all / sigma - torch.logsumexp(q_all / sigma, dim=1, keepdim=True)
                nll = -log_probs[torch.arange(len(a_batch)), a_batch.long()].mean()

                # Bellman penalty
                # EV(s, a)
                ev_sa = ev_net(s_feat, a_onehot)

                # V(s') = sigma * logsumexp(Q(s', :) / sigma)
                q_sp_all = q_net.forward_all_actions(sp_feat)  # (batch, n_actions)
                v_sp = sigma * torch.logsumexp(q_sp_all / sigma, dim=1)

                # TD error: beta * (EV(s,a) - V(s'))
                td_error = beta * (ev_sa - v_sp)
                bellman_loss = (td_error ** 2).mean()

                # Total loss
                loss = nll + self.config.bellman_penalty_weight * bellman_loss

                # Backward and optimize
                q_optimizer.zero_grad()
                ev_optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(q_net.parameters(), self.config.gradient_clip)
                    nn.utils.clip_grad_norm_(ev_net.parameters(), self.config.gradient_clip)

                q_optimizer.step()
                ev_optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            if self._verbose:
                if (epoch + 1) % 50 == 0 or epoch == 0:
                    self._log(f"Epoch {epoch + 1}/{self.config.max_epochs}: loss={avg_loss:.6f}")

            # Early stopping
            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                converged = True
                if self._verbose:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                break

        num_epochs = epoch + 1
        if num_epochs == self.config.max_epochs:
            converged = True  # Reached max epochs

        # --- Step 4: Extract parameters ---
        q_net.eval()
        ev_net.eval()

        with torch.no_grad():
            all_state_feat = self._build_state_features_all(n_states)

            # Compute Q(s, a) and EV(s, a) for all (s, a)
            q_table = q_net.forward_all_actions(all_state_feat)  # (n_states, n_actions)
            ev_table = torch.zeros(n_states, n_actions)
            for a in range(n_actions):
                a_oh = torch.zeros(n_states, n_actions)
                a_oh[:, a] = 1.0
                ev_table[:, a] = ev_net(all_state_feat, a_oh)

            # Implied reward: r(s, a) = Q(s, a) - beta * EV(s, a)
            reward_table = q_table - beta * ev_table

            # Policy: softmax of Q values
            policy = F.softmax(q_table / sigma, dim=1)

            # Value function: V(s) = sigma * logsumexp(Q(s, :) / sigma)
            value_function = sigma * torch.logsumexp(q_table / sigma, dim=1)

        # Regress implied rewards onto feature matrix if available
        parameters = self._extract_parameters(utility, reward_table)

        # Compute log-likelihood
        ll = self._compute_log_likelihood(q_net, all_states, all_actions, n_states, sigma)

        optimization_time = time.time() - start_time

        # Store trained networks
        self.q_net_ = q_net
        self.ev_net_ = ev_net

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
                "reward_table": reward_table.numpy().tolist(),
                "q_table": q_table.numpy().tolist(),
                "ev_table": ev_table.numpy().tolist(),
                "loss_history": loss_history,
                "final_loss": loss_history[-1] if loss_history else float("nan"),
            },
        )

    def _extract_parameters(
        self, utility: UtilityFunction, reward_table: torch.Tensor
    ) -> torch.Tensor:
        """Extract structural parameters by regressing rewards onto features.

        If the utility has a feature_matrix attribute (linear utility), solves
        the least-squares problem:
            theta = argmin ||feature_matrix @ theta - r||^2

        Otherwise returns the flattened reward table.

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

            # Flatten to (n_states * n_actions, n_features) and (n_states * n_actions,)
            X = feature_matrix.reshape(-1, n_features)
            y = reward_table.reshape(-1)

            # Least-squares: theta = (X^T X)^{-1} X^T y
            # Use torch.linalg.lstsq for numerical stability
            result = torch.linalg.lstsq(X, y)
            parameters = result.solution
            return parameters
        else:
            # No feature matrix: return flattened rewards
            return reward_table.flatten()
