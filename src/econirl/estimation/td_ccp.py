"""TD-CCP Neural Estimator for dynamic discrete choice models.

This module implements a Temporal Difference CCP estimator that uses
neural networks to approximate the expected value function components.
The approach combines CCP estimation with approximate value iteration (AVI)
using semi-gradient TD learning.

Algorithm:
1. Estimate CCPs from data using frequency estimator
2. Decompose expected value into per-feature flow components plus entropy
3. Train component neural networks via AVI with semi-gradient TD
4. Use learned EV components for partial MLE of structural parameters

This method is particularly useful for large state spaces where
matrix-inversion-based CCP methods become computationally infeasible.

References:
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Sutton, R.S. and Barto, A.G. (2018). "Reinforcement Learning:
        An Introduction." MIT Press.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from scipy import optimize

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


@dataclass
class TDCCPConfig:
    """Configuration for the TD-CCP Neural Estimator.

    Attributes:
        hidden_dim: Number of units in each hidden layer of the MLP.
        num_hidden_layers: Number of hidden layers in the MLP.
        avi_iterations: Number of approximate value iteration rounds.
        epochs_per_avi: Number of SGD epochs per AVI iteration.
        learning_rate: Learning rate for NN training.
        batch_size: Mini-batch size for SGD training.
        ccp_smoothing: Smoothing constant added to CCP frequency counts
            to avoid log(0).
        outer_max_iter: Maximum iterations for the L-BFGS-B optimizer
            in the partial MLE step.
        outer_tol: Gradient tolerance for L-BFGS-B convergence.
        compute_se: Whether to compute standard errors via numerical Hessian.
        verbose: Whether to print progress messages.
    """

    hidden_dim: int = 64
    num_hidden_layers: int = 2
    avi_iterations: int = 20
    epochs_per_avi: int = 30
    learning_rate: float = 1e-3
    batch_size: int = 8192
    ccp_smoothing: float = 0.01
    outer_max_iter: int = 200
    outer_tol: float = 1e-6
    compute_se: bool = True
    verbose: bool = False


class _EVComponentNetwork(nn.Module):
    """MLP for approximating a single EV component function.

    Maps normalized state features to a scalar value representing
    one component of the expected value decomposition.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_hidden_layers: int):
        super().__init__()
        layers: list[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer (scalar)
        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (batch, input_dim).

        Returns:
            Scalar predictions of shape (batch, 1).
        """
        return self.network(x)


class TDCCPEstimator(BaseEstimator):
    """TD-CCP Neural Estimator for dynamic discrete choice models.

    Combines CCP estimation with neural network-based approximate value
    iteration. The expected value function is decomposed into per-feature
    components, each learned by a separate MLP using semi-gradient TD.

    The partial MLE step then optimizes structural parameters using
    the learned EV components, avoiding the costly inner fixed-point
    loop required by NFXP.

    Example:
        >>> config = TDCCPConfig(avi_iterations=20, epochs_per_avi=30)
        >>> estimator = TDCCPEstimator(config=config)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        config: TDCCPConfig | None = None,
        se_method: SEMethod = "asymptotic",
        **kwargs,
    ):
        """Initialize the TD-CCP estimator.

        Args:
            config: Configuration dataclass. Uses defaults if None.
            se_method: Method for computing standard errors.
            **kwargs: Additional keyword arguments (ignored for compatibility).
        """
        if config is None:
            config = TDCCPConfig()
        self._config = config

        super().__init__(
            se_method=se_method,
            compute_hessian=config.compute_se,
            verbose=config.verbose,
        )

    @property
    def name(self) -> str:
        """Human-readable name of the estimation method."""
        return "TD-CCP Neural"

    @property
    def config(self) -> TDCCPConfig:
        """Return the configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Step 1: CCP estimation
    # ------------------------------------------------------------------

    def _estimate_ccps(
        self,
        panel: Panel,
        num_states: int,
        num_actions: int,
    ) -> torch.Tensor:
        """Estimate CCPs from data using frequency estimator with smoothing.

        P_hat(a|s) = (N(s,a) + smoothing) / (N(s) + num_actions * smoothing)

        Args:
            panel: Panel data with observed choices.
            num_states: Number of discrete states.
            num_actions: Number of discrete actions.

        Returns:
            CCP matrix of shape (num_states, num_actions).
        """
        smoothing = self._config.ccp_smoothing
        counts = torch.zeros((num_states, num_actions), dtype=torch.float32)

        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                counts[s, a] += 1

        state_counts = counts.sum(dim=1, keepdim=True)
        ccps = (counts + smoothing) / (state_counts + num_actions * smoothing)
        return ccps

    # ------------------------------------------------------------------
    # Step 2: Flow decomposition
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_flow_components(
        ccps: torch.Tensor,
        feature_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-feature flow functions and entropy flow.

        For each feature k:
            flow_k[s] = sum_a ccps[s,a] * feature_matrix[s,a,k]

        Entropy:
            flow_H[s] = -sum_a ccps[s,a] * log(ccps[s,a])

        Args:
            ccps: CCP matrix of shape (num_states, num_actions).
            feature_matrix: Feature tensor of shape (num_states, num_actions, num_features).

        Returns:
            Tuple of (flow_features, flow_entropy) where:
            - flow_features: shape (num_states, num_features)
            - flow_entropy: shape (num_states,)
        """
        # flow_k[s] = sum_a ccps[s,a] * features[s,a,k]
        flow_features = torch.einsum("sa,sak->sk", ccps, feature_matrix)

        # Entropy: -sum_a p(a|s) log p(a|s)
        safe_ccps = torch.clamp(ccps, min=1e-10)
        flow_entropy = -(ccps * torch.log(safe_ccps)).sum(dim=1)

        return flow_features, flow_entropy

    # ------------------------------------------------------------------
    # Step 3: Train component networks via AVI with semi-gradient TD
    # ------------------------------------------------------------------

    def _build_state_features(self, states: torch.Tensor, num_states: int) -> torch.Tensor:
        """Create normalized features for NN input from state indices.

        Maps integer state indices to a feature vector suitable for
        neural network input. Currently uses a single normalized
        feature: state / (num_states - 1).

        Args:
            states: Tensor of integer state indices.
            num_states: Total number of states (for normalization).

        Returns:
            Feature tensor of shape (len(states), 1).
        """
        denom = max(num_states - 1, 1)
        return (states.float() / denom).unsqueeze(-1)

    def _train_component_network(
        self,
        flow: torch.Tensor,
        states: torch.Tensor,
        next_states: torch.Tensor,
        num_states: int,
        gamma: float,
    ) -> tuple[_EVComponentNetwork, list[float]]:
        """Train a single EV component network via AVI with semi-gradient TD.

        For each AVI round:
            1. Compute targets: Y = flow[s_t] + gamma * net(features[s_{t+1}]).detach()
            2. Train net to minimize MSE between net(features[s_t]) and Y
               for epochs_per_avi epochs.

        Args:
            flow: Flow values of shape (num_states,) for this component.
            states: All observed s_t indices (flattened from panel).
            next_states: All observed s_{t+1} indices.
            num_states: Total number of states.
            gamma: Discount factor.

        Returns:
            Tuple of (trained network, list of per-epoch losses).
        """
        cfg = self._config
        input_dim = 1  # normalized state

        net = _EVComponentNetwork(input_dim, cfg.hidden_dim, cfg.num_hidden_layers)
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

        # Precompute state features for the whole dataset
        feat_s = self._build_state_features(states, num_states)
        feat_sp = self._build_state_features(next_states, num_states)
        flow_s = flow[states]  # flow values at s_t

        n_samples = len(states)
        losses: list[float] = []

        for avi_iter in range(cfg.avi_iterations):
            # Compute TD targets with current network (frozen)
            with torch.no_grad():
                v_next = net(feat_sp).squeeze(-1)
            targets = flow_s + gamma * v_next

            # Train for epochs_per_avi epochs on these targets
            for epoch in range(cfg.epochs_per_avi):
                # Shuffle indices for mini-batches
                perm = torch.randperm(n_samples)

                epoch_loss = 0.0
                n_batches = 0
                for start in range(0, n_samples, cfg.batch_size):
                    end = min(start + cfg.batch_size, n_samples)
                    idx = perm[start:end]

                    preds = net(feat_s[idx]).squeeze(-1)
                    loss = torch.nn.functional.mse_loss(preds, targets[idx])

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    n_batches += 1

                avg_loss = epoch_loss / max(n_batches, 1)
                losses.append(avg_loss)

        return net, losses

    def _train_all_components(
        self,
        flow_features: torch.Tensor,
        flow_entropy: torch.Tensor,
        panel: Panel,
        num_states: int,
        gamma: float,
    ) -> tuple[list[_EVComponentNetwork], _EVComponentNetwork, dict[str, list[float]]]:
        """Train all EV component networks (one per feature + entropy).

        Args:
            flow_features: Per-feature flow values, shape (num_states, num_features).
            flow_entropy: Entropy flow values, shape (num_states,).
            panel: Panel data (for transition samples).
            num_states: Total number of states.
            gamma: Discount factor.

        Returns:
            Tuple of (feature_nets, entropy_net, loss_histories).
        """
        states = panel.get_all_states()
        next_states = panel.get_all_next_states()
        num_features = flow_features.shape[1]

        feature_nets: list[_EVComponentNetwork] = []
        loss_histories: dict[str, list[float]] = {}

        # Train one network per feature component
        for k in range(num_features):
            self._log(f"Training EV component network for feature {k}")
            net, losses = self._train_component_network(
                flow=flow_features[:, k],
                states=states,
                next_states=next_states,
                num_states=num_states,
                gamma=gamma,
            )
            feature_nets.append(net)
            loss_histories[f"feature_{k}"] = losses

        # Train entropy network
        self._log("Training EV component network for entropy")
        entropy_net, entropy_losses = self._train_component_network(
            flow=flow_entropy,
            states=states,
            next_states=next_states,
            num_states=num_states,
            gamma=gamma,
        )
        loss_histories["entropy"] = entropy_losses

        return feature_nets, entropy_net, loss_histories

    # ------------------------------------------------------------------
    # Step 4: Partial MLE using learned EV components
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_ev_components(
        feature_nets: list[_EVComponentNetwork],
        entropy_net: _EVComponentNetwork,
        state_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate all trained component networks on a set of state features.

        Args:
            feature_nets: List of trained networks, one per utility feature.
            entropy_net: Trained entropy network.
            state_features: NN input features, shape (num_states, input_dim).

        Returns:
            Tuple of (ev_features, ev_entropy):
            - ev_features: shape (num_states, num_features)
            - ev_entropy: shape (num_states,)
        """
        with torch.no_grad():
            ev_list = [net(state_features).squeeze(-1) for net in feature_nets]
            ev_features = torch.stack(ev_list, dim=1)
            ev_entropy = entropy_net(state_features).squeeze(-1)
        return ev_features, ev_entropy

    def _partial_mle(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        ev_features: torch.Tensor,
        ev_entropy: torch.Tensor,
        initial_params: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, int, int, str]:
        """Optimize structural parameters via partial MLE.

        Using the learned EV components, construct choice-specific values
        and maximize the conditional choice log-likelihood with L-BFGS-B.

        Args:
            panel: Panel data.
            utility: Utility function specification.
            problem: Problem specification.
            transitions: Transition matrices, shape (num_actions, num_states, num_states).
            ev_features: Learned EV for each feature, shape (num_states, num_features).
            ev_entropy: Learned entropy EV, shape (num_states,).
            initial_params: Starting parameter values.

        Returns:
            Tuple of (parameters, log_likelihood, num_iterations, num_feval, message).
        """
        gamma = problem.discount_factor
        sigma = problem.scale_parameter
        num_states = problem.num_states
        num_actions = problem.num_actions
        feature_matrix = utility.feature_matrix  # (S, A, K)

        def _compute_choice_values(params: torch.Tensor) -> torch.Tensor:
            """Compute choice-specific values v(s, a) given parameters.

            v(s, a) = u(s, a; theta)
                      + gamma * sum_k theta_k * (P_a @ ev_k)
                      + gamma * (P_a @ ev_H)

            where P_a is the transition matrix for action a.
            """
            # Flow utility: u(s, a) = theta . phi(s, a)
            flow_u = torch.einsum("sak,k->sa", feature_matrix, params)

            # Continuation values per action
            # For each action a: E[ev_k | s, a] = transitions[a] @ ev_features[:, k]
            continuation = torch.zeros((num_states, num_actions), dtype=torch.float32)
            for a in range(num_actions):
                # Feature part: sum_k theta_k * (P_a @ ev_k)
                ev_next_features = transitions[a] @ ev_features  # (S, K)
                feat_contribution = torch.einsum("sk,k->s", ev_next_features, params)

                # Entropy part: P_a @ ev_H
                ev_next_entropy = transitions[a] @ ev_entropy  # (S,)

                continuation[:, a] = gamma * (feat_contribution + ev_next_entropy)

            return flow_u + continuation

        def _log_likelihood(params: torch.Tensor) -> float:
            """Compute conditional choice log-likelihood."""
            v = _compute_choice_values(params)
            log_probs = torch.nn.functional.log_softmax(v / sigma, dim=1)

            ll = 0.0
            for traj in panel.trajectories:
                for t in range(len(traj)):
                    s = traj.states[t].item()
                    a = traj.actions[t].item()
                    ll += log_probs[s, a].item()
            return ll

        # Store the ll function for Hessian computation later
        self._log_likelihood_fn = _log_likelihood
        self._compute_choice_values_fn = _compute_choice_values

        # Scipy optimizer interface
        def objective(params_np):
            params = torch.tensor(params_np, dtype=torch.float32)
            return -_log_likelihood(params)

        def gradient(params_np):
            eps = 1e-5
            n = len(params_np)
            grad = torch.zeros(n)
            for i in range(n):
                p_plus = params_np.copy()
                p_minus = params_np.copy()
                p_plus[i] += eps
                p_minus[i] -= eps
                grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * eps)
            return grad.numpy()

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        lower, upper = utility.get_parameter_bounds()
        bounds = list(zip(lower.numpy(), upper.numpy()))

        result = optimize.minimize(
            objective,
            initial_params.numpy(),
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={
                "maxiter": self._config.outer_max_iter,
                "gtol": self._config.outer_tol,
            },
        )

        params_opt = torch.tensor(result.x, dtype=torch.float32)
        ll_opt = -result.fun

        return params_opt, ll_opt, result.nit, result.nfev, result.message

    # ------------------------------------------------------------------
    # Main estimation routine
    # ------------------------------------------------------------------

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run the full TD-CCP estimation pipeline.

        Args:
            panel: Panel data with observed choices.
            utility: Utility function specification.
            problem: Problem specification.
            transitions: Transition matrices P(s'|s,a).
            initial_params: Starting values for partial MLE.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            EstimationResult with estimated parameters and diagnostics.
        """
        start_time = time.time()
        num_states = problem.num_states
        num_actions = problem.num_actions
        gamma = problem.discount_factor
        sigma = problem.scale_parameter

        # Step 1: Estimate CCPs from data
        self._log("Step 1: Estimating CCPs from data")
        ccps = self._estimate_ccps(panel, num_states, num_actions)

        # Step 2: Decompose flows
        self._log("Step 2: Computing flow decomposition")
        feature_matrix = utility.feature_matrix
        flow_features, flow_entropy = self._compute_flow_components(ccps, feature_matrix)

        # Step 3: Train component networks
        self._log("Step 3: Training EV component networks via AVI")
        feature_nets, entropy_net, loss_histories = self._train_all_components(
            flow_features=flow_features,
            flow_entropy=flow_entropy,
            panel=panel,
            num_states=num_states,
            gamma=gamma,
        )

        # Extract learned EV components for all states
        all_state_features = self._build_state_features(
            torch.arange(num_states), num_states
        )
        ev_features, ev_entropy = self._evaluate_ev_components(
            feature_nets, entropy_net, all_state_features
        )

        # Step 4: Partial MLE
        self._log("Step 4: Running partial MLE")
        params_opt, ll_opt, n_iter, n_feval, opt_msg = self._partial_mle(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
            ev_features=ev_features,
            ev_entropy=ev_entropy,
            initial_params=initial_params,
        )
        self._log(f"  Parameters: {params_opt.numpy()}")
        self._log(f"  Log-likelihood: {ll_opt:.4f}")

        # Compute policy and value function from optimized parameters
        v = self._compute_choice_values_fn(params_opt)
        policy = torch.nn.functional.softmax(v / sigma, dim=1)
        V = sigma * torch.logsumexp(v / sigma, dim=1)

        # Step 5: Hessian for standard errors
        hessian = None
        if self._config.compute_se:
            self._log("Step 5: Computing numerical Hessian")

            def ll_fn(params):
                return torch.tensor(self._log_likelihood_fn(params))

            hessian = compute_numerical_hessian(params_opt, ll_fn)

        optimization_time = time.time() - start_time

        # Total inner iterations = avi_iterations * epochs_per_avi * num_components
        num_features = utility.num_parameters
        num_components = num_features + 1  # features + entropy
        total_inner = (
            self._config.avi_iterations
            * self._config.epochs_per_avi
            * num_components
        )

        return EstimationResult(
            parameters=params_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            gradient_contributions=None,
            converged=True,
            num_iterations=n_iter,
            num_function_evals=n_feval,
            num_inner_iterations=total_inner,
            message=f"TD-CCP completed: {opt_msg}",
            optimization_time=optimization_time,
            metadata={
                "loss_histories": loss_histories,
                "avi_iterations": self._config.avi_iterations,
                "epochs_per_avi": self._config.epochs_per_avi,
                "ev_features": ev_features,
                "ev_entropy": ev_entropy,
            },
        )
