"""Neural Network Efficient Estimator (NNES) for dynamic discrete choice.

Two-phase estimator that combines neural function approximation with
structural MLE. Phase 1 trains a neural V(s) network via Bellman residual
minimization on observed data. Phase 2 plugs the learned V-network into
an NFXP-style outer MLE over structural parameters theta.

Algorithm:
    Phase 1: Value function approximation
        1. Initialize V-network V_phi(s)
        2. For each epoch:
           a. For each (s, a, s') in data:
              - Compute Bellman target: u(s,a;theta_init) + beta * V_phi(s')
              - Compute V prediction: sigma * logsumexp_a [u(s,a;theta_init) + beta * V_phi(s')] / sigma
              - Minimize Bellman residual: ||V_phi(s) - T V_phi(s)||^2

    Phase 2: Structural MLE
        1. Given learned V_phi, define:
           - Q(s,a;theta) = u(s,a;theta) + beta * E[V_phi(s') | s, a]
           - P(a|s;theta) = softmax(Q/sigma)
        2. Maximize CCP log-likelihood over theta using L-BFGS-B

Reference:
    Nguyen, H. (2025). "Neural Network Estimators for Dynamic Discrete
    Choice Models." Working Paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


class _ValueNetwork(nn.Module):
    """MLP that maps state features to a scalar value V(s)."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (batch, input_dim), output (batch,)."""
        return self.net(x).squeeze(-1)


class NNESEstimator(BaseEstimator):
    """Neural Network Efficient Estimator.

    Phase 1 learns V(s) via Bellman residual minimization. Phase 2 uses
    the learned V in a structural MLE for theta. This avoids the costly
    inner fixed-point loop of NFXP while maintaining efficiency.

    Attributes:
        hidden_dim: Hidden units per layer in V-network.
        num_layers: Number of hidden layers.
        v_lr: Learning rate for V-network training.
        v_epochs: Number of training epochs for V-network.
        outer_max_iter: Maximum L-BFGS-B iterations for Phase 2.
        outer_tol: Gradient tolerance for L-BFGS-B.
        compute_se: Whether to compute standard errors.

    Example:
        >>> estimator = NNESEstimator(hidden_dim=32, v_epochs=500)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        v_lr: float = 1e-3,
        v_epochs: int = 500,
        v_batch_size: int = 512,
        outer_max_iter: int = 200,
        outer_tol: float = 1e-6,
        compute_se: bool = True,
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
    ):
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_se,
            verbose=verbose,
        )
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._v_lr = v_lr
        self._v_epochs = v_epochs
        self._v_batch_size = v_batch_size
        self._outer_max_iter = outer_max_iter
        self._outer_tol = outer_tol
        self._compute_se = compute_se

    @property
    def name(self) -> str:
        return "NNES (Nguyen 2025)"

    def _build_state_features(self, states: torch.Tensor, n_states: int) -> torch.Tensor:
        """Normalize state indices to [0, 1] features for NN input."""
        denom = max(n_states - 1, 1)
        return (states.float() / denom).unsqueeze(-1)

    def _train_value_network(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        init_params: torch.Tensor,
    ) -> _ValueNetwork:
        """Phase 1: Train V(s) network via Bellman residual minimization.

        Minimizes ||V_phi(s) - T_theta V_phi(s)||^2 where T_theta is the
        soft Bellman operator at the initial parameter guess.
        """
        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        v_net = _ValueNetwork(1, self._hidden_dim, self._num_layers)
        optimizer = torch.optim.Adam(v_net.parameters(), lr=self._v_lr)

        # Get all transitions from data
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        all_next_states = panel.get_all_next_states()

        feat_s = self._build_state_features(all_states, n_states)
        feat_sp = self._build_state_features(all_next_states, n_states)

        # Compute flow utility at initial params
        feature_matrix = utility.feature_matrix  # (S, A, K)
        flow_u = torch.einsum("sak,k->sa", feature_matrix, init_params)

        n_samples = len(all_states)

        self._log(f"Phase 1: Training V-network for {self._v_epochs} epochs")

        for epoch in range(self._v_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self._v_batch_size):
                end = min(start + self._v_batch_size, n_samples)
                idx = perm[start:end]

                # V(s) prediction
                v_s = v_net(feat_s[idx])

                # Bellman target: sigma * logsumexp_a [Q(s,a)/sigma]
                # Q(s,a) = u(s,a;theta) + beta * V(s')
                # For each sample, compute V(s') for the observed next state
                with torch.no_grad():
                    v_sp = v_net(feat_sp[idx])

                # Build Q-values for all actions at observed states
                s_idx = all_states[idx]
                q_vals = flow_u[s_idx] + beta * torch.einsum(
                    "ast,t->sa",
                    transitions[:, s_idx, :].permute(1, 0, 2),
                    v_net(self._build_state_features(torch.arange(n_states), n_states)).detach(),
                ).reshape(len(idx), n_actions)

                # Bellman target
                target = sigma * torch.logsumexp(q_vals / sigma, dim=1)

                loss = nn.functional.mse_loss(v_s, target.detach())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if self._verbose and (epoch + 1) % 100 == 0:
                self._log(f"  V-net epoch {epoch+1}: loss={epoch_loss/max(n_batches,1):.6f}")

        return v_net

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run two-phase NNES estimation."""
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        feature_matrix = utility.feature_matrix  # (S, A, K)

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        # Phase 1: Train V-network
        v_net = self._train_value_network(
            panel, utility, problem, transitions, initial_params,
        )

        # Extract V(s) for all states
        all_state_feat = self._build_state_features(torch.arange(n_states), n_states)
        with torch.no_grad():
            v_all = v_net(all_state_feat)  # (S,)

        # Precompute E[V(s') | s, a] = transitions[a] @ V
        ev_sa = torch.zeros(n_states, n_actions)
        for a in range(n_actions):
            ev_sa[:, a] = transitions[a] @ v_all  # (S,)

        # Phase 2: Structural MLE
        self._log("Phase 2: Structural MLE over theta")

        def log_likelihood(params: torch.Tensor) -> float:
            flow_u = torch.einsum("sak,k->sa", feature_matrix, params)
            q_vals = flow_u + beta * ev_sa
            log_probs = torch.nn.functional.log_softmax(q_vals / sigma, dim=1)

            ll = 0.0
            for traj in panel.trajectories:
                for t in range(len(traj)):
                    s = traj.states[t].item()
                    a = traj.actions[t].item()
                    ll += log_probs[s, a].item()
            return ll

        self._ll_fn = log_likelihood

        def objective(params_np):
            params = torch.tensor(params_np, dtype=torch.float32)
            return -log_likelihood(params)

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

        lower, upper = utility.get_parameter_bounds()
        bounds = list(zip(lower.numpy(), upper.numpy()))

        result = optimize.minimize(
            objective,
            initial_params.numpy(),
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={
                "maxiter": self._outer_max_iter,
                "gtol": self._outer_tol,
            },
        )

        params_opt = torch.tensor(result.x, dtype=torch.float32)
        ll_opt = -result.fun

        # Compute final policy
        flow_u = torch.einsum("sak,k->sa", feature_matrix, params_opt)
        q_vals = flow_u + beta * ev_sa
        policy = torch.nn.functional.softmax(q_vals / sigma, dim=1)
        V = sigma * torch.logsumexp(q_vals / sigma, dim=1)

        # Hessian for standard errors
        hessian = None
        if self._compute_se:
            self._log("Computing numerical Hessian")

            def ll_fn(params):
                return torch.tensor(log_likelihood(params))

            hessian = compute_numerical_hessian(params_opt, ll_fn)

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=params_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            converged=result.success,
            num_iterations=result.nit,
            num_function_evals=result.nfev,
            message=f"NNES: {result.message}",
            optimization_time=elapsed,
            metadata={
                "v_network_values": v_all,
                "ev_sa": ev_sa,
            },
        )
