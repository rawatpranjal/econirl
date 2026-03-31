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


@dataclass
class NNESConfig:
    """Configuration for NNES estimator.

    Attributes:
        hidden_dim: Hidden units per layer in V-network.
        num_layers: Number of hidden layers in V-network.
        v_lr: Learning rate for V-network training (Phase 1).
        v_epochs: Number of training epochs for V-network per outer iteration.
        v_batch_size: Mini-batch size for V-network SGD.
        outer_max_iter: Maximum L-BFGS-B iterations for structural MLE (Phase 2).
        outer_tol: Gradient tolerance for L-BFGS-B convergence.
        n_outer_iterations: Number of Phase 1 + Phase 2 alternations.
        compute_se: Whether to compute standard errors.
        se_method: Standard error method. The Phase 2 pseudo-likelihood Hessian
            is the correct semiparametrically efficient variance estimator due to
            Neyman orthogonality (Nguyen 2025, Propositions 3-4): the score is
            orthogonal to V-approximation error, so no bias correction is needed.
        verbose: Whether to print progress.
    """

    hidden_dim: int = 32
    num_layers: int = 2
    v_lr: float = 1e-3
    v_epochs: int = 500
    v_batch_size: int = 512
    outer_max_iter: int = 200
    outer_tol: float = 1e-6
    n_outer_iterations: int = 3
    compute_se: bool = True
    se_method: SEMethod = "asymptotic"
    verbose: bool = False


class NNESEstimator(BaseEstimator):
    """Neural Network Efficient Estimator (Nguyen 2025).

    Phase 1 learns V(s) via Bellman residual minimization. Phase 2 uses
    the learned V in a structural MLE for theta. This avoids the costly
    inner fixed-point loop of NFXP while maintaining semiparametric efficiency.

    Standard errors are valid despite the neural V-approximation because of
    Neyman orthogonality (Nguyen 2025, Propositions 3-4): the score of the
    Phase 2 pseudo-likelihood is orthogonal to perturbations in V, so the
    Hessian of the pseudo-likelihood is the correct variance estimator with
    no bias correction needed. This is the zero Jacobian property.

    Args:
        config: NNESConfig or keyword arguments matching NNESConfig fields.

    Example:
        >>> estimator = NNESEstimator(hidden_dim=32, v_epochs=500)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> # Access V-network training diagnostics
        >>> result.metadata["v_loss_per_outer"]
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
        n_outer_iterations: int = 3,
        compute_se: bool = True,
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
        config: NNESConfig | None = None,
    ):
        if config is not None:
            hidden_dim = config.hidden_dim
            num_layers = config.num_layers
            v_lr = config.v_lr
            v_epochs = config.v_epochs
            v_batch_size = config.v_batch_size
            outer_max_iter = config.outer_max_iter
            outer_tol = config.outer_tol
            n_outer_iterations = config.n_outer_iterations
            compute_se = config.compute_se
            se_method = config.se_method
            verbose = config.verbose

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
        self._n_outer_iterations = n_outer_iterations
        self._compute_se = compute_se
        self._config = NNESConfig(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            v_lr=v_lr,
            v_epochs=v_epochs,
            v_batch_size=v_batch_size,
            outer_max_iter=outer_max_iter,
            outer_tol=outer_tol,
            n_outer_iterations=n_outer_iterations,
            compute_se=compute_se,
            se_method=se_method,
            verbose=verbose,
        )

    @property
    def name(self) -> str:
        return "NNES (Nguyen 2025)"

    @property
    def config(self) -> NNESConfig:
        """Return current configuration."""
        return self._config

    def _bootstrap_params_from_ccp(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        sigma: float,
        beta: float,
        feature_matrix: torch.Tensor,
        n_states: int,
        n_actions: int,
        bounds: list,
    ) -> torch.Tensor:
        """Estimate initial params from data CCPs via Hotz-Miller inversion.

        Uses CCP frequencies to compute the EV matrix-inversion closed form,
        then solves for theta via a quick partial MLE.
        """
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        # Estimate CCPs from data
        counts = torch.zeros(n_states, n_actions)
        for s, a in zip(all_states, all_actions):
            counts[s, a] += 1
        state_counts = counts.sum(dim=1, keepdim=True).clamp(min=1)
        ccps = (counts + 0.01) / (state_counts + n_actions * 0.01)

        # Hotz-Miller: compute EV via matrix inversion
        # P_pi[s, s'] = sum_a pi(a|s) P(s'|s,a)
        P_pi = torch.einsum("sa,ast->st", ccps, transitions)

        # Expected flow per-feature: flow_k[s] = sum_a pi(a|s) * phi(s,a,k)
        flow_features = torch.einsum("sa,sak->sk", ccps, feature_matrix)

        # Entropy: H(s) = -sum_a p(a|s) log p(a|s)
        safe_ccps = ccps.clamp(min=1e-10)
        entropy = -(ccps * safe_ccps.log()).sum(dim=1)

        # Solve for EV components: ev_k = (I - beta * P_pi)^{-1} flow_k
        # Use float64 for numerical stability at high discount factors
        # (condition number of I - beta*P_pi ~ 1/(1-beta) ~ 10,000 at beta=0.9999)
        I = torch.eye(n_states, dtype=torch.float64)
        try:
            M = I - beta * P_pi.double()
            ev_features = torch.linalg.solve(M, flow_features.double()).float()  # (S, K)
            ev_entropy = torch.linalg.solve(M, entropy.double()).float()  # (S,)
        except RuntimeError:
            # Fallback: return small positive values
            return torch.full((feature_matrix.shape[2],), 0.01)

        # Partial MLE with exact EV
        def objective(params_np):
            params = torch.tensor(params_np, dtype=torch.float32)
            flow_u = torch.einsum("sak,k->sa", feature_matrix, params)
            continuation = torch.zeros(n_states, n_actions)
            for a in range(n_actions):
                ev_next = transitions[a] @ ev_features  # (S, K)
                feat_part = torch.einsum("sk,k->s", ev_next, params)
                ent_part = transitions[a] @ ev_entropy
                continuation[:, a] = beta * (feat_part + sigma * ent_part)
            v = flow_u + continuation
            log_probs = torch.nn.functional.log_softmax(v / sigma, dim=1)
            return -log_probs[all_states, all_actions].sum().item()

        # Start slightly above zero to avoid boundary stall
        x0 = torch.full((feature_matrix.shape[2],), 0.01).numpy()
        result = optimize.minimize(
            objective, x0, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": 100},
        )
        return torch.tensor(result.x, dtype=torch.float32)

    def _build_state_features(self, states: torch.Tensor, problem: DDCProblem) -> torch.Tensor:
        """Build state features from state indices using problem's encoder."""
        if problem.state_encoder is not None:
            return problem.state_encoder(states)
        denom = max(problem.num_states - 1, 1)
        return (states.float() / denom).unsqueeze(-1)

    def _train_value_network(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        init_params: torch.Tensor,
    ) -> tuple[_ValueNetwork, list[float]]:
        """Phase 1: Train V(s) network via Bellman residual minimization.

        Minimizes ||V_phi(s) - T_theta V_phi(s)||^2 where T_theta is the
        soft Bellman operator at the initial parameter guess.

        Returns:
            Tuple of (trained V-network, list of per-epoch average losses).
        """
        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        v_net = _ValueNetwork(problem.state_dim or 1, self._hidden_dim, self._num_layers)
        optimizer = torch.optim.Adam(v_net.parameters(), lr=self._v_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=50, min_lr=1e-6,
        )

        # Get all transitions from data
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        all_next_states = panel.get_all_next_states()

        feat_s = self._build_state_features(all_states, problem)
        feat_sp = self._build_state_features(all_next_states, problem)

        # Compute flow utility at initial params
        feature_matrix = utility.feature_matrix  # (S, A, K)
        flow_u = torch.einsum("sak,k->sa", feature_matrix, init_params)

        # Pre-compute all state features for EV calculation
        all_state_feats = self._build_state_features(torch.arange(n_states), problem)

        n_samples = len(all_states)

        self._log(f"Phase 1: Training V-network for {self._v_epochs} epochs")

        loss_history: list[float] = []
        best_loss = float("inf")
        best_state_dict = None
        patience_counter = 0
        max_patience = 100

        for epoch in range(self._v_epochs):
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self._v_batch_size):
                end = min(start + self._v_batch_size, n_samples)
                idx = perm[start:end]

                # V(s) prediction
                v_s = v_net(feat_s[idx])

                # Build Q-values for all actions at observed states
                # Q(s,a) = u(s,a;theta) + beta * sum_s' P(s'|s,a) V(s')
                s_idx = all_states[idx]
                with torch.no_grad():
                    v_all = v_net(all_state_feats).detach()
                # EV[batch_i, a] = sum_s' P(s'|s_idx[i], a) * V(s')
                # transitions[:, s_idx, :] has shape (A, batch, S)
                ev = torch.einsum("abs,s->ba", transitions[:, s_idx, :], v_all)
                q_vals = flow_u[s_idx] + beta * ev

                # Bellman target
                target = sigma * torch.logsumexp(q_vals / sigma, dim=1)

                loss = nn.functional.mse_loss(v_s, target.detach())

                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping to prevent divergence
                torch.nn.utils.clip_grad_norm_(v_net.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)
            scheduler.step(avg_loss)

            # Early stopping with best model checkpoint
            if avg_loss < best_loss - 1e-8:
                best_loss = avg_loss
                best_state_dict = {k: v.clone() for k, v in v_net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if self._verbose and (epoch + 1) % 100 == 0:
                self._log(f"  V-net epoch {epoch+1}: loss={avg_loss:.6f}")

            # Stop if diverging or converged
            if patience_counter >= max_patience:
                self._log(f"  V-net early stopping at epoch {epoch+1} (patience={max_patience})")
                break
            if avg_loss > 1e8:
                self._log(f"  V-net divergence detected at epoch {epoch+1}, reverting to best")
                break

        # Restore best model
        if best_state_dict is not None:
            v_net.load_state_dict(best_state_dict)

        return v_net, loss_history

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run NNES estimation with outer iterations.

        Alternates between Phase 1 (train V-network with current theta)
        and Phase 2 (optimize theta with current V) to avoid the
        model mismatch from training V under zero/incorrect rewards.
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        feature_matrix = utility.feature_matrix  # (S, A, K)
        all_state_feat = self._build_state_features(torch.arange(n_states), problem)
        lower, upper = utility.get_parameter_bounds()
        bounds = list(zip(lower.numpy(), upper.numpy()))

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        # If initial params are all zeros, bootstrap from CCP-based estimate.
        # A V-network trained with zero rewards learns nothing useful.
        if torch.all(initial_params == 0):
            self._log("Bootstrapping initial params from CCP-based MLE")
            initial_params = self._bootstrap_params_from_ccp(
                panel, utility, problem, transitions, sigma, beta,
                feature_matrix, n_states, n_actions, bounds,
            )
            self._log(f"  Bootstrap params: {initial_params.numpy()}")

        current_params = initial_params.clone()
        total_nit = 0
        total_nfev = 0
        last_result = None
        v_loss_per_outer: list[float] = []
        all_v_loss_history: list[list[float]] = []

        for outer_iter in range(self._n_outer_iterations):
            self._log(f"Outer iteration {outer_iter + 1}/{self._n_outer_iterations}")

            # Phase 1: Train V-network with current params
            v_net, v_loss_history = self._train_value_network(
                panel, utility, problem, transitions, current_params,
            )
            v_loss_per_outer.append(v_loss_history[-1] if v_loss_history else float("nan"))
            all_v_loss_history.append(v_loss_history)

            # Extract V(s) for all states
            with torch.no_grad():
                v_all = v_net(all_state_feat)  # (S,)

            # Precompute E[V(s') | s, a] = transitions[a] @ V
            ev_sa = torch.zeros(n_states, n_actions)
            for a in range(n_actions):
                ev_sa[:, a] = transitions[a] @ v_all  # (S,)

            # Phase 2: Structural MLE
            self._log(f"  Phase 2: Structural MLE over theta")

            # Need closure over current ev_sa
            _ev_sa = ev_sa

            def log_likelihood(params: torch.Tensor) -> float:
                flow_u = torch.einsum("sak,k->sa", feature_matrix, params)
                q_vals = flow_u + beta * _ev_sa
                log_probs = torch.nn.functional.log_softmax(q_vals / sigma, dim=1)
                all_states = panel.get_all_states()
                all_actions = panel.get_all_actions()
                return log_probs[all_states, all_actions].sum().item()

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

            last_result = optimize.minimize(
                objective,
                current_params.numpy(),
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={
                    "maxiter": self._outer_max_iter,
                    "gtol": self._outer_tol,
                },
            )

            current_params = torch.tensor(last_result.x, dtype=torch.float32)
            total_nit += last_result.nit
            total_nfev += last_result.nfev
            self._log(f"  Params: {current_params.numpy()}, LL: {-last_result.fun:.2f}")

        # Store final ll function for Hessian
        self._ll_fn = log_likelihood

        params_opt = current_params
        ll_opt = -last_result.fun

        # Compute final policy using last ev_sa
        flow_u = torch.einsum("sak,k->sa", feature_matrix, params_opt)
        q_vals = flow_u + beta * ev_sa
        policy = torch.nn.functional.softmax(q_vals / sigma, dim=1)
        V = sigma * torch.logsumexp(q_vals / sigma, dim=1)

        # Hessian for standard errors.
        # By Nguyen (2025) Propositions 3-4 (Neyman orthogonality), the score
        # of the Phase 2 pseudo-likelihood is orthogonal to V-approximation
        # error. Therefore the Hessian of ℓ(θ | V̂_fixed) is the correct
        # semiparametrically efficient variance estimator — no bias correction
        # needed despite using an approximate V-network.
        hessian = None
        if self._compute_se:
            self._log("Computing Hessian (semiparametrically efficient via orthogonality)")

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
            converged=last_result.success,
            num_iterations=total_nit,
            num_function_evals=total_nfev,
            message=f"NNES ({self._n_outer_iterations} outer iters): {last_result.message}",
            optimization_time=elapsed,
            metadata={
                "v_network_values": v_all,
                "ev_sa": ev_sa,
                "n_outer_iterations": self._n_outer_iterations,
                "v_loss_per_outer": v_loss_per_outer,
                "v_loss_history": all_v_loss_history,
            },
        )
