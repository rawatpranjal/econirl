"""Sieve Estimator (SEES) for dynamic discrete choice models.

Approximates the value function V(s) with sieve basis functions
(Fourier or polynomial), then performs penalized maximum likelihood
jointly over structural parameters theta and basis coefficients alpha.

This avoids both the costly inner fixed-point loop of NFXP and the
neural network training of NNES, using a closed-form basis expansion
that can be solved with standard nonlinear optimization.

Algorithm:
    1. Construct sieve basis Psi(s) of dimension K (Fourier or polynomial)
    2. V(s) ~ Psi(s) . alpha, where alpha are basis coefficients
    3. Q(s,a;theta,alpha) = u(s,a;theta) + beta * E[Psi(s').alpha | s,a]
    4. P(a|s) = softmax(Q(s,a) / sigma)
    5. Maximize: LL(theta, alpha) - lambda * ||alpha||^2

Reference:
    Luo, Y. & Sang, Y. (2024). "Sieve Estimation of Dynamic Discrete
    Choice Models." Working Paper.
"""

from __future__ import annotations

import time

import numpy as np
import torch
from scipy import optimize

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


class SEESEstimator(BaseEstimator):
    """Sieve Estimator for dynamic discrete choice.

    Approximates V(s) with basis functions and jointly optimizes
    structural parameters and basis coefficients via penalized MLE.

    Attributes:
        basis_type: Type of sieve basis ("fourier" or "polynomial").
        basis_dim: Number of basis functions.
        penalty_lambda: L2 penalty on basis coefficients.
        max_iter: Maximum L-BFGS-B iterations.
        tol: Gradient tolerance for convergence.
        compute_se: Whether to compute standard errors.

    Example:
        >>> estimator = SEESEstimator(basis_type="fourier", basis_dim=8)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        basis_type: str = "fourier",
        basis_dim: int = 8,
        penalty_lambda: float = 0.01,
        max_iter: int = 500,
        tol: float = 1e-6,
        compute_se: bool = True,
        se_method: SEMethod = "asymptotic",
        verbose: bool = False,
    ):
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_se,
            verbose=verbose,
        )
        self._basis_type = basis_type
        self._basis_dim = basis_dim
        self._penalty_lambda = penalty_lambda
        self._max_iter = max_iter
        self._tol = tol
        self._compute_se = compute_se

    @property
    def name(self) -> str:
        return f"SEES ({self._basis_type}, Luo & Sang 2024)"

    def _build_basis(self, n_states: int) -> torch.Tensor:
        """Construct sieve basis matrix Psi(s).

        Args:
            n_states: Number of discrete states.

        Returns:
            Basis matrix, shape (n_states, basis_dim).
        """
        # Normalized state values in [0, 1]
        s_norm = torch.linspace(0, 1, n_states)

        if self._basis_type == "fourier":
            # Fourier basis: [1, cos(pi*s), sin(pi*s), cos(2pi*s), sin(2pi*s), ...]
            basis = torch.zeros(n_states, self._basis_dim)
            basis[:, 0] = 1.0  # Constant term
            for k in range(1, self._basis_dim):
                freq = (k + 1) // 2
                if k % 2 == 1:
                    basis[:, k] = torch.cos(freq * np.pi * s_norm)
                else:
                    basis[:, k] = torch.sin(freq * np.pi * s_norm)
            return basis

        elif self._basis_type == "polynomial":
            # Polynomial basis: [1, s, s^2, s^3, ...]
            # Use Chebyshev-normalized states for numerical stability
            s_cheb = 2 * s_norm - 1  # Map to [-1, 1]
            basis = torch.zeros(n_states, self._basis_dim)
            for k in range(self._basis_dim):
                basis[:, k] = s_cheb ** k
            return basis

        else:
            raise ValueError(f"Unknown basis type: {self._basis_type}")

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run sieve estimation."""
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor

        feature_matrix = utility.feature_matrix  # (S, A, K)
        n_theta = utility.num_parameters
        n_alpha = self._basis_dim

        # Build sieve basis
        basis = self._build_basis(n_states)  # (S, basis_dim)

        # Precompute E[Psi(s') | s, a] = transitions[a] @ basis
        # Shape: (n_actions, n_states, basis_dim)
        expected_basis = torch.zeros(n_actions, n_states, n_alpha)
        for a in range(n_actions):
            expected_basis[a] = transitions[a] @ basis  # (S, basis_dim)

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        # Joint parameter vector: [theta, alpha]
        initial_alpha = torch.zeros(n_alpha)
        x0 = torch.cat([initial_params, initial_alpha])

        def log_likelihood(x: torch.Tensor) -> float:
            theta = x[:n_theta]
            alpha = x[n_theta:]

            # V(s) ~ basis @ alpha
            # Q(s,a) = u(s,a;theta) + beta * E[basis(s') . alpha | s, a]
            flow_u = torch.einsum("sak,k->sa", feature_matrix, theta)

            # Continuation: beta * transitions[a] @ (basis @ alpha)
            continuation = torch.zeros(n_states, n_actions)
            for a in range(n_actions):
                continuation[:, a] = beta * (expected_basis[a] @ alpha)

            q_vals = flow_u + continuation
            log_probs = torch.nn.functional.log_softmax(q_vals / sigma, dim=1)

            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            ll = log_probs[all_states, all_actions].sum().item()

            # L2 penalty on alpha
            penalty = self._penalty_lambda * (alpha ** 2).sum().item()
            return ll - penalty

        self._ll_fn = log_likelihood

        def objective(x_np):
            x = torch.tensor(x_np, dtype=torch.float32)
            return -log_likelihood(x)

        def gradient(x_np):
            eps = 1e-5
            n = len(x_np)
            grad = np.zeros(n)
            for i in range(n):
                x_plus = x_np.copy()
                x_minus = x_np.copy()
                x_plus[i] += eps
                x_minus[i] -= eps
                grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
            return grad

        # Bounds: theta bounds from utility, alpha unbounded
        lower_theta, upper_theta = utility.get_parameter_bounds()
        lower = torch.cat([lower_theta, torch.full((n_alpha,), -50.0)])
        upper = torch.cat([upper_theta, torch.full((n_alpha,), 50.0)])
        bounds = list(zip(lower.numpy(), upper.numpy()))

        self._log(f"SEES: {n_theta} structural + {n_alpha} basis params")

        result = optimize.minimize(
            objective,
            x0.numpy(),
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={
                "maxiter": self._max_iter,
                "gtol": self._tol,
            },
        )

        x_opt = torch.tensor(result.x, dtype=torch.float32)
        theta_opt = x_opt[:n_theta]
        alpha_opt = x_opt[n_theta:]
        ll_opt = -result.fun

        self._log(f"theta: {theta_opt.numpy()}")
        self._log(f"alpha: {alpha_opt.numpy()}")

        # Compute final policy
        flow_u = torch.einsum("sak,k->sa", feature_matrix, theta_opt)
        continuation = torch.zeros(n_states, n_actions)
        for a in range(n_actions):
            continuation[:, a] = beta * (expected_basis[a] @ alpha_opt)

        q_vals = flow_u + continuation
        policy = torch.nn.functional.softmax(q_vals / sigma, dim=1)
        V = sigma * torch.logsumexp(q_vals / sigma, dim=1)

        # Hessian for theta SEs (marginal over alpha)
        hessian = None
        if self._compute_se:
            self._log("Computing Hessian for standard errors")

            # Hessian over full (theta, alpha) then extract theta block
            def full_ll_fn(x):
                return torch.tensor(log_likelihood(x))

            full_hessian = compute_numerical_hessian(x_opt, full_ll_fn)
            # Extract theta-theta block
            hessian = full_hessian[:n_theta, :n_theta]

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=theta_opt,
            log_likelihood=ll_opt,
            value_function=V,
            policy=policy,
            hessian=hessian,
            converged=result.success,
            num_iterations=result.nit,
            num_function_evals=result.nfev,
            message=f"SEES: {result.message}",
            optimization_time=elapsed,
            metadata={
                "alpha": alpha_opt,
                "basis_type": self._basis_type,
                "basis_dim": self._basis_dim,
                "basis_matrix": basis,
            },
        )
