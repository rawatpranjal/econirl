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
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
from scipy import optimize

from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


@dataclass
class SEESConfig:
    """Configuration for SEES estimator.

    Attributes:
        basis_type: Sieve basis type ("fourier" or "polynomial").
        basis_dim: Number of basis functions.
        penalty_lambda: L2 penalty on basis coefficients alpha.
        max_iter: Maximum L-BFGS-B iterations.
        tol: Gradient tolerance for convergence.
        compute_se: Whether to compute standard errors.
        se_method: Standard error method.
        verbose: Whether to print progress.
    """

    basis_type: str = "fourier"
    basis_dim: int = 8
    penalty_lambda: float = 0.01
    max_iter: int = 500
    tol: float = 1e-6
    compute_se: bool = True
    se_method: SEMethod = "asymptotic"
    verbose: bool = False


class SEESEstimator(BaseEstimator):
    """Sieve Estimator for dynamic discrete choice.

    Approximates V(s) with basis functions and jointly optimizes
    structural parameters and basis coefficients via penalized MLE.
    Standard errors use the Schur complement to marginalize out
    basis coefficients alpha, giving the correct marginal Hessian for theta.

    Args:
        config: SEESConfig or keyword arguments matching SEESConfig fields.

    Example:
        >>> estimator = SEESEstimator(basis_type="fourier", basis_dim=8)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> # Access basis coefficients
        >>> result.metadata["alpha"]
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
        config: SEESConfig | None = None,
    ):
        if config is not None:
            basis_type = config.basis_type
            basis_dim = config.basis_dim
            penalty_lambda = config.penalty_lambda
            max_iter = config.max_iter
            tol = config.tol
            compute_se = config.compute_se
            se_method = config.se_method
            verbose = config.verbose

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
        self._config = SEESConfig(
            basis_type=basis_type,
            basis_dim=basis_dim,
            penalty_lambda=penalty_lambda,
            max_iter=max_iter,
            tol=tol,
            compute_se=compute_se,
            se_method=se_method,
            verbose=verbose,
        )

    @property
    def name(self) -> str:
        return f"SEES ({self._basis_type}, Luo & Sang 2024)"

    @property
    def config(self) -> SEESConfig:
        """Return current configuration."""
        return self._config

    def _build_basis(self, n_states: int) -> jnp.ndarray:
        """Construct sieve basis matrix Psi(s).

        Args:
            n_states: Number of discrete states.

        Returns:
            Basis matrix, shape (n_states, basis_dim).
        """
        # Normalized state values in [0, 1]
        s_norm = jnp.linspace(0, 1, n_states)

        if self._basis_type == "fourier":
            # Fourier basis: [1, cos(pi*s), sin(pi*s), cos(2pi*s), sin(2pi*s), ...]
            basis = jnp.zeros((n_states, self._basis_dim))
            basis = basis.at[:, 0].set(1.0)  # Constant term
            for k in range(1, self._basis_dim):
                freq = (k + 1) // 2
                if k % 2 == 1:
                    basis = basis.at[:, k].set(jnp.cos(freq * np.pi * s_norm))
                else:
                    basis = basis.at[:, k].set(jnp.sin(freq * np.pi * s_norm))
            return basis

        elif self._basis_type == "polynomial":
            # Polynomial basis: [1, s, s^2, s^3, ...]
            # Use Chebyshev-normalized states for numerical stability
            s_cheb = 2 * s_norm - 1  # Map to [-1, 1]
            basis = jnp.zeros((n_states, self._basis_dim))
            for k in range(self._basis_dim):
                basis = basis.at[:, k].set(s_cheb ** k)
            return basis

        else:
            raise ValueError(f"Unknown basis type: {self._basis_type}")

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
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
        expected_basis = jnp.zeros((n_actions, n_states, n_alpha))
        for a in range(n_actions):
            expected_basis = expected_basis.at[a].set(transitions[a] @ basis)

        if initial_params is None:
            initial_params = utility.get_initial_parameters()

        # Joint parameter vector: [theta, alpha]
        initial_alpha = jnp.zeros(n_alpha)
        x0 = jnp.concatenate([initial_params, initial_alpha])

        def log_likelihood(x: jnp.ndarray) -> float:
            theta = x[:n_theta]
            alpha = x[n_theta:]

            # V(s) ~ basis @ alpha
            # Q(s,a) = u(s,a;theta) + beta * E[basis(s') . alpha | s, a]
            flow_u = jnp.einsum("sak,k->sa", feature_matrix, theta)

            # Continuation: beta * transitions[a] @ (basis @ alpha)
            continuation = jnp.zeros((n_states, n_actions))
            for a in range(n_actions):
                continuation = continuation.at[:, a].set(beta * (expected_basis[a] @ alpha))

            q_vals = flow_u + continuation
            log_probs = jax.nn.log_softmax(q_vals / sigma, axis=1)

            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            ll = log_probs[all_states, all_actions].sum().item()

            # L2 penalty on alpha
            penalty = self._penalty_lambda * (alpha ** 2).sum().item()
            return ll - penalty

        self._ll_fn = log_likelihood

        def objective(x_np):
            x = jnp.array(x_np, dtype=jnp.float32)
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
        lower = jnp.concatenate([lower_theta, jnp.full((n_alpha,), -50.0)])
        upper = jnp.concatenate([upper_theta, jnp.full((n_alpha,), 50.0)])
        bounds = list(zip(np.asarray(lower), np.asarray(upper)))

        self._log(f"SEES: {n_theta} structural + {n_alpha} basis params")

        result = optimize.minimize(
            objective,
            np.asarray(x0),
            method="L-BFGS-B",
            jac=gradient,
            bounds=bounds,
            options={
                "maxiter": self._max_iter,
                "gtol": self._tol,
            },
        )

        x_opt = jnp.array(result.x, dtype=jnp.float32)
        theta_opt = x_opt[:n_theta]
        alpha_opt = x_opt[n_theta:]
        ll_opt = -result.fun

        self._log(f"theta: {np.asarray(theta_opt)}")
        self._log(f"alpha: {np.asarray(alpha_opt)}")

        # Compute final policy
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, theta_opt)
        continuation = jnp.zeros((n_states, n_actions))
        for a in range(n_actions):
            continuation = continuation.at[:, a].set(beta * (expected_basis[a] @ alpha_opt))

        q_vals = flow_u + continuation
        policy = jax.nn.softmax(q_vals / sigma, axis=1)
        V = sigma * jax.scipy.special.logsumexp(q_vals / sigma, axis=1)

        # Hessian for theta SEs (marginal over alpha via Schur complement)
        hessian = None
        if self._compute_se:
            self._log("Computing Hessian for standard errors (Schur complement)")

            # Full Hessian over [theta, alpha], then marginalize out alpha
            # via the Schur complement: H_theta = H_tt - H_ta @ H_aa^{-1} @ H_at
            # This correctly accounts for uncertainty from alpha estimation.
            def full_ll_fn(x):
                return jnp.array(log_likelihood(x))

            full_hessian = compute_numerical_hessian(x_opt, full_ll_fn)

            H_tt = full_hessian[:n_theta, :n_theta]
            H_ta = full_hessian[:n_theta, n_theta:]
            H_at = full_hessian[n_theta:, :n_theta]
            H_aa = full_hessian[n_theta:, n_theta:]

            try:
                # Schur complement: H_theta = H_tt - H_ta @ H_aa^{-1} @ H_at
                hessian = H_tt - H_ta @ jnp.linalg.solve(H_aa, H_at)
            except RuntimeError:
                self._log("WARNING: H_aa singular, falling back to H_tt block")
                hessian = H_tt

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
