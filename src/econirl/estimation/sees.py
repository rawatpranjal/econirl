"""Sieve Estimator (SEES) for dynamic discrete choice models.

Approximates the value function V(s) with sieve basis functions, then performs
penalized maximum likelihood jointly over structural parameters theta and basis
coefficients alpha.

This avoids both the costly inner fixed-point loop of NFXP and the
neural network training of NNES, using a closed-form basis expansion
that can be solved with standard nonlinear optimization.

Algorithm:
    1. Construct sieve basis Psi(s) of dimension K (Fourier or polynomial)
    2. V(s) ~ Psi(s) . alpha, where alpha are basis coefficients
    3. Q(s,a;theta,alpha) = u(s,a;theta) + beta * E[Psi(s').alpha | s,a]
    4. P(a|s) = softmax(Q(s,a) / sigma)
    5. Maximize: LL(theta, alpha) - omega * ||V(alpha) - T(V; theta)||^2

Reference:
    Luo, Y. & Sang, Y. (2024). "Sieve Estimation of Dynamic Discrete
    Choice Models." Working Paper.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import UtilityFunction


@dataclass
class SEESConfig:
    """Configuration for SEES estimator.

    Attributes:
        basis_type: Sieve basis type. "bspline" uses cubic B-splines per
            Luo and Sang (2024). "polynomial" uses monomials on [-1, 1].
            "fourier" is an econirl extension not in the original paper.
        basis_dim: Number of basis functions.
        penalty_weight: Weight omega on the equilibrium penalty
            (Luo and Sang 2024, equation 3). Penalizes the Bellman
            equation violation ||V - T(V; theta)||^2. Higher values
            enforce the Bellman constraint more strongly, pushing the
            estimator toward MLE. The paper recommends increasing omega
            until the confidence interval stabilizes. If
            penalty_schedule is set, that value is used instead.
        penalty_schedule: Optional callable mapping sample size n to a
            penalty weight, implementing the omega_n -> infinity schedule
            of Luo and Sang (2024). When set, supersedes penalty_weight.
            Example: lambda n: 1.0 * n ** 0.5.
        spline_degree: Polynomial degree for the B-spline basis (default 3).
        state_basis_mode: "index" always builds the historical basis over
            state indices; "encoded" builds the sieve over
            problem.state_encoder states; "auto" uses encoded features only
            for high-dimensional encoded state spaces and otherwise keeps the
            index basis.
        warm_start_value: Whether to initialize sieve coefficients by solving
            the Bellman equation at the initial theta and projecting the value
            function into the sieve basis.
        max_iter: Maximum L-BFGS-B iterations.
        tol: Gradient tolerance for convergence.
        compute_se: Whether to compute standard errors.
        se_method: Standard error method.
        verbose: Whether to print progress.
    """

    basis_type: str = "bspline"
    basis_dim: int = 8
    penalty_weight: float = 10.0
    penalty_schedule: Callable[[int], float] | None = None
    spline_degree: int = 3
    state_basis_mode: str = "auto"
    warm_start_value: bool = True
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
        basis_type: str = "bspline",
        basis_dim: int = 8,
        penalty_weight: float = 10.0,
        penalty_schedule: Callable[[int], float] | None = None,
        spline_degree: int = 3,
        state_basis_mode: str = "auto",
        warm_start_value: bool = True,
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
            penalty_weight = config.penalty_weight
            penalty_schedule = config.penalty_schedule
            spline_degree = config.spline_degree
            state_basis_mode = config.state_basis_mode
            warm_start_value = config.warm_start_value
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
        self._penalty_weight = penalty_weight
        self._penalty_schedule = penalty_schedule
        self._spline_degree = spline_degree
        if state_basis_mode not in {"auto", "index", "encoded"}:
            raise ValueError(
                "state_basis_mode must be one of 'auto', 'index', or 'encoded'"
            )
        self._state_basis_mode = state_basis_mode
        self._warm_start_value = warm_start_value
        self._max_iter = max_iter
        self._tol = tol
        self._compute_se = compute_se
        self._config = SEESConfig(
            basis_type=basis_type,
            basis_dim=basis_dim,
            penalty_weight=penalty_weight,
            penalty_schedule=penalty_schedule,
            spline_degree=spline_degree,
            state_basis_mode=state_basis_mode,
            warm_start_value=warm_start_value,
            max_iter=max_iter,
            tol=tol,
            compute_se=compute_se,
            se_method=se_method,
            verbose=verbose,
        )
        self._last_basis_metadata: dict[str, object] = {}

    @property
    def name(self) -> str:
        return f"SEES ({self._basis_type}, Luo & Sang 2024)"

    @property
    def config(self) -> SEESConfig:
        """Return current configuration."""
        return self._config

    def _build_basis(
        self,
        n_states: int,
        problem: DDCProblem | None = None,
    ) -> jnp.ndarray:
        """Construct sieve basis matrix Psi(s).

        Args:
            n_states: Number of discrete states.
            problem: Optional DDCProblem. When supplied and
                state_basis_mode permits it, high-dimensional state encoders
                are used to build an encoded-state basis.

        Returns:
            Basis matrix, shape (n_states, basis_dim).
        """
        use_encoded = self._use_encoded_basis(problem)
        if use_encoded:
            return self._build_encoded_state_basis(problem)

        self._last_basis_metadata = {
            "basis_source": "state_index",
            "basis_family": self._basis_type,
            "state_feature_dim": None,
            "configured_basis_dim": self._basis_dim,
        }
        return self._build_index_basis(n_states)

    def _use_encoded_basis(self, problem: DDCProblem | None) -> bool:
        if self._state_basis_mode == "index":
            return False
        if problem is None or problem.state_encoder is None:
            if self._state_basis_mode == "encoded":
                raise ValueError("state_basis_mode='encoded' requires problem.state_encoder")
            return False
        if self._state_basis_mode == "encoded":
            return True
        return (problem.state_dim or 0) > 2

    def _build_index_basis(self, n_states: int) -> jnp.ndarray:
        """Construct the historical index-based sieve basis."""
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
            # Monomial basis: [1, s, s^2, s^3, ...] on [-1, 1]
            s_cheb = 2 * s_norm - 1  # Map to [-1, 1] for conditioning
            basis = jnp.zeros((n_states, self._basis_dim))
            for k in range(self._basis_dim):
                basis = basis.at[:, k].set(s_cheb ** k)
            return basis

        elif self._basis_type == "bspline":
            # Cubic B-spline basis per Luo and Sang (2024). The K basis
            # functions span the unit interval with equally-spaced knots
            # and degree `spline_degree` (default 3 = cubic).
            from scipy.interpolate import BSpline as _BSpline

            degree = self._spline_degree
            K = self._basis_dim
            n_interior = K - degree - 1
            if n_interior < 0:
                raise ValueError(
                    f"basis_dim ({K}) must exceed spline_degree ({degree}) "
                    f"to admit a clamped B-spline basis."
                )
            interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
            knots = np.concatenate([
                np.zeros(degree + 1),
                interior,
                np.ones(degree + 1),
            ])
            s_grid = np.asarray(s_norm)
            basis_np = np.zeros((n_states, K))
            for k in range(K):
                coeffs = np.zeros(K)
                coeffs[k] = 1.0
                spline = _BSpline(knots, coeffs, degree, extrapolate=False)
                basis_np[:, k] = np.nan_to_num(spline(s_grid), nan=0.0)
            return jnp.array(basis_np, dtype=jnp.float64)

        else:
            raise ValueError(f"Unknown basis type: {self._basis_type}")

    def _build_encoded_state_basis(self, problem: DDCProblem | None) -> jnp.ndarray:
        """Build a stable basis over encoded state features.

        The high-dimensional known-truth DGP exposes a finite grid through
        problem.state_encoder. A Gaussian RBF dictionary on those encoded
        states is then orthonormalized with an SVD. With basis_dim >= S this
        spans every finite-state value function; with fewer columns it is a
        smooth feature-aware sieve.
        """
        if problem is None or problem.state_encoder is None:
            raise ValueError("encoded-state SEES basis requires problem.state_encoder")

        states = jnp.arange(problem.num_states, dtype=jnp.int32)
        features = np.asarray(problem.state_encoder(states), dtype=np.float64)
        if features.ndim == 1:
            features = features[:, None]
        if features.shape[0] != problem.num_states:
            raise ValueError(
                "problem.state_encoder must return one row per state; "
                f"got {features.shape[0]} rows for {problem.num_states} states"
            )

        mean = features.mean(axis=0, keepdims=True)
        scale = np.maximum(features.std(axis=0, keepdims=True), 1e-8)
        z = (features - mean) / scale
        sqdist = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
        nearest_dist = np.min(
            np.where(sqdist > 1e-12, sqdist, np.inf),
            axis=1,
        )
        finite_nearest = nearest_dist[np.isfinite(nearest_dist)]
        bandwidth_sq = (
            float(np.median(finite_nearest)) if finite_nearest.size else 1.0
        )
        bandwidth_sq = max(bandwidth_sq, 1e-6)

        requested = max(1, int(self._basis_dim))
        n_centers = min(requested, problem.num_states)
        if n_centers == problem.num_states:
            center_idx = np.arange(problem.num_states)
        else:
            center_idx = np.unique(
                np.linspace(0, problem.num_states - 1, n_centers).round().astype(int)
            )
            while center_idx.size < n_centers:
                missing = [
                    idx for idx in range(problem.num_states) if idx not in set(center_idx)
                ]
                center_idx = np.sort(np.concatenate([center_idx, missing[:1]]))

        raw_rbf = np.exp(-0.5 * sqdist[:, center_idx] / bandwidth_sq)
        if requested >= problem.num_states:
            raw = raw_rbf
        else:
            raw = np.column_stack([np.ones(problem.num_states), raw_rbf])
            raw = raw[:, : min(requested, raw.shape[1])]

        # Orthonormalize for stable alpha scaling and projection.
        u, singular_values, _ = np.linalg.svd(raw, full_matrices=False)
        rank = int(np.sum(singular_values > 1e-10))
        if rank == 0:
            raise ValueError("encoded-state basis is numerically rank deficient")
        use = min(requested, rank)
        basis = u[:, :use] * np.sqrt(float(problem.num_states))

        self._last_basis_metadata = {
            "basis_source": "encoded_state",
            "basis_family": "rbf_svd",
            "state_feature_dim": int(features.shape[1]),
            "configured_basis_dim": self._basis_dim,
            "actual_basis_dim": int(use),
            "rbf_bandwidth_sq": bandwidth_sq,
            "encoded_basis_rank": rank,
        }
        return jnp.array(basis, dtype=jnp.float64)

    def _project_value_solution(
        self,
        basis: jnp.ndarray,
        feature_matrix: jnp.ndarray,
        initial_params: jnp.ndarray,
        problem: DDCProblem,
        transitions: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float, bool]:
        """Project the Bellman solution at initial theta into the sieve."""
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, initial_params)
        operator = SoftBellmanOperator(problem, transitions)
        solution = value_iteration(
            operator,
            flow_u,
            tol=1e-10,
            max_iter=10_000,
        )
        basis_np = np.asarray(basis, dtype=np.float64)
        value_np = np.asarray(solution.V, dtype=np.float64)
        alpha_np, *_ = np.linalg.lstsq(basis_np, value_np, rcond=None)
        projected = basis_np @ alpha_np
        projection_rmse = float(np.sqrt(np.mean((projected - value_np) ** 2)))
        return (
            jnp.array(alpha_np, dtype=jnp.float64),
            projection_rmse,
            bool(solution.converged),
        )

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run sieve estimation (Luo and Sang 2024, equation 4).

        Jointly optimizes structural parameters theta and basis
        coefficients alpha by maximizing the penalized criterion:

            LL(theta, alpha) - omega * ||V(alpha) - T(V(alpha); theta)||^2

        where T is the soft Bellman operator. The penalty enforces
        the equilibrium condition V = T(V; theta).
        """
        start_time = time.time()

        n_states = problem.num_states
        n_actions = problem.num_actions
        sigma = problem.scale_parameter
        beta = problem.discount_factor
        if self._penalty_schedule is not None:
            n_obs = sum(len(t.actions) for t in panel.trajectories)
            omega = float(self._penalty_schedule(n_obs))
        else:
            omega = self._penalty_weight

        feature_matrix = jnp.array(utility.feature_matrix, dtype=jnp.float64)
        n_theta = utility.num_parameters
        obs_states = panel.get_all_states()
        obs_actions = panel.get_all_actions()
        n_obs = int(panel.num_observations)

        state_action_counts = jnp.zeros((n_states, n_actions), dtype=jnp.float64)
        state_action_counts = state_action_counts.at[obs_states, obs_actions].add(1.0)

        # Build sieve basis
        basis = self._build_basis(n_states, problem)  # (S, basis_dim)
        basis_metadata = dict(self._last_basis_metadata)
        n_alpha = int(basis.shape[1])

        # Precompute E[Psi(s') | s, a] = transitions[a] @ basis
        # Shape: (n_actions, n_states, basis_dim)
        transitions_f64 = jnp.array(transitions, dtype=jnp.float64)
        expected_basis = jnp.einsum("ast,tk->ask", transitions_f64, basis)

        if initial_params is None:
            initial_params = utility.get_initial_parameters()
        initial_params = jnp.array(initial_params, dtype=jnp.float64)

        # Joint parameter vector: [theta, alpha]
        projection_rmse = float("nan")
        projection_converged = False
        if self._warm_start_value:
            initial_alpha, projection_rmse, projection_converged = (
                self._project_value_solution(
                    basis=basis,
                    feature_matrix=feature_matrix,
                    initial_params=initial_params,
                    problem=problem,
                    transitions=transitions_f64,
                )
            )
        else:
            initial_alpha = jnp.zeros(n_alpha, dtype=jnp.float64)
        x0 = jnp.concatenate([initial_params, initial_alpha])

        def criterion_parts(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
            theta = x[:n_theta]
            alpha = x[n_theta:]

            # V(s) = basis(s) @ alpha
            V_approx = basis @ alpha  # (S,)

            # Q(s,a) = u(s,a;theta) + beta * E[Psi(s') @ alpha | s, a]
            flow_u = jnp.einsum("sak,k->sa", feature_matrix, theta)
            # expected_basis shape: (A, S, K), alpha shape: (K,)
            continuation = beta * jnp.einsum("ask,k->sa", expected_basis, alpha)
            q_vals = flow_u + continuation

            # Log-likelihood
            log_probs = jax.nn.log_softmax(q_vals / sigma, axis=1)
            ll_sum = jnp.sum(state_action_counts * log_probs)

            # Bellman penalty: ||V - T(V; theta)||^2
            # T(V; theta) = sigma * logsumexp(Q / sigma)
            TV = sigma * jax.scipy.special.logsumexp(q_vals / sigma, axis=1)
            bellman_mse = jnp.mean((V_approx - TV) ** 2)

            return ll_sum, bellman_mse

        def penalized_criterion_mean(x: jnp.ndarray) -> jnp.ndarray:
            ll_sum, bellman_mse = criterion_parts(x)
            return ll_sum / n_obs - omega * bellman_mse

        def penalized_criterion_sum(x: jnp.ndarray) -> jnp.ndarray:
            return n_obs * penalized_criterion_mean(x)

        self._ll_fn = penalized_criterion_sum

        # JAX autodiff objective (minimization: negate the penalized log-likelihood)
        _neg_penalized_ll = jax.jit(lambda x: -penalized_criterion_mean(x))

        # Bounds: theta bounds from utility, alpha bounded loosely
        lower_theta, upper_theta = utility.get_parameter_bounds()
        alpha_bound = 1e5
        lower = jnp.concatenate([
            lower_theta.astype(jnp.float64),
            jnp.full((n_alpha,), -alpha_bound, dtype=jnp.float64),
        ])
        upper = jnp.concatenate([
            upper_theta.astype(jnp.float64),
            jnp.full((n_alpha,), alpha_bound, dtype=jnp.float64),
        ])

        self._log(
            f"SEES: {n_theta} structural + {n_alpha} basis params, "
            f"omega={omega}, basis_source={basis_metadata.get('basis_source')}"
        )

        result = minimize_lbfgsb(
            _neg_penalized_ll,
            x0,
            bounds=(lower, upper),
            maxiter=self._max_iter,
            tol=self._tol,
            verbose=self._verbose,
            desc="SEES L-BFGS-B",
            param_names=utility.parameter_names,
        )

        x_opt = jnp.array(result.x, dtype=jnp.float64)
        theta_opt = x_opt[:n_theta]
        alpha_opt = x_opt[n_theta:]

        self._log(f"theta: {np.asarray(theta_opt)}")
        self._log(f"alpha: {np.asarray(alpha_opt)}")

        # Compute final policy and value function
        flow_u = jnp.einsum("sak,k->sa", feature_matrix, theta_opt)
        continuation = beta * jnp.einsum(
            "ask,k->sa", expected_basis, alpha_opt
        )
        q_vals = flow_u + continuation
        policy = jax.nn.softmax(q_vals / sigma, axis=1)
        V = sigma * jax.scipy.special.logsumexp(q_vals / sigma, axis=1)

        # Compute pure log-likelihood (without penalty) for reporting
        log_probs = jax.nn.log_softmax(q_vals / sigma, axis=1)
        ll_opt = float(log_probs[obs_states, obs_actions].sum())

        # Bellman violation at solution
        bellman_residual = basis @ alpha_opt - V
        bellman_viol = float(jnp.max(jnp.abs(bellman_residual)))
        bellman_rmse = float(jnp.sqrt(jnp.mean(bellman_residual**2)))

        # Hessian for theta SEs via Schur complement (Corollary 3.1)
        # H = H_theta_theta - H'_beta_theta @ H_beta_beta^{-1} @ H_beta_theta
        hessian = None
        if self._compute_se:
            self._log("Computing Hessian for standard errors (Schur complement)")

            full_hessian = jax.hessian(penalized_criterion_sum)(
                jnp.array(x_opt, dtype=jnp.float64)
            )

            H_tt = full_hessian[:n_theta, :n_theta]
            H_ta = full_hessian[:n_theta, n_theta:]
            H_at = full_hessian[n_theta:, :n_theta]
            H_aa = full_hessian[n_theta:, n_theta:]

            try:
                hessian = H_tt - H_ta @ jnp.linalg.solve(H_aa, H_at)
                if not bool(jnp.all(jnp.isfinite(hessian))):
                    raise np.linalg.LinAlgError("non-finite Schur complement")
            except Exception:
                self._log("WARNING: H_aa singular, using ridge-regularized Schur complement")
                ridge = 1e-8 * jnp.eye(H_aa.shape[0], dtype=H_aa.dtype)
                hessian = H_tt - H_ta @ jnp.linalg.solve(H_aa - ridge, H_at)

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
                "basis_dim": n_alpha,
                "configured_basis_dim": self._basis_dim,
                "basis_matrix": basis,
                "bellman_violation": bellman_viol,
                "bellman_rmse": bellman_rmse,
                "penalty_weight": omega,
                "penalty_objective_scale": "mean_loglik_minus_omega_bellman_mse",
                "warm_start_value": self._warm_start_value,
                "initial_value_projection_rmse": projection_rmse,
                "initial_value_projection_converged": projection_converged,
                **basis_metadata,
            },
        )
