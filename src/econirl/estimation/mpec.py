"""MPEC (Mathematical Programming with Equilibrium Constraints) estimator.

This module implements the MPEC approach from Su and Judd (2012) for
estimating dynamic discrete choice models. Instead of nesting a
fixed-point solve inside the optimizer (as NFXP does), MPEC treats the
value function V as explicit decision variables optimized jointly with
the structural parameters theta.

The constrained problem is:

    min_{theta, V}  -L(theta, V)
    s.t.            V = T(V; theta)

where T is the soft Bellman operator. This is solved via augmented
Lagrangian, converting the constrained problem into a sequence of
unconstrained problems each solvable with L-BFGS-B and jax.grad.

Reference:
    Su, C.-L. and Judd, K.L. (2012). "Constrained Optimization Approaches
    to Estimation of Structural Models." Econometrica, 80(5), 2213-2230.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import UtilityFunction


@dataclass
class MPECConfig:
    """Configuration for the MPEC estimator.

    Attributes:
        solver: Optimization method. "slsqp" handles equality constraints
            natively via scipy SLSQP (recommended, matches Su-Judd and ruspy).
            "augmented_lagrangian" converts to unconstrained L-BFGS-B subproblems.
        max_iter: Maximum optimizer iterations (SLSQP) or inner iterations (AL).
        tol: Convergence tolerance for the optimizer.
        constraint_tol: Tolerance for Bellman constraint violation.
        rho_initial: Initial AL penalty weight (AL solver only).
        rho_max: Maximum AL penalty weight (AL solver only).
        rho_growth: Factor by which rho increases each AL outer iteration.
        outer_max_iter: Maximum AL outer iterations (AL solver only).
    """

    solver: str = "slsqp"
    max_iter: int = 500
    tol: float = 1e-10
    constraint_tol: float = 1e-8
    rho_initial: float = 1.0
    rho_max: float = 1e6
    rho_growth: float = 10.0
    outer_max_iter: int = 50
    # Backward compat aliases (map to max_iter / tol)
    inner_max_iter: int | None = None
    inner_tol: float | None = None

    def __post_init__(self):
        if self.inner_max_iter is not None:
            self.max_iter = self.inner_max_iter
        if self.inner_tol is not None:
            self.tol = self.inner_tol


class MPECEstimator(BaseEstimator):
    """MPEC estimator for dynamic discrete choice models.

    Avoids nested fixed-point solving by treating V as decision variables
    alongside theta (Su and Judd 2012). Two solver backends are available.

    SLSQP (default, recommended): Handles the Bellman equality constraint
    natively via sequential quadratic programming, matching the approach
    in ruspy and the spirit of the original paper. No penalty tuning needed.

    Augmented Lagrangian (fallback): Converts the constrained problem into
    a sequence of unconstrained L-BFGS-B subproblems with increasing
    penalty on the Bellman constraint violation.

    Both solvers recover the same MLE and produce identical standard errors
    at convergence because the Bellman constraint is satisfied exactly.

    Example:
        >>> estimator = MPECEstimator(verbose=True)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        config: MPECConfig | None = None,
        se_method: SEMethod = "asymptotic",
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the MPEC estimator.

        Args:
            config: MPEC configuration. If None, uses defaults.
            se_method: Method for computing standard errors
            compute_hessian: Whether to compute Hessian for standard errors
            verbose: Print progress messages
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        self._config = config or MPECConfig()

    @property
    def name(self) -> str:
        return "MPEC (Su-Judd)"

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run MPEC optimization."""
        cfg = self._config
        if cfg.solver == "slsqp":
            return self._optimize_slsqp(panel, utility, problem, transitions, initial_params)
        elif cfg.solver == "augmented_lagrangian":
            return self._optimize_al(panel, utility, problem, transitions, initial_params)
        else:
            raise ValueError(f"Unknown MPEC solver: {cfg.solver!r}. Use 'slsqp' or 'augmented_lagrangian'.")

    def _setup_common(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None,
    ):
        """Shared initialization for both solvers."""
        beta = problem.discount_factor
        sigma = problem.scale_parameter

        transitions_f64 = jnp.array(transitions, dtype=jnp.float64)
        features = jnp.array(utility.feature_matrix, dtype=jnp.float64)
        obs_states = panel.get_all_states()
        obs_actions = panel.get_all_actions()
        n_params = utility.num_parameters
        n_states = problem.num_states

        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            if bool(jnp.all(initial_params == 0)):
                initial_params = self._estimate_initial_params(
                    panel, utility, problem
                )
        theta = jnp.array(initial_params, dtype=jnp.float64)

        # Initialize V by solving Bellman at initial theta
        operator = SoftBellmanOperator(problem, transitions_f64)
        init_utility = jnp.einsum("sak,k->sa", features, theta)
        init_result = value_iteration(
            operator, init_utility, tol=1e-6, max_iter=10000
        )
        V = jnp.array(init_result.V, dtype=jnp.float64)

        # Bounds for theta
        lower, upper = utility.get_parameter_bounds()
        if lower is not None:
            theta_bounds = list(zip(np.asarray(lower), np.asarray(upper)))
        else:
            theta_bounds = [(None, None)] * n_params
        V_bounds = [(None, None)] * n_states
        bounds = theta_bounds + V_bounds

        return (beta, sigma, transitions_f64, features, obs_states, obs_actions,
                n_params, n_states, theta, V, operator, bounds)

    def _optimize_slsqp(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
    ) -> EstimationResult:
        """SLSQP solver with native equality constraints (recommended).

        Handles the Bellman constraint V = T(V; theta) directly within the
        optimizer, matching the approach in ruspy and the spirit of Su and
        Judd (2012). No augmented Lagrangian penalty tuning required.
        """
        start_time = time.time()
        cfg = self._config
        (beta, sigma, transitions_f64, features, obs_states, obs_actions,
         n_params, n_states, theta, V, operator, bounds) = self._setup_common(
            panel, utility, problem, transitions, initial_params)

        n_evals = 0

        # JIT-compiled JAX functions
        @jax.jit
        def _bellman_constraint(theta_V):
            th = theta_V[:n_params]
            v = theta_V[n_params:]
            u = jnp.einsum("sak,k->sa", features, th)
            EV = jnp.einsum("ast,t->as", transitions_f64, v)
            Q = u + beta * EV.T
            TV = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
            return v - TV

        @jax.jit
        def _neg_log_likelihood(theta_V):
            th = theta_V[:n_params]
            v = theta_V[n_params:]
            u = jnp.einsum("sak,k->sa", features, th)
            EV = jnp.einsum("ast,t->as", transitions_f64, v)
            Q = u + beta * EV.T
            log_probs = jax.nn.log_softmax(Q / sigma, axis=1)
            return -log_probs[obs_states, obs_actions].sum()

        _nll_grad = jax.jit(jax.grad(_neg_log_likelihood))
        _constraint_jac = jax.jit(jax.jacobian(_bellman_constraint))

        def obj_scipy(x):
            nonlocal n_evals
            n_evals += 1
            return float(_neg_log_likelihood(jnp.array(x, dtype=jnp.float64)))

        def grad_scipy(x):
            return np.asarray(_nll_grad(jnp.array(x, dtype=jnp.float64)))

        def constraint_fn(x):
            return np.asarray(_bellman_constraint(jnp.array(x, dtype=jnp.float64)))

        def constraint_jac_fn(x):
            return np.asarray(_constraint_jac(jnp.array(x, dtype=jnp.float64)))

        x0 = np.concatenate([np.asarray(theta), np.asarray(V)])

        self._log("Starting MPEC with SLSQP solver")

        result = optimize.minimize(
            obj_scipy, x0,
            method="SLSQP",
            jac=grad_scipy,
            bounds=bounds,
            constraints={
                "type": "eq",
                "fun": constraint_fn,
                "jac": constraint_jac_fn,
            },
            options={
                "maxiter": cfg.max_iter,
                "ftol": cfg.tol,
            },
        )

        theta = jnp.array(result.x[:n_params], dtype=jnp.float64)
        V = jnp.array(result.x[n_params:], dtype=jnp.float64)

        c = _bellman_constraint(jnp.array(result.x, dtype=jnp.float64))
        violation = float(jnp.abs(c).max())
        converged = violation < cfg.constraint_tol and result.success

        # Compute final policy and log-likelihood
        final_utility = jnp.einsum("sak,k->sa", features, theta)
        EV_final = jnp.einsum("ast,t->as", transitions_f64, V)
        Q_final = final_utility + beta * EV_final.T
        final_policy = jax.nn.softmax(Q_final / sigma, axis=1)
        log_probs_final = jax.nn.log_softmax(Q_final / sigma, axis=1)
        final_ll = float(log_probs_final[obs_states, obs_actions].sum())

        self._log(
            f"SLSQP finished: LL = {final_ll:.4f}, "
            f"|constraint| = {violation:.2e}, converged = {converged}"
        )

        # Standard errors via analytical score
        hessian = None
        gradient_contributions = None
        final_params = jnp.array(theta, dtype=jnp.float32)

        if self._compute_hessian:
            self._log("Computing standard errors via analytical score")
            scores = _compute_mpec_score(
                final_params, panel, features, operator,
                V, final_policy, beta, sigma,
            )
            gradient_contributions = scores
            hessian = -(scores.T @ scores)

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=final_params,
            log_likelihood=final_ll,
            value_function=jnp.array(V, dtype=jnp.float32),
            policy=jnp.array(final_policy, dtype=jnp.float32),
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=result.nit,
            num_function_evals=n_evals,
            message="converged" if converged else result.message,
            optimization_time=elapsed,
            metadata={
                "method": "slsqp",
                "final_constraint_violation": violation,
            },
        )

    def _optimize_al(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
    ) -> EstimationResult:
        """Augmented Lagrangian solver (fallback).

        Converts the constrained problem into a sequence of unconstrained
        L-BFGS-B subproblems with increasing penalty on constraint violation.
        """
        start_time = time.time()
        cfg = self._config
        (beta, sigma, transitions_f64, features, obs_states, obs_actions,
         n_params, n_states, theta, V, operator, bounds) = self._setup_common(
            panel, utility, problem, transitions, initial_params)

        lam = jnp.zeros(n_states, dtype=jnp.float64)
        rho = cfg.rho_initial

        def bellman_constraint(theta_V):
            th = theta_V[:n_params]
            v = theta_V[n_params:]
            u = jnp.einsum("sak,k->sa", features, th)
            EV = jnp.einsum("ast,t->as", transitions_f64, v)
            Q = u + beta * EV.T
            TV = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
            return v - TV

        def neg_log_likelihood(theta_V):
            th = theta_V[:n_params]
            v = theta_V[n_params:]
            u = jnp.einsum("sak,k->sa", features, th)
            EV = jnp.einsum("ast,t->as", transitions_f64, v)
            Q = u + beta * EV.T
            log_probs = jax.nn.log_softmax(Q / sigma, axis=1)
            return -log_probs[obs_states, obs_actions].sum()

        def augmented_lagrangian(theta_V, lam_val, rho_val):
            c = bellman_constraint(theta_V)
            nll = neg_log_likelihood(theta_V)
            return nll + jnp.dot(lam_val, c) + (rho_val / 2.0) * jnp.dot(c, c)

        al_grad_fn = jax.jit(jax.grad(augmented_lagrangian))

        converged = False
        n_inner_total = 0
        n_evals = 0
        violation = float("inf")

        self._log("Starting MPEC augmented Lagrangian optimization")

        for outer_iter in range(cfg.outer_max_iter):
            x0 = np.concatenate([np.asarray(theta), np.asarray(V)])
            lam_jax = jnp.array(lam, dtype=jnp.float64)
            rho_jax = jnp.float64(rho)

            def obj_scipy(x):
                nonlocal n_evals
                n_evals += 1
                return float(augmented_lagrangian(
                    jnp.array(x, dtype=jnp.float64), lam_jax, rho_jax
                ))

            def grad_scipy(x):
                g = al_grad_fn(jnp.array(x, dtype=jnp.float64), lam_jax, rho_jax)
                return np.asarray(g)

            result = optimize.minimize(
                obj_scipy, x0,
                method="L-BFGS-B",
                jac=grad_scipy,
                bounds=bounds,
                options={
                    "maxiter": cfg.max_iter,
                    "ftol": cfg.tol,
                    "gtol": cfg.tol,
                },
            )
            n_inner_total += result.nit

            theta = jnp.array(result.x[:n_params], dtype=jnp.float64)
            V = jnp.array(result.x[n_params:], dtype=jnp.float64)

            c = bellman_constraint(jnp.array(result.x, dtype=jnp.float64))
            violation = float(jnp.abs(c).max())

            u_current = jnp.einsum("sak,k->sa", features, theta)
            EV = jnp.einsum("ast,t->as", transitions_f64, V)
            Q = u_current + beta * EV.T
            log_probs = jax.nn.log_softmax(Q / sigma, axis=1)
            ll = float(log_probs[obs_states, obs_actions].sum())

            self._log(
                f"AL iter {outer_iter+1}: LL = {ll:.4f}, "
                f"|constraint| = {violation:.2e}, rho = {rho:.1e}"
            )

            if violation < cfg.constraint_tol:
                converged = True
                self._log(
                    f"MPEC converged at AL iter {outer_iter+1}: "
                    f"constraint violation = {violation:.2e}"
                )
                break

            lam = lam + rho * c
            rho = min(rho * cfg.rho_growth, cfg.rho_max)

        final_utility = jnp.einsum("sak,k->sa", features, theta)
        EV_final = jnp.einsum("ast,t->as", transitions_f64, V)
        Q_final = final_utility + beta * EV_final.T
        final_policy = jax.nn.softmax(Q_final / sigma, axis=1)
        log_probs_final = jax.nn.log_softmax(Q_final / sigma, axis=1)
        final_ll = float(log_probs_final[obs_states, obs_actions].sum())

        hessian = None
        gradient_contributions = None
        final_params = jnp.array(theta, dtype=jnp.float32)

        if self._compute_hessian:
            self._log("Computing standard errors via analytical score")
            scores = _compute_mpec_score(
                final_params, panel, features, operator,
                V, final_policy, beta, sigma,
            )
            gradient_contributions = scores
            hessian = -(scores.T @ scores)

        elapsed = time.time() - start_time

        return EstimationResult(
            parameters=final_params,
            log_likelihood=final_ll,
            value_function=jnp.array(V, dtype=jnp.float32),
            policy=jnp.array(final_policy, dtype=jnp.float32),
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=outer_iter + 1,
            num_function_evals=n_evals,
            num_inner_iterations=n_inner_total,
            message="converged" if converged else "max iterations reached",
            optimization_time=elapsed,
            metadata={
                "method": "augmented_lagrangian",
                "final_constraint_violation": violation,
                "final_rho": rho,
            },
        )

    def _estimate_initial_params(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> jnp.ndarray:
        """Estimate rough starting values from data."""
        n_params = utility.num_parameters
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        total_obs = all_states.shape[0]
        replace_mask = all_actions == 1
        n_replace = int(replace_mask.sum())

        if n_replace > 0 and total_obs > 0:
            avg_mileage = float(all_states[replace_mask].astype(jnp.float32).sum()) / n_replace
            replace_rate = n_replace / total_obs
            n_states = problem.num_states
            op_cost_init = 1.0 / n_states
            rc_init = max(0.5, op_cost_init * avg_mileage / max(replace_rate, 0.01))

            init = np.full(n_params, 0.01, dtype=np.float32)
            init[0] = op_cost_init
            for i in range(1, n_params - 1):
                init[i] = op_cost_init * 0.1 ** i
            init[-1] = rc_init
            return jnp.array(init, dtype=jnp.float32)

        return jnp.full((n_params,), 0.01, dtype=jnp.float32)


def _compute_mpec_score(
    params: jnp.ndarray,
    panel: Panel,
    features: jnp.ndarray,
    operator: SoftBellmanOperator,
    V: jnp.ndarray,
    policy: jnp.ndarray,
    beta: float,
    sigma: float,
) -> jnp.ndarray:
    """Compute per-observation score for MPEC standard errors.

    At convergence V satisfies the Bellman equation, so the analytical
    score formula is identical to NFXP. Uses the implicit function
    theorem: (I - beta * P_pi) * dV/dtheta = sum_a pi(a|s) * dphi/dtheta.
    """
    features = jnp.array(features, dtype=jnp.float64)
    policy = jnp.array(policy, dtype=jnp.float64)
    V = jnp.array(V, dtype=jnp.float64)

    # F = I - beta * P_pi
    P_pi = jnp.einsum("sa,ast->st", policy, operator.transitions)
    n = operator.problem.num_states
    F = jnp.eye(n, dtype=jnp.float64) - beta * P_pi

    # dT/dtheta[s,k] = sum_a pi(a|s) * phi(s,a,k)
    dT_dtheta = jnp.einsum("sa,sak->sk", policy, features)

    # Solve F @ dV/dtheta = dT/dtheta
    dV_dtheta = jnp.linalg.solve(F, dT_dtheta)

    # dQ/dtheta[s,a,k] = phi(s,a,k) + beta * sum_s' P(s'|s,a) * dV(s')/dtheta_k
    EV_deriv = jnp.einsum("ast,tk->ask", operator.transitions, dV_dtheta)
    dQ_dtheta = features + beta * jnp.transpose(EV_deriv, (1, 0, 2))

    # E_pi[dQ] = sum_a pi(a|s) * dQ(s,a,k)
    E_dQ = jnp.einsum("sa,sak->sk", policy, dQ_dtheta)

    # Per-observation score
    all_states = panel.get_all_states()
    all_actions = panel.get_all_actions()
    dQ_obs = dQ_dtheta[all_states, all_actions]
    E_dQ_obs = E_dQ[all_states]
    scores = (1.0 / sigma) * (dQ_obs - E_dQ_obs)

    return scores.astype(jnp.float32)
