"""MPEC (Mathematical Programming with Equilibrium Constraints) estimator.

This module implements the MPEC approach from Su and Judd (2012) for
estimating dynamic discrete choice models. Instead of nesting a
fixed-point solve inside the optimizer (as NFXP does), MPEC treats the
value function V as explicit decision variables optimized jointly with
the structural parameters theta.

The constrained problem is:

    min_{theta, V}  -L(theta, V)
    s.t.            V = T(V; theta)

where T is the soft Bellman operator.

Two solvers are provided:

  "sqp" (default):  Sequential Quadratic Programming. At each outer
      iteration, linearise the Bellman constraint and solve the
      resulting equality-constrained QP via the full KKT system
      (JAX-native, jnp.linalg.solve).  A damped BFGS Hessian
      approximation is maintained.  An L1 merit function drives
      backtracking line search.  This is the JAX-ecosystem alternative
      to KNITRO's interior-point engine used in Su & Judd (2012).

  "augmented_lagrangian" (legacy): Penalty-based AL with jaxopt
      L-BFGS-B inner solves.  Kept for backward compatibility but
      does not converge reliably at beta >= 0.99.

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

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod
from econirl.preferences.base import UtilityFunction


def _mpec_warmup_nll(theta_only, V_warm, features, transitions_f64,
                     obs_states, obs_actions, beta, sigma):
    u = jnp.einsum("sak,k->sa", features, theta_only)
    EV = jnp.einsum("ast,t->as", transitions_f64, V_warm)
    Q = u + beta * EV.T
    lp = jax.nn.log_softmax(Q / sigma, axis=1)
    return -lp[obs_states, obs_actions].sum()


_mpec_warmup_nll_jit = jax.jit(_mpec_warmup_nll)


@dataclass
class MPECConfig:
    """Configuration for the MPEC estimator.

    Attributes:
        solver: "sqp" (default, recommended) uses Sequential Quadratic
            Programming with a JAX-native KKT solver and damped BFGS
            Hessian.  "augmented_lagrangian" / "slsqp" are legacy aliases
            for the penalty-based AL solver (less reliable at high beta).
        outer_max_iter: Maximum SQP / AL outer iterations.
        tol: KKT stationarity tolerance (SQP) or inner L-BFGS-B tol (AL).
        constraint_tol: Maximum Bellman constraint violation at convergence.
        max_iter: Inner L-BFGS-B iterations per AL step (AL solver only).
        rho_initial: Initial AL penalty weight (AL solver only).
        rho_max: Maximum AL penalty weight (AL solver only).
        rho_growth: AL penalty growth factor (AL solver only).
    """

    solver: str = "sqp"
    outer_max_iter: int = 200
    tol: float = 1e-6
    constraint_tol: float = 1e-5
    # AL-specific (ignored by SQP)
    max_iter: int = 500
    rho_initial: float = 1.0
    rho_max: float = 1e6
    rho_growth: float = 10.0
    # Backward compat aliases
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
    alongside theta (Su and Judd 2012). The Bellman equality constraint
    V = T(V; theta) is enforced via augmented Lagrangian: a sequence of
    unconstrained jaxopt L-BFGS-B inner subproblems with increasing
    penalty on the constraint violation. Both "slsqp" and
    "augmented_lagrangian" config values route to this path. No scipy
    dependency is required.

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
        if cfg.solver == "sqp":
            return self._optimize_sqp(panel, utility, problem, transitions, initial_params)
        elif cfg.solver in ("slsqp", "augmented_lagrangian"):
            return self._optimize_al(panel, utility, problem, transitions, initial_params)
        else:
            raise ValueError(
                f"Unknown MPEC solver: {cfg.solver!r}. "
                "Use 'sqp' (recommended) or 'augmented_lagrangian'."
            )

    def _optimize_sqp(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
    ) -> EstimationResult:
        """MPEC via Sequential Quadratic Programming (JAX-native).

        At each outer iteration:
          1. Compute objective gradient g and Bellman constraint c, Jacobian J.
          2. Solve KKT system [H J^T; J 0] [d; mu] = [-g; -c] for search
             direction d and updated multipliers mu.
          3. Backtracking line search on an L1 merit function.
          4. Damped BFGS update of the Lagrangian Hessian approximation H.

        Convergence when max(|c|) < constraint_tol AND max(|g + J^T mu|) < tol.
        """
        start_time = time.time()
        cfg = self._config
        (beta, sigma, transitions_f64, features, obs_states, obs_actions,
         n_params, n_states, theta, V, operator, _) = self._setup_common(
            panel, utility, problem, transitions, initial_params)

        n_vars = n_params + n_states

        # --- JAX-native functions (JIT-compiled once) ---

        @jax.jit
        def bellman_residual(x):
            th, v = x[:n_params], x[n_params:]
            u = jnp.einsum("sak,k->sa", features, th)
            EV = jnp.einsum("ast,t->as", transitions_f64, v)
            Q = u + beta * EV.T
            TV = sigma * jax.scipy.special.logsumexp(Q / sigma, axis=1)
            return v - TV  # shape (n_states,)

        @jax.jit
        def neg_log_likelihood(x):
            th, v = x[:n_params], x[n_params:]
            u = jnp.einsum("sak,k->sa", features, th)
            EV = jnp.einsum("ast,t->as", transitions_f64, v)
            Q = u + beta * EV.T
            lp = jax.nn.log_softmax(Q / sigma, axis=1)
            return -lp[obs_states, obs_actions].sum()

        grad_fn = jax.jit(jax.value_and_grad(neg_log_likelihood))
        jac_fn = jax.jit(jax.jacobian(bellman_residual))  # (n_states, n_vars)

        # L1 merit function for line search
        @jax.jit
        def merit(x, penalty):
            return neg_log_likelihood(x) + penalty * jnp.abs(bellman_residual(x)).sum()

        # Damped BFGS update (maintains positive definiteness)
        def _bfgs_update(H, s, y):
            Hs = H @ s
            sHs = jnp.dot(s, Hs)
            sy = jnp.dot(s, y)
            # Damping: ensure sy >= 0.2 * s'Hs
            theta_damp = jnp.where(sy >= 0.2 * sHs, 1.0, 0.8 * sHs / (sHs - sy))
            y_damp = theta_damp * y + (1.0 - theta_damp) * Hs
            sy_damp = jnp.dot(s, y_damp)
            rho = 1.0 / jnp.where(jnp.abs(sy_damp) < 1e-12, 1e-12, sy_damp)
            I = jnp.eye(n_vars)
            A = I - rho * jnp.outer(s, y_damp)
            return A @ H @ A.T + rho * jnp.outer(s, s)

        # --- Warm-up: JAXopt LBFGS on LL(theta | V fixed) ---
        # Optimising theta with V held fixed at the initial Bellman FP
        # moves theta close to the MLE before the constrained SQP begins.
        # This avoids the spurious local optima that arise when SQP starts
        # far from the truth with V pinned at the wrong equilibrium.
        self._log("Warm-up: LBFGS on theta with V fixed (JAXopt)")
        try:
            from jaxopt import LBFGS as JAXoptLBFGS

            _warmup = JAXoptLBFGS(fun=_mpec_warmup_nll_jit, maxiter=50, tol=1e-6)
            _ws = _warmup.run(
                theta, V, features, transitions_f64,
                obs_states, obs_actions, beta, sigma,
            )
            theta = jnp.array(_ws.params, dtype=jnp.float64)
            # Re-solve V at warmed-up theta so the starting (theta, V) is feasible
            _warm_u = jnp.einsum("sak,k->sa", features, theta)
            _warm_vi = value_iteration(operator, _warm_u, tol=1e-8, max_iter=10000)
            V = jnp.array(_warm_vi.V, dtype=jnp.float64)
            self._log("Warm-up complete")
        except Exception as e:
            self._log(f"Warm-up skipped ({e})")

        # --- Initialise SQP ---
        x = jnp.concatenate([theta, V])
        H = jnp.eye(n_vars, dtype=jnp.float64)  # BFGS Hessian approximation
        mu = jnp.zeros(n_states, dtype=jnp.float64)  # Lagrange multipliers

        converged = False
        n_evals = 0
        violation = float("inf")
        kkt_err = float("inf")
        ll = float(-neg_log_likelihood(x))

        self._log("Starting MPEC SQP optimisation")

        from tqdm import tqdm
        pbar = tqdm(
            range(cfg.outer_max_iter),
            desc="MPEC SQP",
            disable=not self._verbose,
            leave=True,
        )

        for outer_iter in pbar:
            nll, g = grad_fn(x)
            c = bellman_residual(x)
            J = jac_fn(x)  # (n_states, n_vars)
            n_evals += 1

            violation = float(jnp.abs(c).max())
            # Lagrangian stationarity: g + J^T mu
            kkt_err = float(jnp.abs(g + J.T @ mu).max())
            ll = float(-nll)

            pbar.set_postfix({
                "LL": f"{ll:.2f}",
                "|c|": f"{violation:.1e}",
                "|KKT|": f"{kkt_err:.1e}",
            })
            self._log(
                f"SQP iter {outer_iter+1}: LL={ll:.4f} |c|={violation:.2e} |KKT|={kkt_err:.2e}"
            )

            if violation < cfg.constraint_tol and kkt_err < cfg.tol:
                converged = True
                self._log(f"MPEC SQP converged at iter {outer_iter+1}")
                pbar.close()
                break

            # --- KKT system solve ---
            # [H  J^T] [d ] = [-g]
            # [J  0  ] [mu]   [-c]
            # Regularise H for numerical stability
            H_reg = H + 1e-6 * jnp.eye(n_vars)
            KKT = jnp.block([
                [H_reg,                        J.T],
                [J,    jnp.zeros((n_states, n_states))],
            ])
            rhs = jnp.concatenate([-g, -c])
            try:
                sol = jnp.linalg.solve(KKT, rhs)
                d = sol[:n_vars]
                mu_new = sol[n_vars:]
            except Exception:
                # Fallback: gradient descent step
                d = -g
                mu_new = mu

            # --- L1 merit line search ---
            penalty = float(jnp.max(jnp.abs(mu_new))) + 1.0
            merit_0 = float(merit(x, penalty))
            # Directional derivative of merit at x
            dd_merit = float(g @ d) - penalty * float(jnp.abs(c).sum())

            alpha = 1.0
            x_new = x + alpha * d
            for _ in range(25):
                m_new = float(merit(x_new, penalty))
                if m_new <= merit_0 + 1e-4 * alpha * dd_merit:
                    break
                alpha *= 0.5
                x_new = x + alpha * d
            n_evals += _  # approximate

            # --- BFGS Hessian update ---
            nll_new, g_new = grad_fn(x_new)
            c_new = bellman_residual(x_new)
            J_new = jac_fn(x_new)
            n_evals += 1

            s_step = x_new - x
            lag_grad_old = g + J.T @ mu
            lag_grad_new = g_new + J_new.T @ mu_new
            y_step = lag_grad_new - lag_grad_old

            if float(jnp.abs(s_step).max()) > 1e-14:
                H = _bfgs_update(H, s_step, y_step)

            x = x_new
            mu = mu_new

        # Final values
        theta_final = x[:n_params].astype(jnp.float32)
        V_final = x[n_params:]
        c_final = bellman_residual(x)
        violation_final = float(jnp.abs(c_final).max())

        u_final = jnp.einsum("sak,k->sa", features, x[:n_params])
        EV_final = jnp.einsum("ast,t->as", transitions_f64, V_final)
        Q_final = u_final + beta * EV_final.T
        final_policy = jax.nn.softmax(Q_final / sigma, axis=1)
        lp_final = jax.nn.log_softmax(Q_final / sigma, axis=1)
        final_ll = float(lp_final[obs_states, obs_actions].sum())

        hessian = None
        gradient_contributions = None
        if self._compute_hessian:
            self._log("Computing standard errors via analytical score")
            scores = _compute_mpec_score(
                theta_final, panel, features, operator,
                V_final, final_policy, beta, sigma,
            )
            gradient_contributions = scores
            hessian = -(scores.T @ scores)

        elapsed = time.time() - start_time
        return EstimationResult(
            parameters=theta_final,
            log_likelihood=final_ll,
            value_function=V_final.astype(jnp.float32),
            policy=final_policy.astype(jnp.float32),
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged,
            num_iterations=outer_iter + 1,
            num_function_evals=n_evals,
            num_inner_iterations=0,  # no inner Bellman loop
            message="converged" if converged else "max iterations reached",
            optimization_time=elapsed,
            metadata={
                "method": "sqp",
                "final_constraint_violation": violation_final,
                "final_kkt_error": kkt_err,
            },
        )

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

        # Build bounds arrays once — they do not change across AL outer iterations
        lo = jnp.array(
            [b[0] if b[0] is not None else -jnp.inf for b in bounds],
            dtype=jnp.float64,
        )
        hi = jnp.array(
            [b[1] if b[1] is not None else jnp.inf for b in bounds],
            dtype=jnp.float64,
        )

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

        from tqdm import tqdm
        pbar = tqdm(
            range(cfg.outer_max_iter),
            desc="MPEC AL",
            disable=not self._verbose,
            leave=True,
        )
        for outer_iter in pbar:
            x0 = jnp.concatenate([theta, V])
            lam_jax = jnp.array(lam, dtype=jnp.float64)
            rho_jax = jnp.float64(rho)

            def al_val_and_grad(x):
                val = augmented_lagrangian(x, lam_jax, rho_jax)
                grad = al_grad_fn(x, lam_jax, rho_jax)
                return val, grad

            result = minimize_lbfgsb(
                al_val_and_grad,
                x0,
                bounds=(lo, hi),
                maxiter=cfg.max_iter,
                tol=cfg.tol,
                verbose=False,
                desc=f"MPEC AL inner (outer {outer_iter + 1})",
                value_and_grad=True,
            )
            n_evals += result.nfev
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

            pbar.set_postfix({"LL": f"{ll:.2f}", "|c|": f"{violation:.1e}", "rho": f"{rho:.0e}"})
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
