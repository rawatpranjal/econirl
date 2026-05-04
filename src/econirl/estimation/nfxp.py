"""Nested Fixed Point (NFXP) estimator.

This module implements the NFXP algorithm from Rust (1987, 1988) for
estimating dynamic discrete choice models, with the SA then NK polyalgorithm
from Iskhakov, Jorgensen, Rust and Schjerning (2016).

Algorithm:
    Outer loop: Maximize log-likelihood via BHHH or L-BFGS-B
    Inner loop: Solve Bellman equation via SA then NK polyalgorithm
    Gradient: Automatic via jax.grad through optimistix implicit differentiation

The log-likelihood is:
    L(theta) = sum_i sum_t log P(a_{it} | s_{it}; theta)

where choice probabilities come from the logit model:
    P(a|s; theta) = exp(Q(s,a;theta)/sigma) / sum_{a'} exp(Q(s,a';theta)/sigma)

References:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines"
    Iskhakov et al. (2016). "Comment on Constrained Optimization
        Approaches to Estimation of Structural Models." Econometrica.
    Blondel et al. (2022). "Efficient and Modular Implicit Differentiation."
"""

from __future__ import annotations

import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
from econirl.core.solvers import (
    backward_induction,
    hybrid_iteration,
    optimistix_solve,
    policy_iteration,
    value_iteration,
)
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


def estimate_transitions_from_panel(
    panel: Panel,
    num_states: int,
    max_increment: int = 2,
) -> jnp.ndarray:
    """Estimate mileage transition probabilities from panel data.

    First-stage estimator: counts mileage increments from keep-action
    observations to get P(delta=0), P(delta=1), P(delta=2).

    Args:
        panel: Panel with observed states and actions
        num_states: Number of discrete states
        max_increment: Maximum mileage increment (default 2)

    Returns:
        Transition matrix of shape (num_actions, num_states, num_states)
    """
    counts = np.zeros(max_increment + 1)

    for traj in panel.trajectories:
        for t in range(len(traj.states) - 1):
            if int(traj.actions[t]) == 0:  # keep action
                inc = int(traj.states[t + 1]) - int(traj.states[t])
                if inc >= 0:
                    inc = min(inc, max_increment)
                    counts[inc] += 1

    if counts.sum() == 0:
        probs = np.ones(max_increment + 1) / (max_increment + 1)
    else:
        probs = counts / counts.sum()

    # Build transition matrices
    n = num_states
    trans = np.zeros((2, n, n), dtype=np.float64)

    for a in range(2):
        for s in range(n):
            src = 0 if a == 1 else s
            for k, p in enumerate(probs):
                dest = min(src + k, n - 1)
                trans[a, s, dest] += p

    return jnp.array(trans)


class NFXPEstimator(BaseEstimator):
    """Nested Fixed Point estimator for dynamic discrete choice models.

    Implements the Iskhakov et al. (2016) SA then NK polyalgorithm with
    automatic gradients via JAX implicit differentiation through the
    Bellman fixed point (Blondel et al. 2022).

    For each candidate parameter vector theta:
    1. Compute flow utility matrix U(s,a; theta)
    2. Solve for value function V(s; theta) via SA then NK polyalgorithm
    3. Compute choice probabilities P(a|s; theta)
    4. Evaluate log-likelihood and gradient (automatic via jax.grad)

    Example:
        >>> estimator = NFXPEstimator(optimizer="BHHH", verbose=True)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        se_method: SEMethod = "asymptotic",
        optimizer: Literal["L-BFGS-B", "BFGS", "BHHH"] = "BHHH",
        inner_solver: Literal[
            "sa", "nk", "polyalgorithm", "optimistix",
            "value", "policy", "hybrid",
        ] = "polyalgorithm",
        inner_tol: float = 1e-12,
        inner_max_iter: int = 100000,
        switch_tol: float = 1e-3,
        outer_tol: float = 1e-6,
        outer_max_iter: int = 1000,
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the NFXP estimator.

        Args:
            se_method: Method for computing standard errors
            optimizer: Optimizer for outer loop.
                - "BHHH": Berndt-Hall-Hall-Hausman (uses per-observation scores)
                - "L-BFGS-B": Scipy L-BFGS-B with bounds
                - "BFGS": Scipy BFGS
            inner_solver: Solver for inner fixed-point problem. Canonical names
                follow Iskhakov et al. (2016).
                - "sa": Successive approximation (pure contraction)
                - "nk": Newton-Kantorovich iteration on the Bellman residual
                - "polyalgorithm": SA then NK polyalgorithm (default)
                - "optimistix": Optimistix fixed-point with implicit differentiation
                Legacy aliases "value", "policy", "hybrid" map to "sa", "nk",
                "polyalgorithm" respectively.
            inner_tol: Final convergence tolerance for inner solver
            inner_max_iter: Max iterations for inner solver
            switch_tol: SA to NK switch tolerance (polyalgorithm only)
            outer_tol: Gradient tolerance for outer optimization
            outer_max_iter: Max outer optimization iterations
            compute_hessian: Whether to compute Hessian for standard errors
            verbose: Print progress messages
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        self._optimizer = optimizer
        _solver_aliases = {
            "sa": "value",
            "nk": "policy",
            "polyalgorithm": "hybrid",
        }
        self._inner_solver = _solver_aliases.get(inner_solver, inner_solver)
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._switch_tol = switch_tol
        self._outer_tol = outer_tol
        self._outer_max_iter = outer_max_iter

    @property
    def name(self) -> str:
        return "NFXP (Nested Fixed Point)"

    def _solve_inner(
        self,
        operator: SoftBellmanOperator,
        flow_utility: jnp.ndarray,
    ):
        """Solve the inner dynamic programming problem."""
        if self._inner_solver == "policy":
            return policy_iteration(
                operator, flow_utility,
                tol=self._inner_tol, max_iter=self._inner_max_iter,
                eval_method="matrix",
            )
        elif self._inner_solver == "hybrid":
            return hybrid_iteration(
                operator, flow_utility,
                tol=self._inner_tol, max_iter=self._inner_max_iter,
                switch_tol=self._switch_tol,
            )
        elif self._inner_solver == "optimistix":
            V = optimistix_solve(
                operator.problem, operator.transitions, flow_utility,
                tol=self._inner_tol, max_steps=self._inner_max_iter,
            )
            result = operator.apply(flow_utility, V)
            from econirl.core.solvers import SolverResult
            return SolverResult(
                Q=result.Q, V=result.V, policy=result.policy,
                converged=True, num_iterations=0, final_error=0.0,
            )
        else:
            return value_iteration(
                operator, flow_utility,
                tol=self._inner_tol, max_iter=self._inner_max_iter,
            )

    def _make_log_likelihood_fn(
        self,
        features: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        obs_states: jnp.ndarray,
        obs_actions: jnp.ndarray,
    ):
        """Create a differentiable log-likelihood function.

        Returns a function theta -> scalar LL that can be differentiated
        with jax.grad and jax.hessian. Uses optimistix.fixed_point with
        ImplicitAdjoint for automatic gradient through the Bellman fixed point.
        """
        beta = problem.discount_factor
        sigma = problem.scale_parameter

        def log_likelihood(theta):
            utility = jnp.einsum("sak,k->sa", features, theta)
            V = optimistix_solve(problem, transitions, utility,
                                 tol=self._inner_tol,
                                 max_steps=self._inner_max_iter)
            log_probs = _compute_log_probs(utility, V, transitions, beta, sigma)
            return log_probs[obs_states, obs_actions].sum()

        return log_likelihood

    def _compute_analytical_score(
        self,
        params: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        operator: SoftBellmanOperator,
        V: jnp.ndarray,
        policy: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """Compute analytical per-observation score via implicit differentiation.

        Uses the implicit function theorem to compute dV/dtheta without
        differentiating through the fixed-point iteration:
            (I - beta*P_pi) * dV/dtheta = sum_a pi(a|s) * dU(s,a)/dtheta

        Args:
            params: Current parameter vector
            panel: Observed data
            utility: Utility specification (must have feature_matrix)
            operator: Bellman operator
            V: Converged value function
            policy: Converged choice probabilities

        Returns:
            Tuple of (per_obs_scores, log_likelihood) where
            per_obs_scores has shape (n_obs, n_params)
        """
        beta = operator.problem.discount_factor
        sigma = operator.problem.scale_parameter

        features = jnp.array(utility.feature_matrix, dtype=jnp.float64)

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

        # Log-likelihood
        flow_utility = utility.compute(params)
        log_probs = operator.compute_log_choice_probabilities(
            jnp.array(flow_utility, dtype=jnp.float64), V
        )
        ll = float(log_probs[all_states, all_actions].sum())

        return scores.astype(jnp.float32), ll

    def _bhhh_optimize(
        self,
        initial_params: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        operator: SoftBellmanOperator,
    ) -> tuple[jnp.ndarray, float, int, int, int, bool]:
        """Run BHHH optimization with analytical gradient."""
        params = jnp.array(initial_params, dtype=jnp.float32)
        n_params = len(params)
        total_inner = 0
        n_evals = 0
        converged = False
        prev_ll = -float("inf")
        ll = prev_ll

        from tqdm import tqdm
        pbar = tqdm(
            range(self._outer_max_iter),
            desc="NFXP BHHH",
            disable=not self._verbose,
            leave=True,
        )
        for iteration in pbar:
            flow_utility = jnp.array(utility.compute(params), dtype=jnp.float64)
            solver_result = self._solve_inner(operator, flow_utility)
            total_inner += solver_result.num_iterations

            scores, ll = self._compute_analytical_score(
                params, panel, utility, operator, solver_result.V, solver_result.policy
            )
            n_evals += 1

            grad = scores.sum(axis=0)
            grad_norm = float(jnp.abs(grad).max())
            ll_change = abs(ll - prev_ll) if prev_ll > -float("inf") else float("inf")

            postfix = {"LL": f"{ll:.2f}", "|g|": f"{grad_norm:.1e}", "dLL": f"{ll_change:.1e}"}
            for j, nm in enumerate(utility.parameter_names[:3]):
                postfix[nm] = f"{float(params[j]):.5f}"
            pbar.set_postfix(postfix)

            if grad_norm < self._outer_tol or (iteration > 10 and ll_change < 1e-10):
                converged = True
                pbar.set_postfix({**postfix, "status": "converged"})
                pbar.close()
                self._log(f"BHHH converged at iter {iteration+1}: |grad| = {grad_norm:.2e}")
                break

            prev_ll = ll

            H_bhhh = scores.T @ scores + 1e-8 * jnp.eye(n_params)
            direction = jnp.linalg.solve(H_bhhh, grad)
            if not bool(jnp.all(jnp.isfinite(direction))):
                direction = grad

            # Step-halving line search
            step_size = 1.0
            for _ in range(15):
                new_params = params + step_size * direction
                flow_u_new = jnp.array(utility.compute(new_params), dtype=jnp.float64)
                solver_new = self._solve_inner(operator, flow_u_new)
                total_inner += solver_new.num_iterations
                n_evals += 1

                log_probs = operator.compute_log_choice_probabilities(
                    flow_u_new, solver_new.V
                )
                all_s = panel.get_all_states()
                all_a = panel.get_all_actions()
                new_ll = float(log_probs[all_s, all_a].sum())

                if new_ll > ll:
                    break
                step_size *= 0.5

            params = new_params

        return params, ll, iteration + 1, n_evals, total_inner, converged

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
        mileage_at_replace = float(all_states[replace_mask].astype(jnp.float32).sum())

        if n_replace > 0 and total_obs > 0:
            replace_rate = n_replace / total_obs
            avg_mileage = mileage_at_replace / n_replace
            n_states = problem.num_states
            op_cost_init = 1.0 / n_states
            rc_init = max(0.5, op_cost_init * avg_mileage / max(replace_rate, 0.01))

            # Last parameter is always replacement cost. Fill operating cost
            # coefficients with decreasing magnitudes for higher-order terms.
            init = np.full(n_params, 0.01, dtype=np.float32)
            init[0] = op_cost_init
            for i in range(1, n_params - 1):
                init[i] = op_cost_init * 0.1 ** i
            init[-1] = rc_init
            return jnp.array(init, dtype=jnp.float32)

        return jnp.full((n_params,), 0.01, dtype=jnp.float32)

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
        initial_params: jnp.ndarray | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run NFXP optimization."""
        import warnings

        # Allow kwargs to override outer_max_iter for warm-start bootstrap
        outer_max_iter_override = kwargs.pop("outer_max_iter", None)
        saved_outer_max_iter = self._outer_max_iter
        if outer_max_iter_override is not None:
            self._outer_max_iter = outer_max_iter_override

        start_time = time.time()

        beta = problem.discount_factor
        if beta > 0.99 and self._inner_max_iter < 50000 and self._inner_solver == "value":
            warnings.warn(
                f"High discount factor beta={beta} may require inner_max_iter > 50000.",
                UserWarning,
            )

        # Initialize parameters
        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            if bool(jnp.all(initial_params == 0)):
                initial_params = self._estimate_initial_params(panel, utility, problem)

        transitions_f64 = jnp.array(transitions, dtype=jnp.float64)
        operator = SoftBellmanOperator(problem, transitions_f64)

        finite_horizon = problem.num_periods is not None

        if finite_horizon:
            # Finite-horizon via backward induction + JAX-native L-BFGS-B
            num_periods = problem.num_periods
            self._log(f"Starting finite-horizon NFXP ({num_periods} periods)")
            total_inner = 0
            n_evals = 0

            obs_states_fh = panel.get_all_states()
            obs_actions_fh = panel.get_all_actions()

            # Pure JAX objective so minimize_lbfgsb can differentiate through it.
            def neg_ll_jax_fh(params):
                nonlocal total_inner, n_evals
                n_evals += 1
                params = jnp.asarray(params, dtype=jnp.float32)
                flow_u = jnp.array(utility.compute(params), dtype=jnp.float64)
                utility_seq = jnp.stack([flow_u] * num_periods)
                fh_result = backward_induction(operator, utility_seq)
                sigma = problem.scale_parameter
                log_policy = jax.nn.log_softmax(fh_result.Q / sigma, axis=1)
                total_inner += num_periods
                return -log_policy[obs_states_fh, obs_actions_fh].sum()

            lower, upper = utility.get_parameter_bounds()
            fh_bounds = (jnp.asarray(lower), jnp.asarray(upper)) if lower is not None else None

            result = minimize_lbfgsb(
                neg_ll_jax_fh,
                jnp.asarray(initial_params, dtype=jnp.float64),
                bounds=fh_bounds,
                maxiter=self._outer_max_iter,
                tol=self._outer_tol,
                verbose=self._verbose,
                desc="NFXP FH",
                param_names=list(utility.parameter_names),
            )

            final_params = jnp.array(result.x, dtype=jnp.float32)
            ll = -result.fun
            n_iter = result.nit
            opt_converged = result.success

        elif self._optimizer == "BHHH":
            self._log("Starting BHHH optimization with analytical gradient")
            params, ll, n_iter, n_evals, total_inner, opt_converged = self._bhhh_optimize(
                initial_params, panel, utility, operator,
            )
            final_params = params

        else:
            # Scipy optimizer with jax.grad for automatic gradient.
            # EV cache used by _solve_cached. minimize_lbfgsb calls jax.grad
            # internally, which re-evaluates the objective. Cache the Bellman
            # solution to avoid solving twice per outer iteration.
            total_inner = 0
            n_evals = 0
            ev_cache = {}  # mutable dict used as cache between closures

            features = jnp.array(utility.feature_matrix, dtype=jnp.float64)
            obs_states = panel.get_all_states()
            obs_actions = panel.get_all_actions()

            def _solve_cached(params):
                """Solve Bellman with caching on parameter vector."""
                params_key = tuple(np.asarray(params).ravel().tolist())
                if ev_cache.get("key") == params_key:
                    return ev_cache["V"]
                u = jnp.einsum("sak,k->sa", features, params)
                V = optimistix_solve(problem, transitions_f64, u,
                                     tol=self._inner_tol, max_steps=self._inner_max_iter)
                ev_cache["key"] = params_key
                ev_cache["V"] = V
                return V

            # Automatic gradient via jax.grad through the optimistix fixed point.
            # minimize_lbfgsb calls jax.grad internally, so pass the pure JAX
            # log-likelihood directly (negated for minimization).
            ll_fn = self._make_log_likelihood_fn(
                features, transitions_f64, problem, obs_states, obs_actions,
            )

            def neg_ll_jax(params):
                nonlocal n_evals
                n_evals += 1
                return -ll_fn(params)

            self._log(f"Starting optimization with {self._optimizer}")
            ih_bounds = None
            if self._optimizer == "L-BFGS-B":
                lower, upper = utility.get_parameter_bounds()
                if lower is not None:
                    ih_bounds = (jnp.asarray(lower), jnp.asarray(upper))

            result = minimize_lbfgsb(
                neg_ll_jax,
                jnp.asarray(initial_params, dtype=jnp.float64),
                bounds=ih_bounds,
                maxiter=self._outer_max_iter,
                tol=self._outer_tol,
                verbose=self._verbose,
                desc=f"NFXP {self._optimizer}",
                param_names=list(utility.parameter_names),
            )

            final_params = jnp.array(result.x, dtype=jnp.float32)
            ll = -result.fun
            n_iter = result.nit
            opt_converged = result.success

        # Compute final value function and policy
        flow_utility = jnp.array(utility.compute(final_params), dtype=jnp.float64)

        if finite_horizon:
            utility_seq = jnp.stack([flow_utility] * problem.num_periods)
            fh_result = backward_induction(operator, utility_seq)
            final_V = fh_result.V
            final_policy = fh_result.policy
            final_inner_iterations = problem.num_periods
        else:
            solver_result = self._solve_inner(operator, flow_utility)
            final_V = solver_result.V
            final_policy = solver_result.policy
            final_inner_iterations = solver_result.num_iterations
            total_inner += final_inner_iterations

        # Compute Hessian and gradient contributions for standard errors
        hessian = None
        gradient_contributions = None

        if self._compute_hessian and not finite_horizon:
            self._log("Computing standard errors via analytical score")
            scores, final_ll = self._compute_analytical_score(
                final_params, panel, utility, operator, final_V, final_policy,
            )
            gradient_contributions = scores
            hessian = -(scores.T @ scores)
            ll = final_ll

        elif self._compute_hessian and finite_horizon:
            self._log("Computing numerical Hessian for finite-horizon SEs")

            def ll_fn_fh(params):
                flow_u = jnp.array(utility.compute(params), dtype=jnp.float64)
                utility_seq = jnp.stack([flow_u] * problem.num_periods)
                fh_r = backward_induction(operator, utility_seq)
                sigma = problem.scale_parameter
                lp = jax.nn.log_softmax(fh_r.Q / sigma, axis=1)
                all_s = panel.get_all_states()
                all_a = panel.get_all_actions()
                return float(lp[all_s, all_a].sum())

            hessian = compute_numerical_hessian(final_params, ll_fn_fh)

        optimization_time = time.time() - start_time

        # Restore original outer_max_iter if it was overridden
        if outer_max_iter_override is not None:
            self._outer_max_iter = saved_outer_max_iter

        return EstimationResult(
            parameters=final_params,
            log_likelihood=ll,
            value_function=final_V,
            policy=final_policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=opt_converged,
            num_iterations=n_iter,
            num_function_evals=n_evals,
            num_inner_iterations=total_inner,
            message="Converged" if opt_converged else "Did not converge",
            optimization_time=optimization_time,
            metadata={
                "optimizer": self._optimizer,
                "inner_solver": self._inner_solver,
                "inner_tol": self._inner_tol,
                "switch_tol": self._switch_tol if self._inner_solver == "hybrid" else None,
                "outer_tol": self._outer_tol,
                "num_function_evals": n_evals,
                "num_inner_iterations": total_inner,
                "final_inner_iterations": final_inner_iterations,
            },
        )

    def compute_log_likelihood(
        self,
        params: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: jnp.ndarray,
    ) -> float:
        """Compute log-likelihood at given parameters."""
        transitions_f64 = jnp.array(transitions, dtype=jnp.float64)
        operator = SoftBellmanOperator(problem, transitions_f64)
        flow_utility = jnp.array(utility.compute(params), dtype=jnp.float64)

        solver_result = self._solve_inner(operator, flow_utility)

        log_probs = operator.compute_log_choice_probabilities(
            flow_utility, solver_result.V
        )

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        return float(log_probs[all_states, all_actions].sum())


def _compute_log_probs(utility, V, transitions, beta, sigma):
    """Compute log choice probabilities from utility and value function."""
    EV = jnp.einsum("ast,t->as", transitions, V)
    Q = utility + beta * EV.T
    return jax.nn.log_softmax(Q / sigma, axis=1)
