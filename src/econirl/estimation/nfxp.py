"""Nested Fixed Point (NFXP) estimator.

This module implements the NFXP algorithm from Rust (1987, 1988) for
estimating dynamic discrete choice models, with the SA→NK polyalgorithm
from Iskhakov, Jørgensen, Rust & Schjerning (2016).

Algorithm:
    Outer loop: Maximize log-likelihood via BHHH or L-BFGS-B
    Inner loop: Solve Bellman equation via SA→NK polyalgorithm
    Gradient: Analytical via implicit function theorem (no numerical diffs)

The log-likelihood is:
    ℓ(θ) = Σ_i Σ_t log P(a_{it} | s_{it}; θ)

where choice probabilities come from the logit model:
    P(a|s; θ) = exp(Q(s,a;θ)/σ) / Σ_{a'} exp(Q(s,a';θ)/σ)

References:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines"
    Iskhakov et al. (2016). "Comment on Constrained Optimization
        Approaches to Estimation of Structural Models." Econometrica.
    OpenSourceEconomics/ruspy — Python reference implementation
"""

from __future__ import annotations

import time
from typing import Literal

import numpy as np
import torch
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration, policy_iteration, hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


def estimate_transitions_from_panel(
    panel: Panel,
    num_states: int,
    max_increment: int = 2,
) -> torch.Tensor:
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
            if traj.actions[t].item() == 0:  # keep action
                inc = traj.states[t + 1].item() - traj.states[t].item()
                if inc >= 0:
                    inc = min(inc, max_increment)
                    counts[inc] += 1

    if counts.sum() == 0:
        probs = np.ones(max_increment + 1) / (max_increment + 1)
    else:
        probs = counts / counts.sum()

    # Build transition matrices
    n = num_states
    trans = torch.zeros(2, n, n, dtype=torch.float64)

    for a in range(2):
        for s in range(n):
            if a == 1:  # replace: transition from state 0
                src = 0
            else:
                src = s
            for k, p in enumerate(probs):
                dest = min(src + k, n - 1)
                trans[a, s, dest] += p
            # Absorbing: accumulate probability at last state
            if src + max_increment >= n:
                overflow = sum(probs[j] for j in range(max_increment + 1) if src + j >= n)
                trans[a, s, n - 1] += 0  # already handled by min(dest, n-1)

    return trans


class NFXPEstimator(BaseEstimator):
    """Nested Fixed Point estimator for dynamic discrete choice models.

    Implements the Iskhakov et al. (2016) SA→NK polyalgorithm with
    analytical gradients via implicit differentiation and BHHH optimization.

    For each candidate parameter vector θ:
    1. Compute flow utility matrix U(s,a; θ)
    2. Solve for value function V(s; θ) via SA→NK polyalgorithm
    3. Compute choice probabilities P(a|s; θ)
    4. Evaluate log-likelihood and analytical gradient

    Example:
        >>> estimator = NFXPEstimator(optimizer="BHHH", verbose=True)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        se_method: SEMethod = "asymptotic",
        optimizer: Literal["L-BFGS-B", "BFGS", "Newton-CG", "BHHH"] = "BHHH",
        inner_solver: Literal["value", "policy", "hybrid"] = "hybrid",
        inner_tol: float = 1e-12,
        inner_max_iter: int = 100000,
        switch_tol: float = 1e-3,
        outer_tol: float = 1e-6,
        outer_max_iter: int = 1000,
        compute_hessian: bool = True,
        analytical_gradient: bool = True,
        verbose: bool = False,
    ):
        """Initialize the NFXP estimator.

        Args:
            se_method: Method for computing standard errors
            optimizer: Optimizer for outer loop.
                - "BHHH": Berndt-Hall-Hall-Hausman (recommended, uses analytical
                          per-observation Jacobian for Hessian approximation)
                - "L-BFGS-B": Scipy L-BFGS-B with bounds
                - "BFGS": Scipy BFGS
            inner_solver: Solver for inner fixed-point problem.
                - "policy": Policy iteration (recommended for n < 1000)
                - "hybrid": SA→NK polyalgorithm per Iskhakov et al. (2016)
                - "value": Pure contraction (slow for high beta)
            inner_tol: Final convergence tolerance for inner solver
            inner_max_iter: Max iterations for inner solver
            switch_tol: SA→NK switch tolerance (hybrid solver only)
            outer_tol: Gradient tolerance for outer optimization
            outer_max_iter: Max outer optimization iterations
            compute_hessian: Whether to compute Hessian for standard errors
            analytical_gradient: Use analytical gradient via implicit
                differentiation (recommended). Falls back to numerical if False.
            verbose: Print progress messages
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        self._optimizer = optimizer
        self._inner_solver = inner_solver
        self._inner_tol = inner_tol
        self._inner_max_iter = inner_max_iter
        self._switch_tol = switch_tol
        self._outer_tol = outer_tol
        self._outer_max_iter = outer_max_iter
        self._analytical_gradient = analytical_gradient

    @property
    def name(self) -> str:
        return "NFXP (Nested Fixed Point)"

    def _solve_inner(
        self,
        operator: SoftBellmanOperator,
        flow_utility: torch.Tensor,
    ):
        """Solve the inner dynamic programming problem."""
        if self._inner_solver == "policy":
            return policy_iteration(
                operator,
                flow_utility,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
                eval_method="matrix",
            )
        elif self._inner_solver == "hybrid":
            return hybrid_iteration(
                operator,
                flow_utility,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
                switch_tol=self._switch_tol,
            )
        else:
            return value_iteration(
                operator,
                flow_utility,
                tol=self._inner_tol,
                max_iter=self._inner_max_iter,
            )

    def _compute_frechet_derivative(
        self,
        policy: torch.Tensor,
        operator: SoftBellmanOperator,
    ) -> torch.Tensor:
        """Compute Fréchet derivative of the Bellman operator: dT/dV.

        For the soft Bellman operator with logit shocks:
            dT/dV = β · P_π where P_π[s,s'] = Σ_a π(a|s) P(s'|s,a)

        Args:
            policy: Choice probabilities, shape (num_states, num_actions)
            operator: Bellman operator with transitions

        Returns:
            F = I - β·P_π, shape (num_states, num_states)
            This is the matrix used in NK steps and implicit differentiation.
        """
        beta = operator.problem.discount_factor
        n = operator.problem.num_states
        dtype = operator.transitions.dtype

        # P_π[s,s'] = Σ_a π(a|s) P(s'|s,a)
        P_pi = torch.einsum("sa,ast->st", policy.to(dtype), operator.transitions)

        # F = I - β·P_π
        F = torch.eye(n, dtype=dtype) - beta * P_pi
        return F

    def _compute_analytical_score(
        self,
        params: torch.Tensor,
        panel: Panel,
        utility: UtilityFunction,
        operator: SoftBellmanOperator,
        V: torch.Tensor,
        policy: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Compute analytical per-observation score via implicit differentiation.

        Uses the implicit function theorem to compute dV/dθ without
        differentiating through the fixed-point iteration:
            (I - β·P_π) · dV/dθ = Σ_a π(a|s) · dU(s,a)/dθ

        Then the per-observation log-likelihood gradient is:
            d log P(a_i|s_i)/dθ = (1/σ) · [dQ(s,a)/dθ - Σ_a' π(a'|s)·dQ(s,a')/dθ]

        References:
            Rust (2000) NFXP Manual, Section 3.2
            Iskhakov et al. (2016), implicit differentiation approach

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
        dtype = operator.transitions.dtype
        beta = operator.problem.discount_factor
        sigma = operator.problem.scale_parameter
        n_states = operator.problem.num_states
        n_actions = operator.problem.num_actions
        n_params = len(params)

        # Get feature matrix: φ(s,a,k) where U(s,a) = Σ_k θ_k · φ(s,a,k)
        features = utility.feature_matrix.to(dtype)  # (S, A, K)

        # Compute F = I - β·P_π (same matrix used in NK steps)
        F = self._compute_frechet_derivative(policy, operator)

        # For each parameter k, compute dT/dθ_k = Σ_a π(a|s) · φ(s,a,k)
        # This is the RHS of the implicit differentiation equation
        # dT/dθ_k[s] = Σ_a π(a|s) · φ(s,a,k)
        policy_64 = policy.to(dtype)
        dT_dtheta = torch.einsum("sa,sak->sk", policy_64, features)  # (S, K)

        # Solve F · dV/dθ = dT/dθ for dV/dθ
        # F is (S, S), dT_dtheta is (S, K), so we solve K systems at once
        dV_dtheta = torch.linalg.solve(F, dT_dtheta)  # (S, K)

        # Compute dQ(s,a)/dθ_k = φ(s,a,k) + β · Σ_{s'} P(s'|s,a) · dV(s')/dθ_k
        # EV_deriv[a,s,k] = Σ_{s'} P(s'|s,a) · dV(s')/dθ_k
        EV_deriv = torch.einsum("ast,tk->ask", operator.transitions, dV_dtheta)  # (A, S, K)
        dQ_dtheta = features + beta * EV_deriv.permute(1, 0, 2)  # (S, A, K)

        # Per-observation score: d log P(a_i|s_i)/dθ_k
        # = (1/σ) · [dQ(s_i,a_i)/dθ_k - Σ_{a'} π(a'|s_i) · dQ(s_i,a')/dθ_k]
        # = (1/σ) · [dQ(s_i,a_i)/dθ_k - E_π[dQ(s_i,·)/dθ_k]]
        E_dQ = torch.einsum("sa,sak->sk", policy_64, dQ_dtheta)  # (S, K)

        # Get observation data
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()

        # Index into dQ and E_dQ for each observation
        # dQ_obs[i,k] = dQ(s_i, a_i, k)
        dQ_obs = dQ_dtheta[all_states, all_actions]  # (N, K)
        E_dQ_obs = E_dQ[all_states]  # (N, K)

        scores = (1.0 / sigma) * (dQ_obs - E_dQ_obs)  # (N, K)

        # Also compute log-likelihood
        log_probs = operator.compute_log_choice_probabilities(
            utility.compute(params).to(dtype), V
        )
        ll = log_probs[all_states, all_actions].sum().item()

        return scores.float(), ll

    def _bhhh_optimize(
        self,
        initial_params: torch.Tensor,
        panel: Panel,
        utility: UtilityFunction,
        operator: SoftBellmanOperator,
        solver_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, float, int, int, int, bool]:
        """Run BHHH (Berndt-Hall-Hall-Hausman) optimization.

        BHHH uses the outer product of per-observation scores as the
        Hessian approximation: H ≈ Σ_i s_i s_i' (guaranteed PSD).

        Args:
            initial_params: Starting parameter values
            panel: Observed data
            utility: Utility specification
            operator: Bellman operator (with float64 transitions)
            solver_dtype: Dtype for solver computations

        Returns:
            Tuple of (params, ll, n_iter, n_evals, n_inner, converged)
        """
        params = initial_params.clone().float()
        n_params = len(params)
        total_inner = 0
        n_evals = 0
        converged = False

        prev_ll = -float("inf")

        for iteration in range(self._outer_max_iter):
            # Solve inner problem
            flow_utility = utility.compute(params).to(solver_dtype)
            solver_result = self._solve_inner(operator, flow_utility)
            total_inner += solver_result.num_iterations

            if not solver_result.converged:
                self._log(f"Warning: Inner loop did not converge (iter {iteration+1})")

            # Compute analytical score and log-likelihood
            scores, ll = self._compute_analytical_score(
                params, panel, utility, operator, solver_result.V, solver_result.policy
            )
            n_evals += 1

            # Gradient = sum of per-observation scores
            grad = scores.sum(dim=0)  # (K,)

            if self._verbose and (iteration + 1) % 5 == 0:
                self._log(f"BHHH iter {iteration+1}: LL = {ll:.4f}, |grad| = {grad.norm():.2e}")

            # Check convergence: gradient norm or LL change
            grad_norm = grad.abs().max().item()
            ll_change = abs(ll - prev_ll) if prev_ll > -float("inf") else float("inf")

            if grad_norm < self._outer_tol or (iteration > 10 and ll_change < 1e-10):
                converged = True
                self._log(f"BHHH converged at iter {iteration+1}: |grad| = {grad_norm:.2e}, ΔLL = {ll_change:.2e}")
                break

            prev_ll = ll

            # BHHH Hessian approximation: H = S'S (outer product of scores)
            H_bhhh = scores.T @ scores  # (K, K)

            # Add small ridge for numerical stability
            H_bhhh += 1e-8 * torch.eye(n_params)

            # Newton direction: Δθ = H^{-1} · g
            try:
                direction = torch.linalg.solve(H_bhhh, grad)
            except torch.linalg.LinAlgError:
                direction = grad  # Fallback to gradient ascent

            # Step-halving line search
            step_size = 1.0
            new_ll = ll
            for _ in range(15):
                new_params = params + step_size * direction
                flow_u_new = utility.compute(new_params).to(solver_dtype)
                solver_new = self._solve_inner(operator, flow_u_new)
                total_inner += solver_new.num_iterations
                n_evals += 1

                log_probs = operator.compute_log_choice_probabilities(
                    flow_u_new, solver_new.V
                )
                all_s = panel.get_all_states()
                all_a = panel.get_all_actions()
                new_ll = log_probs[all_s, all_a].sum().item()

                if new_ll > ll:
                    break
                step_size *= 0.5
            else:
                # Line search failed — accept small step anyway
                new_params = params + step_size * direction

            params = new_params

        return params, ll, iteration + 1, n_evals, total_inner, converged

    def _estimate_initial_params(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> torch.Tensor:
        """Estimate rough starting values from data."""
        n_params = utility.num_parameters
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        total_obs = all_states.shape[0]
        replace_mask = all_actions == 1
        n_replace = replace_mask.sum().item()
        mileage_at_replace = all_states[replace_mask].float().sum().item()

        if n_replace > 0 and total_obs > 0:
            replace_rate = n_replace / total_obs
            avg_mileage = mileage_at_replace / n_replace

            if n_params == 2:
                n_states = problem.num_states
                op_cost_init = 1.0 / n_states
                rc_init = max(0.5, op_cost_init * avg_mileage / max(replace_rate, 0.01))
                return torch.tensor([op_cost_init, rc_init], dtype=torch.float32)

        return torch.full((n_params,), 0.01, dtype=torch.float32)

    def _optimize(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
        initial_params: torch.Tensor | None = None,
        **kwargs,
    ) -> EstimationResult:
        """Run NFXP optimization.

        Args:
            panel: Panel data with observed choices
            utility: Utility function specification
            problem: Problem specification
            transitions: Transition probability matrices
            initial_params: Starting values (defaults to data-driven)

        Returns:
            EstimationResult with optimized parameters
        """
        import warnings

        start_time = time.time()

        beta = problem.discount_factor
        if beta > 0.99 and self._inner_max_iter < 50000 and self._inner_solver == "value":
            warnings.warn(
                f"High discount factor beta={beta} may require inner_max_iter > 50000. "
                f"Current: {self._inner_max_iter}. Consider increasing for convergence.",
                UserWarning,
            )

        # Initialize parameters
        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            if (initial_params == 0).all():
                initial_params = self._estimate_initial_params(
                    panel, utility, problem
                )

        # Use float64 for high discount factors (condition number ≈ 1/(1-β))
        use_float64 = beta > 0.99
        solver_dtype = torch.float64 if use_float64 else torch.float32
        transitions_solver = transitions.to(solver_dtype)
        operator = SoftBellmanOperator(problem, transitions_solver)

        # Run optimization
        if self._optimizer == "BHHH" and self._analytical_gradient:
            # BHHH with analytical gradient (Iskhakov et al. recommended approach)
            self._log("Starting BHHH optimization with analytical gradient")

            params, ll, n_iter, n_evals, n_inner, opt_converged = self._bhhh_optimize(
                initial_params, panel, utility, operator, solver_dtype
            )
            final_params = params

        else:
            # Scipy optimizer (L-BFGS-B, BFGS, etc.)
            total_inner_iterations = 0
            num_function_evals = 0

            def objective(params_np):
                nonlocal total_inner_iterations, num_function_evals
                num_function_evals += 1

                params = torch.tensor(params_np, dtype=torch.float32)
                flow_utility = utility.compute(params).to(solver_dtype)
                solver_result = self._solve_inner(operator, flow_utility)
                total_inner_iterations += solver_result.num_iterations

                if not solver_result.converged:
                    self._log("Warning: Inner loop did not converge")

                log_probs = operator.compute_log_choice_probabilities(
                    flow_utility, solver_result.V
                )
                all_states = panel.get_all_states()
                all_actions = panel.get_all_actions()
                ll = log_probs[all_states, all_actions].sum().item()

                if self._verbose and num_function_evals % 10 == 0:
                    self._log(f"Eval {num_function_evals}: LL = {ll:.4f}")

                return -ll

            if self._analytical_gradient:
                # Analytical gradient for scipy optimizers
                def gradient(params_np):
                    params = torch.tensor(params_np, dtype=torch.float32)
                    flow_utility = utility.compute(params).to(solver_dtype)
                    solver_result = self._solve_inner(operator, flow_utility)
                    scores, _ = self._compute_analytical_score(
                        params, panel, utility, operator,
                        solver_result.V, solver_result.policy,
                    )
                    return -scores.sum(dim=0).numpy()  # negative for minimization
            else:
                # Numerical gradient fallback
                def gradient(params_np):
                    grad = np.zeros(len(params_np))
                    for i in range(len(params_np)):
                        eps_i = max(1e-5, abs(params_np[i]) * 1e-4)
                        p_plus = params_np.copy()
                        p_minus = params_np.copy()
                        p_plus[i] += eps_i
                        p_minus[i] -= eps_i
                        grad[i] = (objective(p_plus) - objective(p_minus)) / (2 * eps_i)
                    return grad

            self._log(f"Starting optimization with {self._optimizer}")

            bounds = None
            if self._optimizer == "L-BFGS-B":
                lower, upper = utility.get_parameter_bounds()
                bounds = list(zip(lower.numpy(), upper.numpy()))

            result = optimize.minimize(
                objective,
                initial_params.numpy(),
                method=self._optimizer if self._optimizer != "BHHH" else "L-BFGS-B",
                jac=gradient if self._optimizer in ["L-BFGS-B", "BFGS", "BHHH"] else None,
                bounds=bounds,
                options={
                    "maxiter": self._outer_max_iter,
                    "gtol": self._outer_tol,
                },
            )

            final_params = torch.tensor(result.x, dtype=torch.float32)
            ll = -result.fun
            n_iter = result.nit
            n_evals = num_function_evals
            n_inner = total_inner_iterations
            opt_converged = result.success

        # Compute final value function and policy
        flow_utility = utility.compute(final_params).to(solver_dtype)
        solver_result = self._solve_inner(operator, flow_utility)

        # Compute Hessian and gradient contributions for standard errors
        hessian = None
        gradient_contributions = None

        if self._compute_hessian:
            self._log("Computing standard errors via analytical score")

            # Per-observation scores (for robust SEs and BHHH Hessian)
            scores, final_ll = self._compute_analytical_score(
                final_params, panel, utility, operator,
                solver_result.V, solver_result.policy,
            )
            gradient_contributions = scores

            # Hessian from outer product of scores (BHHH approximation)
            # or numerical Hessian for exact asymptotic SEs
            hessian = -(scores.T @ scores)  # Negative because LL Hessian
            # Add the expected Hessian correction for proper SEs
            # For BHHH, this is the standard approximation
            ll = final_ll

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=final_params,
            log_likelihood=ll,
            value_function=solver_result.V,
            policy=solver_result.policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=opt_converged,
            num_iterations=n_iter,
            num_function_evals=n_evals,
            num_inner_iterations=n_inner,
            message="BHHH converged" if opt_converged else "Did not converge",
            optimization_time=optimization_time,
            metadata={
                "optimizer": self._optimizer,
                "inner_solver": self._inner_solver,
                "analytical_gradient": self._analytical_gradient,
                "inner_tol": self._inner_tol,
                "switch_tol": self._switch_tol if self._inner_solver == "hybrid" else None,
                "outer_tol": self._outer_tol,
            },
        )

    def _compute_gradient_contributions(
        self,
        params: torch.Tensor,
        panel: Panel,
        utility: UtilityFunction,
        operator: SoftBellmanOperator,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Compute per-observation gradient contributions.

        Uses analytical score if available, falls back to numerical.
        """
        solver_dtype = operator.transitions.dtype
        flow_utility = utility.compute(params).to(solver_dtype)
        solver_result = self._solve_inner(operator, flow_utility)

        if self._analytical_gradient and hasattr(utility, 'feature_matrix'):
            scores, _ = self._compute_analytical_score(
                params, panel, utility, operator,
                solver_result.V, solver_result.policy,
            )
            return scores

        # Numerical fallback
        n_obs = panel.num_observations
        n_params = len(params)
        gradients = torch.zeros((n_obs, n_params))

        log_probs_base = operator.compute_log_choice_probabilities(
            flow_utility, solver_result.V
        )

        for k in range(n_params):
            eps_k = max(eps, abs(params[k].item()) * 1e-4)

            params_plus = params.clone()
            params_plus[k] += eps_k
            flow_plus = utility.compute(params_plus).to(solver_dtype)
            sol_plus = self._solve_inner(operator, flow_plus)
            lp_plus = operator.compute_log_choice_probabilities(flow_plus, sol_plus.V)

            params_minus = params.clone()
            params_minus[k] -= eps_k
            flow_minus = utility.compute(params_minus).to(solver_dtype)
            sol_minus = self._solve_inner(operator, flow_minus)
            lp_minus = operator.compute_log_choice_probabilities(flow_minus, sol_minus.V)

            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            gradients[:, k] = (
                lp_plus[all_states, all_actions] - lp_minus[all_states, all_actions]
            ) / (2 * eps_k)

        return gradients

    def compute_log_likelihood(
        self,
        params: torch.Tensor,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
        transitions: torch.Tensor,
    ) -> float:
        """Compute log-likelihood at given parameters."""
        beta = problem.discount_factor
        solver_dtype = torch.float64 if beta > 0.99 else torch.float32
        operator = SoftBellmanOperator(problem, transitions.to(solver_dtype))
        flow_utility = utility.compute(params).to(solver_dtype)

        solver_result = self._solve_inner(operator, flow_utility)

        log_probs = operator.compute_log_choice_probabilities(
            flow_utility, solver_result.V
        )

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        return log_probs[all_states, all_actions].sum().item()
