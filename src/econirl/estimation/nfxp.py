"""Nested Fixed Point (NFXP) estimator.

This module implements the NFXP algorithm from Rust (1987, 1988) for
estimating dynamic discrete choice models. NFXP is the classic approach
that nests the solution of the dynamic programming problem within the
likelihood maximization.

Algorithm:
    Outer loop: Maximize log-likelihood over θ using L-BFGS
    Inner loop: Solve Bellman equation for V(s; θ) via value iteration

The log-likelihood is:
    ℓ(θ) = Σ_i Σ_t log P(a_{it} | s_{it}; θ)

where choice probabilities come from the logit model:
    P(a|s; θ) = exp(Q(s,a;θ)/σ) / Σ_{a'} exp(Q(s,a';θ)/σ)

Reference:
    Rust, J. (1987). "Optimal Replacement of GMC Bus Engines"
    Rust, J. (1988). "Maximum Likelihood Estimation of Discrete Control
                      Processes" (NFXP algorithm details)
"""

from __future__ import annotations

import time
from typing import Literal

import torch
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration, policy_iteration, hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction


class NFXPEstimator(BaseEstimator):
    """Nested Fixed Point estimator for dynamic discrete choice models.

    NFXP maximizes the conditional choice probability likelihood by
    nesting the solution of the Bellman equation within optimization.

    For each candidate parameter vector θ:
    1. Compute flow utility matrix U(s,a; θ)
    2. Solve for value function V(s; θ) via value iteration
    3. Compute choice probabilities P(a|s; θ)
    4. Evaluate log-likelihood

    The optimizer (L-BFGS-B by default) iterates until convergence.

    Attributes:
        se_method: Standard error computation method
        optimizer: Scipy optimizer to use
        inner_tol: Convergence tolerance for value iteration
        outer_tol: Convergence tolerance for outer optimization

    Example:
        >>> estimator = NFXPEstimator(se_method="robust", verbose=True)
        >>> result = estimator.estimate(panel, utility, problem, transitions)
        >>> print(result.summary())
    """

    def __init__(
        self,
        se_method: SEMethod = "asymptotic",
        optimizer: Literal["L-BFGS-B", "BFGS", "Newton-CG"] = "L-BFGS-B",
        inner_solver: Literal["value", "policy", "hybrid"] = "hybrid",
        inner_tol: float = 1e-10,
        inner_max_iter: int = 100000,  # Sufficient for beta=0.9999
        switch_tol: float = 1e-3,
        outer_tol: float = 1e-6,
        outer_max_iter: int = 1000,
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the NFXP estimator.

        Args:
            se_method: Method for computing standard errors
            optimizer: Scipy optimizer for outer loop
            inner_solver: Solver for inner loop.
                - "value": Pure contraction (value iteration), linear convergence
                - "policy": Policy iteration with matrix solve, faster for high beta
                - "hybrid": Hybrid contraction + Newton-Kantorovich (recommended).
                           Starts with cheap contractions, switches to NK near
                           solution for quadratic convergence. Per Rust (2000),
                           typically 10-100x faster than pure contraction.
            inner_tol: Tolerance for inner loop convergence
            inner_max_iter: Max iterations for inner loop. Default 100000 is
                           sufficient for beta=0.9999. Value iteration
                           convergence is O(1/(1-beta)), so high discount
                           factors require many more iterations.
            switch_tol: For hybrid solver, switch from contraction to NK when
                       error < this value. Default 1e-3 is usually optimal.
            outer_tol: Tolerance for outer optimization convergence
            outer_max_iter: Max iterations for outer optimization
            compute_hessian: Whether to compute Hessian for inference
            verbose: Whether to print progress messages
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

    def _estimate_initial_params(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> torch.Tensor:
        """Estimate rough starting values from data.

        Uses the empirical replacement rate and average mileage at replacement
        to get a non-degenerate starting point for optimization.
        """
        n_params = utility.num_parameters

        # Compute empirical replacement rate and average mileage
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        total_obs = all_states.shape[0]
        replace_mask = all_actions == 1
        n_replace = replace_mask.sum().item()
        mileage_at_replace = all_states[replace_mask].float().sum().item()

        if n_replace > 0 and total_obs > 0:
            replace_rate = n_replace / total_obs
            avg_mileage = mileage_at_replace / n_replace

            # For Rust-style models with 2 parameters [operating_cost, RC]:
            # At the replacement threshold, roughly:
            #   RC ≈ operating_cost * avg_mileage / replace_rate
            # Use heuristic: set operating_cost so that at avg_mileage,
            # the cost is a moderate fraction of state space
            if n_params == 2:
                n_states = problem.num_states
                # operating_cost ~ 1/n_states (so cost at max state is ~1)
                op_cost_init = 1.0 / n_states
                # RC ~ operating_cost * avg_mileage * (1 - gamma) / replace_rate
                rc_init = max(0.5, op_cost_init * avg_mileage / max(replace_rate, 0.01))
                return torch.tensor([op_cost_init, rc_init], dtype=torch.float32)

        # Generic fallback: small positive values
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
            initial_params: Starting values (defaults to zeros)

        Returns:
            EstimationResult with optimized parameters
        """
        import warnings

        start_time = time.time()

        # Warn if discount is high but inner_max_iter may be insufficient
        # (only relevant for value iteration; policy iteration converges in ~10 iters)
        beta = problem.discount_factor
        if beta > 0.99 and self._inner_max_iter < 50000 and self._inner_solver == "value":
            warnings.warn(
                f"High discount factor beta={beta} may require inner_max_iter > 50000. "
                f"Current: {self._inner_max_iter}. Consider increasing for convergence.",
                UserWarning,
            )

        # Initialize — use data-driven starting values if defaults are zeros
        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            # If all zeros, estimate rough starting values from data
            if (initial_params == 0).all():
                initial_params = self._estimate_initial_params(
                    panel, utility, problem
                )

        # Create Bellman operator
        # Use float64 for high discount factors to avoid numerical issues
        # (condition number of (I - β·P) ≈ 1/(1-β), e.g. 10000 at β=0.9999)
        use_float64 = beta > 0.99
        solver_dtype = torch.float64 if use_float64 else torch.float32
        transitions_solver = transitions.to(solver_dtype)
        operator = SoftBellmanOperator(problem, transitions_solver)

        # Tracking variables
        total_inner_iterations = 0
        num_function_evals = 0

        # Define objective function (negative log-likelihood for minimization)
        def objective(params_np):
            nonlocal total_inner_iterations, num_function_evals
            num_function_evals += 1

            params = torch.tensor(params_np, dtype=torch.float32)

            # Compute flow utility (upcast for solver precision)
            flow_utility = utility.compute(params).to(solver_dtype)

            # Solve for value function (inner loop)
            solver_result = self._solve_inner(operator, flow_utility)
            total_inner_iterations += solver_result.num_iterations

            if not solver_result.converged:
                self._log(f"Warning: Inner loop did not converge")

            # Compute log-likelihood
            log_probs = operator.compute_log_choice_probabilities(
                flow_utility, solver_result.V
            )

            # Sum over observations (vectorized)
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            ll = log_probs[all_states, all_actions].sum().item()

            if self._verbose and num_function_evals % 10 == 0:
                self._log(f"Eval {num_function_evals}: LL = {ll:.4f}")

            # Return negative for minimization
            return -ll

        # Define gradient (numerical, with adaptive step size)
        def gradient(params_np):
            grad = torch.zeros(len(params_np))
            for i in range(len(params_np)):
                # Adaptive epsilon: scale to parameter magnitude
                eps_i = max(1e-5, abs(params_np[i]) * 1e-4)
                params_plus = params_np.copy()
                params_minus = params_np.copy()
                params_plus[i] += eps_i
                params_minus[i] -= eps_i
                grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps_i)
            return grad.numpy()

        # Run optimization
        self._log(f"Starting optimization with {self._optimizer}")

        bounds = None
        if self._optimizer == "L-BFGS-B":
            lower, upper = utility.get_parameter_bounds()
            bounds = list(zip(lower.numpy(), upper.numpy()))

        result = optimize.minimize(
            objective,
            initial_params.numpy(),
            method=self._optimizer,
            jac=gradient if self._optimizer in ["L-BFGS-B", "BFGS"] else None,
            bounds=bounds,
            options={
                "maxiter": self._outer_max_iter,
                "gtol": self._outer_tol,
                "disp": self._verbose,
            },
        )

        # Extract final parameters
        final_params = torch.tensor(result.x, dtype=torch.float32)

        # Compute final value function and policy
        flow_utility = utility.compute(final_params).to(solver_dtype)
        solver_result = self._solve_inner(operator, flow_utility)

        # Compute Hessian for standard errors
        hessian = None
        gradient_contributions = None

        if self._compute_hessian:
            self._log("Computing Hessian for standard errors")

            def ll_fn(params):
                return torch.tensor(-objective(params.numpy()))

            hessian = compute_numerical_hessian(final_params, ll_fn)

            # Compute per-observation gradients for robust SEs
            gradient_contributions = self._compute_gradient_contributions(
                final_params, panel, utility, operator
            )

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=final_params,
            log_likelihood=-result.fun,
            value_function=solver_result.V,
            policy=solver_result.policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=result.success,
            num_iterations=result.nit,
            num_function_evals=num_function_evals,
            num_inner_iterations=total_inner_iterations,
            message=result.message,
            optimization_time=optimization_time,
            metadata={
                "optimizer": self._optimizer,
                "inner_solver": self._inner_solver,
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

        These are needed for robust and clustered standard errors.
        Uses numerical differentiation.
        """
        n_obs = panel.num_observations
        n_params = len(params)

        gradients = torch.zeros((n_obs, n_params))

        # Pre-compute log probabilities at current params
        # Use operator's dtype (float64 for high-beta problems)
        solver_dtype = operator.transitions.dtype
        flow_utility = utility.compute(params).to(solver_dtype)
        solver_result = self._solve_inner(operator, flow_utility)
        log_probs_base = operator.compute_log_choice_probabilities(
            flow_utility, solver_result.V
        )

        # Compute gradient for each parameter (adaptive epsilon)
        for k in range(n_params):
            eps_k = max(eps, abs(params[k].item()) * 1e-4)

            params_plus = params.clone()
            params_plus[k] += eps_k

            flow_utility_plus = utility.compute(params_plus).to(solver_dtype)
            solver_plus = self._solve_inner(operator, flow_utility_plus)
            log_probs_plus = operator.compute_log_choice_probabilities(
                flow_utility_plus, solver_plus.V
            )

            params_minus = params.clone()
            params_minus[k] -= eps_k

            flow_utility_minus = utility.compute(params_minus).to(solver_dtype)
            solver_minus = self._solve_inner(operator, flow_utility_minus)
            log_probs_minus = operator.compute_log_choice_probabilities(
                flow_utility_minus, solver_minus.V
            )

            # Compute gradients for all observations (vectorized)
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            gradients[:, k] = (
                log_probs_plus[all_states, all_actions]
                - log_probs_minus[all_states, all_actions]
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
        """Compute log-likelihood at given parameters.

        Useful for likelihood ratio tests and model comparison.

        Args:
            params: Parameter vector
            panel: Observed data
            utility: Utility specification
            problem: Problem specification
            transitions: Transition matrices

        Returns:
            Log-likelihood value
        """
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
        ll = log_probs[all_states, all_actions].sum().item()

        return ll
