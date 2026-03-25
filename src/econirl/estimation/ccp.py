"""CCP-based estimators: Hotz-Miller and NPL.

This module implements Conditional Choice Probability (CCP) estimators
for dynamic discrete choice models:

1. Hotz-Miller (1993): Two-step estimator using CCPs from data
2. NPL (Aguirregabiria-Mira 2002): Iterated Hotz-Miller that converges to MLE

Key insight: The value function can be recovered from CCPs without solving
the full Bellman equation, via the Hotz-Miller inversion theorem.

For logit errors:
    e(a,x) = γ - log(P(a|x))  where γ ≈ 0.5772 is Euler's constant

References:
    Hotz, V.J. and Miller, R.A. (1993). "Conditional Choice Probabilities
        and the Estimation of Dynamic Models." RES 60(3), 497-529.
    Aguirregabiria, V. and Mira, P. (2002). "Swapping the Nested Fixed Point
        Algorithm." Econometrica 70(4), 1519-1543.
    Aguirregabiria, V. and Mira, P. (2010). "Dynamic discrete choice structural
        models: A survey." Journal of Econometrics 156(1), 38-67.
"""

from __future__ import annotations

import math
import time
from typing import Literal

import torch
from scipy import optimize

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator, EstimationResult
from econirl.inference.standard_errors import SEMethod, compute_numerical_hessian
from econirl.preferences.base import UtilityFunction

# Euler-Mascheroni constant
EULER_GAMMA = 0.5772156649015329


class CCPEstimator(BaseEstimator):
    """CCP-based estimator for dynamic discrete choice models.

    Implements both Hotz-Miller (K=1) and NPL (K>1) estimation via the
    `num_policy_iterations` parameter.

    The algorithm:
    1. Estimate CCPs from data using frequency estimator
    2. For k = 1, ..., K:
       a) Compute emax correction: e(a,x) = γ - log(P(a|x))
       b) Compute valuation matrix via matrix inversion (eq 42, A&M 2010)
       c) Maximize pseudo-likelihood to get θ̂_k
       d) Update CCPs from θ̂_k (for NPL)
    3. Return final estimates

    Attributes:
        num_policy_iterations: Number of NPL iterations (K=1 is Hotz-Miller)
        ccp_min_count: Minimum observations per state for CCP estimation
        convergence_tol: Tolerance for NPL convergence check

    Example:
        >>> # Hotz-Miller (fast, one-step)
        >>> hm = CCPEstimator(num_policy_iterations=1)
        >>> result = hm.estimate(panel, utility, problem, transitions)
        >>>
        >>> # NPL (iterates to MLE)
        >>> npl = CCPEstimator(num_policy_iterations=10)
        >>> result = npl.estimate(panel, utility, problem, transitions)
    """

    def __init__(
        self,
        num_policy_iterations: int = 1,
        ccp_min_count: int = 1,
        ccp_smoothing: float = 1e-6,
        convergence_tol: float = 1e-6,
        outer_tol: float = 1e-6,
        outer_max_iter: int = 1000,
        se_method: SEMethod = "asymptotic",
        compute_hessian: bool = True,
        verbose: bool = False,
    ):
        """Initialize the CCP estimator.

        Args:
            num_policy_iterations: Number of policy iterations (K=1 is Hotz-Miller,
                                   K>1 is NPL). Set to -1 for convergence-based stopping.
            ccp_min_count: Minimum observations per state for reliable CCP estimation.
                          States with fewer observations get uniform CCPs.
            ccp_smoothing: Small value added to CCPs to avoid log(0).
            convergence_tol: Tolerance for NPL convergence (parameter change).
            outer_tol: Tolerance for pseudo-likelihood maximization.
            outer_max_iter: Max iterations for pseudo-likelihood maximization.
            se_method: Method for computing standard errors.
            compute_hessian: Whether to compute Hessian for inference.
            verbose: Whether to print progress messages.
        """
        super().__init__(
            se_method=se_method,
            compute_hessian=compute_hessian,
            verbose=verbose,
        )
        self._num_policy_iterations = num_policy_iterations
        self._ccp_min_count = ccp_min_count
        self._ccp_smoothing = ccp_smoothing
        self._convergence_tol = convergence_tol
        self._outer_tol = outer_tol
        self._outer_max_iter = outer_max_iter

    @property
    def name(self) -> str:
        if self._num_policy_iterations == 1:
            return "Hotz-Miller (CCP)"
        elif self._num_policy_iterations == -1:
            return "NPL (until convergence)"
        else:
            return f"NPL (K={self._num_policy_iterations})"

    def _estimate_ccps_from_data(
        self,
        panel: Panel,
        num_states: int,
        num_actions: int,
    ) -> torch.Tensor:
        """Estimate CCPs from data using frequency estimator.

        P̂(a|s) = N(s,a) / N(s)

        Args:
            panel: Panel data with observed choices
            num_states: Number of states
            num_actions: Number of actions

        Returns:
            CCP matrix of shape (num_states, num_actions)
        """
        # Count state-action frequencies
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        idx = all_states * num_actions + all_actions
        counts = torch.zeros(num_states * num_actions, dtype=torch.float32).scatter_add_(
            0, idx.long(), torch.ones(idx.shape[0])
        ).reshape(num_states, num_actions)

        # Get state counts
        state_counts = counts.sum(dim=1)

        # Compute CCPs with smoothing
        ccps = torch.zeros((num_states, num_actions), dtype=torch.float32)
        for s in range(num_states):
            if state_counts[s] >= self._ccp_min_count:
                # Use empirical frequencies with smoothing
                ccps[s] = (counts[s] + self._ccp_smoothing) / (
                    state_counts[s] + num_actions * self._ccp_smoothing
                )
            else:
                # Not enough data: use uniform distribution
                ccps[s] = 1.0 / num_actions

        return ccps

    def _compute_emax_correction(self, ccps: torch.Tensor) -> torch.Tensor:
        """Compute emax correction for logit errors.

        e(a,x) = γ - log(P(a|x))

        where γ ≈ 0.5772 is Euler's constant.

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)

        Returns:
            Emax correction matrix of shape (num_states, num_actions)
        """
        # Clamp CCPs to avoid log(0)
        ccps_safe = torch.clamp(ccps, min=self._ccp_smoothing)
        return EULER_GAMMA - torch.log(ccps_safe)

    def _compute_policy_weighted_transitions(
        self,
        ccps: torch.Tensor,
        transitions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute policy-weighted transition matrix F_π.

        F_π[s, s'] = Σ_a P(a|s) * P(s'|s,a)

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)
            transitions: Transition matrices of shape (num_actions, num_states, num_states)

        Returns:
            Policy-weighted transition matrix of shape (num_states, num_states)
        """
        num_states = ccps.shape[0]
        num_actions = ccps.shape[1]

        dtype = transitions.dtype
        F_pi = torch.zeros((num_states, num_states), dtype=dtype)
        for a in range(num_actions):
            # transitions[a] has shape (num_states, num_states)
            # ccps[:, a] has shape (num_states,)
            # We want F_pi[s, s'] += P(a|s) * P(s'|s,a)
            F_pi += ccps[:, a:a+1].to(dtype) * transitions[a]

        return F_pi

    def _compute_valuation_matrix(
        self,
        ccps: torch.Tensor,
        transitions: torch.Tensor,
        utility: UtilityFunction,
        parameters: torch.Tensor,
        problem: DDCProblem,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute valuation matrix W^P via matrix inversion.

        W^P = (I - β·F_π)⁻¹ · Σ_a P(a) ⊙ [u(a), e(a)]

        Following equation 42 in Aguirregabiria & Mira (2010).

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)
            transitions: Transition matrices
            utility: Utility function
            parameters: Current parameter estimate
            problem: Problem specification

        Returns:
            Tuple of (W_z, W_e) where:
            - W_z: shape (num_states, num_features) for utility contribution
            - W_e: shape (num_states,) for emax contribution
        """
        beta = problem.discount_factor
        num_states = problem.num_states
        num_actions = problem.num_actions

        # Compute policy-weighted transition matrix
        F_pi = self._compute_policy_weighted_transitions(ccps, transitions)

        # Compute (I - β·F_π)⁻¹
        dtype = F_pi.dtype
        I = torch.eye(num_states, dtype=dtype)
        inv_matrix = torch.linalg.inv(I - beta * F_pi)

        # Compute emax corrections
        e = self._compute_emax_correction(ccps)

        # Compute Σ_a P(a) ⊙ e(a) for each state
        # This gives expected emax correction under current policy
        expected_e = (ccps * e).sum(dim=1).to(inv_matrix.dtype)  # shape (num_states,)

        # W_e = (I - β·F_π)⁻¹ · expected_e
        W_e = inv_matrix @ expected_e

        # For utility contribution, we need the feature matrix
        # u(s,a) = θ · φ(s,a), so we compute expected features
        if hasattr(utility, 'feature_matrix'):
            features = utility.feature_matrix  # shape (num_states, num_actions, num_features)
            num_features = features.shape[2]

            # Compute Σ_a P(a|s) · φ(s,a) for each state
            # expected_features[s, k] = Σ_a P(a|s) · φ(s,a,k)
            dtype = inv_matrix.dtype
            expected_features = torch.einsum('sa,sak->sk', ccps.to(dtype), features.to(dtype))

            # W_z = (I - β·F_π)⁻¹ · expected_features
            W_z = inv_matrix @ expected_features
        else:
            # Fallback: compute utility directly
            flow_utility = utility.compute(parameters)  # shape (num_states, num_actions)
            expected_utility = (ccps * flow_utility).sum(dim=1)  # shape (num_states,)
            W_z = inv_matrix @ expected_utility
            W_z = W_z.unsqueeze(1)  # shape (num_states, 1)

        return W_z, W_e

    def _compute_choice_specific_values(
        self,
        ccps: torch.Tensor,
        transitions: torch.Tensor,
        utility: UtilityFunction,
        parameters: torch.Tensor,
        problem: DDCProblem,
    ) -> torch.Tensor:
        """Compute choice-specific value functions using CCP representation.

        For linear utility u(a,x) = z(a,x)'θ, the Hotz-Miller representation is:
            v(a,x) = z̃(a,x)'θ + ẽ(a,x)

        where:
            z̃(a,x) = z(a,x) + β·E[W_z(x') | x, a]
            ẽ(a,x) = β·E[W_e(x') | x, a]

        This correctly separates linear (in θ) and constant terms.

        Args:
            ccps: CCP matrix
            transitions: Transition matrices
            utility: Utility function
            parameters: Current parameters
            problem: Problem specification

        Returns:
            Choice-specific values of shape (num_states, num_actions)
        """
        beta = problem.discount_factor
        num_states = problem.num_states
        num_actions = problem.num_actions

        # Compute valuation matrix components
        W_z, W_e = self._compute_valuation_matrix(
            ccps, transitions, utility, parameters, problem
        )

        if hasattr(utility, 'feature_matrix'):
            features = utility.feature_matrix  # shape (num_states, num_actions, num_features)
            num_features = features.shape[2]

            # Compute E[W_z(x') | x, a] for each (x, a)
            # transitions[a, s, s'] = P(s'|s,a), W_z has shape (num_states, num_features)
            E_W_z = torch.zeros((num_states, num_actions, num_features), dtype=transitions.dtype)
            for a in range(num_actions):
                E_W_z[:, a, :] = transitions[a] @ W_z  # (num_states, num_features)

            # Compute E[W_e(x') | x, a]
            E_W_e = torch.zeros((num_states, num_actions), dtype=transitions.dtype)
            for a in range(num_actions):
                E_W_e[:, a] = transitions[a] @ W_e

            # z̃(a,x) = z(a,x) + β·E[W_z(x') | x, a]
            z_tilde = features.to(E_W_z.dtype) + beta * E_W_z  # (num_states, num_actions, num_features)

            # ẽ(a,x) = β·E[W_e(x') | x, a]
            e_tilde = beta * E_W_e  # (num_states, num_actions)

            # v(a,x) = z̃(a,x)'θ + ẽ(a,x)
            v = torch.einsum('sak,k->sa', z_tilde, parameters.to(z_tilde.dtype)) + e_tilde
        else:
            # Fallback for non-linear utility
            flow_utility = utility.compute(parameters)
            W = W_z.squeeze(1) + W_e

            EW = torch.zeros((num_states, num_actions), dtype=transitions.dtype)
            for a in range(num_actions):
                EW[:, a] = transitions[a] @ W

            v = flow_utility + beta * EW

        return v

    def _compute_log_likelihood(
        self,
        parameters: torch.Tensor,
        panel: Panel,
        utility: UtilityFunction,
        ccps: torch.Tensor,
        transitions: torch.Tensor,
        problem: DDCProblem,
    ) -> float:
        """Compute pseudo-log-likelihood given CCPs.

        Args:
            parameters: Current parameter estimate
            panel: Panel data
            utility: Utility function
            ccps: Current CCP estimates
            transitions: Transition matrices
            problem: Problem specification

        Returns:
            Log-likelihood value
        """
        sigma = problem.scale_parameter

        # Compute choice-specific values
        v = self._compute_choice_specific_values(
            ccps, transitions, utility, parameters, problem
        )

        # Compute log choice probabilities via softmax
        log_probs = torch.nn.functional.log_softmax(v / sigma, dim=1)

        # Sum log-likelihood over observations
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        ll = log_probs[all_states, all_actions].sum().item()

        return ll

    def _update_ccps_from_values(
        self,
        v: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Update CCPs from choice-specific values.

        P(a|x) = exp(v(a,x)/σ) / Σ_{a'} exp(v(a',x)/σ)

        Args:
            v: Choice-specific values of shape (num_states, num_actions)
            sigma: Scale parameter

        Returns:
            Updated CCPs of shape (num_states, num_actions)
        """
        return torch.nn.functional.softmax(v / sigma, dim=1)

    def _estimate_initial_params(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> torch.Tensor:
        """Estimate rough starting values from data."""
        n_params = utility.num_parameters
        total_obs = 0
        n_replace = 0
        mileage_at_replace = 0.0

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        total_obs = all_states.shape[0]
        replace_mask = all_actions == 1
        n_replace = replace_mask.sum().item()
        mileage_at_replace = all_states[replace_mask].float().sum().item()

        if n_replace > 0 and total_obs > 0 and n_params == 2:
            replace_rate = n_replace / total_obs
            avg_mileage = mileage_at_replace / n_replace
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
        """Run CCP/NPL optimization.

        Args:
            panel: Panel data with observed choices
            utility: Utility function specification
            problem: Problem specification
            transitions: Transition probability matrices
            initial_params: Starting values (defaults to zeros)

        Returns:
            EstimationResult with optimized parameters
        """
        start_time = time.time()

        # Use float64 for high discount factors (condition number ≈ 1/(1-β))
        beta = problem.discount_factor
        if beta > 0.99:
            transitions = transitions.double()

        # Initialize parameters — use data-driven starting values if zeros
        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            if (initial_params == 0).all():
                initial_params = self._estimate_initial_params(
                    panel, utility, problem
                )

        current_params = initial_params.clone()

        # Step 1: Estimate initial CCPs from data
        self._log("Estimating CCPs from data")
        ccps = self._estimate_ccps_from_data(
            panel, problem.num_states, problem.num_actions
        )

        # Track iterations
        num_policy_iterations = 0
        converged = False
        max_iterations = (
            self._num_policy_iterations if self._num_policy_iterations > 0 else 100
        )

        # Step 2: Policy iteration loop
        for k in range(max_iterations):
            num_policy_iterations = k + 1
            self._log(f"Policy iteration {k + 1}")

            prev_params = current_params.clone()

            # Define objective function for this iteration's CCPs
            def objective(params_np):
                params = torch.tensor(params_np, dtype=torch.float32)
                ll = self._compute_log_likelihood(
                    params, panel, utility, ccps, transitions, problem
                )
                return -ll  # Minimize negative log-likelihood

            # Gradient via finite differences (adaptive step size)
            def gradient(params_np):
                grad = torch.zeros(len(params_np))
                for i in range(len(params_np)):
                    eps_i = max(1e-5, abs(params_np[i]) * 1e-4)
                    params_plus = params_np.copy()
                    params_minus = params_np.copy()
                    params_plus[i] += eps_i
                    params_minus[i] -= eps_i
                    grad[i] = (objective(params_plus) - objective(params_minus)) / (2 * eps_i)
                return grad.numpy()

            # Maximize pseudo-likelihood
            lower, upper = utility.get_parameter_bounds()
            bounds = list(zip(lower.numpy(), upper.numpy()))

            result = optimize.minimize(
                objective,
                current_params.numpy(),
                method="L-BFGS-B",
                jac=gradient,
                bounds=bounds,
                options={
                    "maxiter": self._outer_max_iter,
                    "gtol": self._outer_tol,
                    "disp": False,
                },
            )

            current_params = torch.tensor(result.x, dtype=torch.float32)
            current_ll = -result.fun

            self._log(f"  Parameters: {current_params.numpy()}")
            self._log(f"  Log-likelihood: {current_ll:.4f}")

            # Check convergence for NPL
            param_change = torch.norm(current_params - prev_params).item()
            self._log(f"  Parameter change: {param_change:.6f}")

            if param_change < self._convergence_tol:
                converged = True
                self._log("NPL converged!")
                break

            # Update CCPs for next iteration (if doing NPL)
            if k < max_iterations - 1:
                v = self._compute_choice_specific_values(
                    ccps, transitions, utility, current_params, problem
                )
                ccps = self._update_ccps_from_values(v, problem.scale_parameter)

            # Stop if only doing Hotz-Miller (K=1)
            if self._num_policy_iterations == 1:
                break

        # Compute final value function and policy
        v = self._compute_choice_specific_values(
            ccps, transitions, utility, current_params, problem
        )
        final_policy = self._update_ccps_from_values(v, problem.scale_parameter)

        # Compute value function V(s) = σ·log(Σ_a exp(v(a,s)/σ))
        sigma = problem.scale_parameter
        V = sigma * torch.logsumexp(v / sigma, dim=1)

        # Compute final log-likelihood
        final_ll = self._compute_log_likelihood(
            current_params, panel, utility, ccps, transitions, problem
        )

        # Compute Hessian for standard errors
        hessian = None
        gradient_contributions = None

        if self._compute_hessian:
            self._log("Computing Hessian for standard errors")

            def ll_fn(params):
                return torch.tensor(
                    self._compute_log_likelihood(
                        params, panel, utility, ccps, transitions, problem
                    )
                )

            hessian = compute_numerical_hessian(current_params, ll_fn)

            # Compute per-observation gradients for robust SEs
            gradient_contributions = self._compute_gradient_contributions(
                current_params, panel, utility, ccps, transitions, problem
            )

        optimization_time = time.time() - start_time

        return EstimationResult(
            parameters=current_params,
            log_likelihood=final_ll,
            value_function=V,
            policy=final_policy,
            hessian=hessian,
            gradient_contributions=gradient_contributions,
            converged=converged or (self._num_policy_iterations == 1),
            num_iterations=num_policy_iterations,
            num_function_evals=0,  # Not tracked for CCP
            num_inner_iterations=0,  # No inner loop in CCP
            message=f"CCP estimation completed in {num_policy_iterations} policy iterations",
            optimization_time=optimization_time,
            metadata={
                "num_policy_iterations": num_policy_iterations,
                "npl_converged": converged,
            },
        )

    def _compute_gradient_contributions(
        self,
        params: torch.Tensor,
        panel: Panel,
        utility: UtilityFunction,
        ccps: torch.Tensor,
        transitions: torch.Tensor,
        problem: DDCProblem,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Compute per-observation gradient contributions for robust SEs."""
        n_obs = panel.num_observations
        n_params = len(params)

        gradients = torch.zeros((n_obs, n_params))
        sigma = problem.scale_parameter

        # Pre-compute log probabilities at current params
        v_base = self._compute_choice_specific_values(
            ccps, transitions, utility, params, problem
        )
        log_probs_base = torch.nn.functional.log_softmax(v_base / sigma, dim=1)

        # Compute gradient for each parameter
        for k in range(n_params):
            params_plus = params.clone()
            params_plus[k] += eps

            params_minus = params.clone()
            params_minus[k] -= eps

            v_plus = self._compute_choice_specific_values(
                ccps, transitions, utility, params_plus, problem
            )
            log_probs_plus = torch.nn.functional.log_softmax(v_plus / sigma, dim=1)

            v_minus = self._compute_choice_specific_values(
                ccps, transitions, utility, params_minus, problem
            )
            log_probs_minus = torch.nn.functional.log_softmax(v_minus / sigma, dim=1)

            # Compute gradients for all observations
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            gradients[:, k] = (
                log_probs_plus[all_states, all_actions] - log_probs_minus[all_states, all_actions]
            ) / (2 * eps)

        return gradients
