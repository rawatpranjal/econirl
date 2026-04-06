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

import jax
import jax.numpy as jnp
import numpy as np

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.optimizer import minimize_lbfgsb
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
    ) -> jnp.ndarray:
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
        counts = jnp.zeros(num_states * num_actions, dtype=jnp.float32).at[
            idx.astype(jnp.int32)
        ].add(jnp.ones(idx.shape[0])).reshape(num_states, num_actions)

        # Get state counts
        state_counts = counts.sum(axis=1)

        # Compute CCPs with smoothing
        ccps = jnp.zeros((num_states, num_actions), dtype=jnp.float32)
        for s in range(num_states):
            if state_counts[s] >= self._ccp_min_count:
                # Use empirical frequencies with smoothing
                ccps = ccps.at[s].set(
                    (counts[s] + self._ccp_smoothing)
                    / (state_counts[s] + num_actions * self._ccp_smoothing)
                )
            else:
                # Not enough data: use uniform distribution
                ccps = ccps.at[s].set(1.0 / num_actions)

        return ccps

    def _compute_emax_correction(self, ccps: jnp.ndarray) -> jnp.ndarray:
        """Compute emax correction for logit errors.

        e(a,x) = γ - log(P(a|x))

        where γ ≈ 0.5772 is Euler's constant.

        Args:
            ccps: CCP matrix of shape (num_states, num_actions)

        Returns:
            Emax correction matrix of shape (num_states, num_actions)
        """
        # Clamp CCPs to avoid log(0)
        ccps_safe = jnp.maximum(ccps, self._ccp_smoothing)
        return EULER_GAMMA - jnp.log(ccps_safe)

    def _compute_policy_weighted_transitions(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
    ) -> jnp.ndarray:
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
        F_pi = jnp.zeros((num_states, num_states), dtype=dtype)
        for a in range(num_actions):
            # transitions[a] has shape (num_states, num_states)
            # ccps[:, a] has shape (num_states,)
            # We want F_pi[s, s'] += P(a|s) * P(s'|s,a)
            F_pi = F_pi + ccps[:, a:a+1].astype(dtype) * transitions[a]

        return F_pi

    def _compute_valuation_matrix(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        utility: UtilityFunction,
        parameters: jnp.ndarray,
        problem: DDCProblem,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
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
        I = jnp.eye(num_states, dtype=dtype)
        inv_matrix = jnp.linalg.inv(I - beta * F_pi)

        # Compute emax corrections
        e = self._compute_emax_correction(ccps)

        # Compute Σ_a P(a) ⊙ e(a) for each state
        # This gives expected emax correction under current policy
        expected_e = (ccps * e).sum(axis=1).astype(inv_matrix.dtype)  # shape (num_states,)

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
            expected_features = jnp.einsum('sa,sak->sk', ccps.astype(dtype), features.astype(dtype))

            # W_z = (I - β·F_π)⁻¹ · expected_features
            W_z = inv_matrix @ expected_features
        else:
            # Fallback: compute utility directly
            flow_utility = utility.compute(parameters)  # shape (num_states, num_actions)
            expected_utility = (ccps * flow_utility).sum(axis=1)  # shape (num_states,)
            W_z = inv_matrix @ expected_utility
            W_z = W_z[:, None]  # shape (num_states, 1)

        return W_z, W_e

    def _compute_choice_specific_values(
        self,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        utility: UtilityFunction,
        parameters: jnp.ndarray,
        problem: DDCProblem,
    ) -> jnp.ndarray:
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
            E_W_z = jnp.zeros((num_states, num_actions, num_features), dtype=transitions.dtype)
            for a in range(num_actions):
                E_W_z = E_W_z.at[:, a, :].set(transitions[a] @ W_z)  # (num_states, num_features)

            # Compute E[W_e(x') | x, a]
            E_W_e = jnp.zeros((num_states, num_actions), dtype=transitions.dtype)
            for a in range(num_actions):
                E_W_e = E_W_e.at[:, a].set(transitions[a] @ W_e)

            # z̃(a,x) = z(a,x) + β·E[W_z(x') | x, a]
            z_tilde = features.astype(E_W_z.dtype) + beta * E_W_z  # (num_states, num_actions, num_features)

            # ẽ(a,x) = β·E[W_e(x') | x, a]
            e_tilde = beta * E_W_e  # (num_states, num_actions)

            # v(a,x) = z̃(a,x)'θ + ẽ(a,x)
            v = jnp.einsum('sak,k->sa', z_tilde, parameters.astype(z_tilde.dtype)) + e_tilde
        else:
            # Fallback for non-linear utility
            flow_utility = utility.compute(parameters)
            W = W_z.squeeze(1) + W_e

            EW = jnp.zeros((num_states, num_actions), dtype=transitions.dtype)
            for a in range(num_actions):
                EW = EW.at[:, a].set(transitions[a] @ W)

            v = flow_utility + beta * EW

        return v

    def _compute_log_likelihood(
        self,
        parameters: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
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
        log_probs = jax.nn.log_softmax(v / sigma, axis=1)

        # Sum log-likelihood over observations
        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        ll = float(log_probs[all_states, all_actions].sum())

        return ll

    def _update_ccps_from_values(
        self,
        v: jnp.ndarray,
        sigma: float,
    ) -> jnp.ndarray:
        """Update CCPs from choice-specific values.

        P(a|x) = exp(v(a,x)/σ) / Σ_{a'} exp(v(a',x)/σ)

        Args:
            v: Choice-specific values of shape (num_states, num_actions)
            sigma: Scale parameter

        Returns:
            Updated CCPs of shape (num_states, num_actions)
        """
        return jax.nn.softmax(v / sigma, axis=1)

    def _estimate_initial_params(
        self,
        panel: Panel,
        utility: UtilityFunction,
        problem: DDCProblem,
    ) -> jnp.ndarray:
        """Estimate rough starting values from data."""
        n_params = utility.num_parameters
        total_obs = 0
        n_replace = 0
        mileage_at_replace = 0.0

        all_states = panel.get_all_states()
        all_actions = panel.get_all_actions()
        total_obs = all_states.shape[0]
        replace_mask = all_actions == 1
        n_replace = int(replace_mask.sum())
        mileage_at_replace = float(all_states[replace_mask].astype(jnp.float32).sum())

        if n_replace > 0 and total_obs > 0 and n_params == 2:
            replace_rate = n_replace / total_obs
            avg_mileage = mileage_at_replace / n_replace
            n_states = problem.num_states
            op_cost_init = 1.0 / n_states
            rc_init = max(0.5, op_cost_init * avg_mileage / max(replace_rate, 0.01))
            return jnp.array([op_cost_init, rc_init], dtype=jnp.float32)

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
            transitions = jnp.array(transitions, dtype=jnp.float64)

        # Initialize parameters — use data-driven starting values if zeros
        if initial_params is None:
            initial_params = utility.get_initial_parameters()
            if (initial_params == 0).all():
                initial_params = self._estimate_initial_params(
                    panel, utility, problem
                )

        current_params = jnp.array(initial_params)

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
        from tqdm import tqdm
        pbar = tqdm(
            range(max_iterations),
            desc="CCP NPL",
            disable=not self._verbose,
            leave=True,
        )
        for k in pbar:
            num_policy_iterations = k + 1

            prev_params = jnp.array(current_params)

            # A&M (2002) algorithm: compute the valuation matrix ONCE per NPL step
            # from the fixed CCPs. W_z and W_e depend only on CCPs (not theta), so
            # they do not change during the logit optimization step. Caching them
            # reduces per-call cost from O(S^3) to O(S*A*K) for the pseudo-LL.
            W_z, W_e = self._compute_valuation_matrix(
                ccps, transitions, utility, current_params, problem
            )

            # Pre-compute augmented features z_tilde and e_tilde from cached W_z, W_e.
            # v(s,a; theta) = z_tilde(s,a)' * theta + e_tilde(s,a)
            # This is a standard logit — no matrix inversion per gradient call.
            beta = problem.discount_factor
            num_states = problem.num_states
            num_actions = problem.num_actions

            if hasattr(utility, 'feature_matrix'):
                features = jnp.array(utility.feature_matrix, dtype=W_z.dtype)
                # E[W_z(x') | x, a]: shape (S, A, K)
                E_W_z = jnp.stack([transitions[a] @ W_z for a in range(num_actions)], axis=1)
                E_W_e = jnp.stack([transitions[a] @ W_e for a in range(num_actions)], axis=1)
                z_tilde = features + beta * E_W_z          # (S, A, K)
                e_tilde = beta * E_W_e                     # (S, A)
            else:
                z_tilde = None
                e_tilde = None

            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            sigma = problem.scale_parameter

            def neg_ll_fast(params_x):
                """Pseudo-LL using precomputed augmented features — no matrix inversion."""
                if z_tilde is not None:
                    theta = jnp.array(params_x, dtype=z_tilde.dtype)
                    v = jnp.einsum('sak,k->sa', z_tilde, theta) + e_tilde
                else:
                    # Fallback for non-linear utility (rare)
                    return -self._compute_log_likelihood(
                        jnp.array(params_x, dtype=jnp.float32),
                        panel, utility, ccps, transitions, problem,
                    )
                log_probs = jax.nn.log_softmax(v / sigma, axis=1)
                return -float(log_probs[all_states, all_actions].sum())

            def neg_ll_and_grad(params_x):
                val = neg_ll_fast(params_x)
                n = len(params_x)
                grad = jnp.zeros(n, dtype=jnp.float64)
                for i in range(n):
                    eps_i = max(1e-5, float(jnp.abs(params_x[i])) * 1e-4)
                    p_plus = params_x.at[i].add(eps_i)
                    p_minus = params_x.at[i].add(-eps_i)
                    grad = grad.at[i].set(
                        (neg_ll_fast(p_plus) - neg_ll_fast(p_minus)) / (2 * eps_i)
                    )
                return val, grad

            # Maximize pseudo-likelihood (standard logit with augmented features)
            lower, upper = utility.get_parameter_bounds()

            result = minimize_lbfgsb(
                neg_ll_and_grad,
                jnp.array(current_params, dtype=jnp.float64),
                bounds=(jnp.asarray(lower, dtype=jnp.float64),
                        jnp.asarray(upper, dtype=jnp.float64)),
                maxiter=self._outer_max_iter,
                tol=self._outer_tol,
                verbose=False,
                desc="CCP",
                value_and_grad=True,
                param_names=list(utility.parameter_names),
                jit=False,
            )

            current_params = jnp.array(result.x, dtype=jnp.float32)
            current_ll = -result.fun

            # Check convergence for NPL
            param_change = float(jnp.linalg.norm(current_params - prev_params))
            postfix = {"LL": f"{current_ll:.2f}", "d_param": f"{param_change:.1e}"}
            for j, nm in enumerate(utility.parameter_names[:3]):
                postfix[nm] = f"{float(current_params[j]):.5f}"
            pbar.set_postfix(postfix)

            if param_change < self._convergence_tol:
                converged = True
                pbar.set_postfix({**postfix, "status": "converged"})
                pbar.close()
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
        V = sigma * jax.scipy.special.logsumexp(v / sigma, axis=1)

        # Compute final log-likelihood
        final_ll = self._compute_log_likelihood(
            current_params, panel, utility, ccps, transitions, problem
        )

        # Compute Hessian for standard errors
        hessian = None
        gradient_contributions = None

        if self._compute_hessian:
            self._log("Computing Hessian for standard errors")

            # Use the FULL log-likelihood (re-solve Bellman at each perturbation)
            # rather than the CCP pseudo-likelihood with fixed CCPs. The pseudo-LL
            # with fixed CCPs is locally flat in directions that primarily affect
            # CCPs (like replacement_cost), causing the Hessian to be rank-deficient.
            operator = SoftBellmanOperator(problem, transitions)

            def ll_fn(params):
                flow_u = utility.compute(params).astype(transitions.dtype)
                from econirl.core.solvers import value_iteration
                sol = value_iteration(operator, flow_u, tol=1e-12, max_iter=100_000)
                log_probs = operator.compute_log_choice_probabilities(flow_u, sol.V)
                all_states = panel.get_all_states()
                all_actions = panel.get_all_actions()
                return log_probs[all_states, all_actions].sum()

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
        params: jnp.ndarray,
        panel: Panel,
        utility: UtilityFunction,
        ccps: jnp.ndarray,
        transitions: jnp.ndarray,
        problem: DDCProblem,
        eps: float = 1e-5,
    ) -> jnp.ndarray:
        """Compute per-observation gradient contributions for robust SEs."""
        n_obs = panel.num_observations
        n_params = len(params)

        gradients = jnp.zeros((n_obs, n_params))
        sigma = problem.scale_parameter

        # Pre-compute log probabilities at current params
        v_base = self._compute_choice_specific_values(
            ccps, transitions, utility, params, problem
        )
        log_probs_base = jax.nn.log_softmax(v_base / sigma, axis=1)

        # Compute gradient for each parameter
        for k in range(n_params):
            params_plus = params.at[k].add(eps)
            params_minus = params.at[k].add(-eps)

            v_plus = self._compute_choice_specific_values(
                ccps, transitions, utility, params_plus, problem
            )
            log_probs_plus = jax.nn.log_softmax(v_plus / sigma, axis=1)

            v_minus = self._compute_choice_specific_values(
                ccps, transitions, utility, params_minus, problem
            )
            log_probs_minus = jax.nn.log_softmax(v_minus / sigma, axis=1)

            # Compute gradients for all observations
            all_states = panel.get_all_states()
            all_actions = panel.get_all_actions()
            gradients = gradients.at[:, k].set(
                (log_probs_plus[all_states, all_actions] - log_probs_minus[all_states, all_actions])
                / (2 * eps)
            )

        return gradients
