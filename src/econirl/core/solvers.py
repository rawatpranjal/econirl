"""Solvers for the dynamic programming fixed point.

This module provides iterative methods to solve for the value function
and optimal policy in dynamic discrete choice models with logit shocks.

Available solvers:
- value_iteration: Simple fixed-point iteration (guaranteed convergence)
- policy_iteration: Often faster convergence via policy evaluation step
- hybrid_iteration: Contraction + Newton-Kantorovich (best for high beta)

The hybrid solver implements Rust (1987, 2000)'s recommended approach:
1. Start with cheap contraction iterations
2. Switch to Newton-Kantorovich when close to solution
3. Achieve quadratic convergence in the final phase

For high discount factors (beta > 0.99), the hybrid solver can be
10-100x faster than pure contraction (value_iteration).

All methods exploit the contraction property of the soft Bellman operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from econirl.core.bellman import BellmanResult, SoftBellmanOperator
from econirl.core.types import DDCProblem


@dataclass
class SolverResult:
    """Result from solving the dynamic programming problem.

    Attributes:
        Q: Converged action-value function, shape (num_states, num_actions)
        V: Converged value function, shape (num_states,)
        policy: Optimal choice probabilities, shape (num_states, num_actions)
        converged: Whether the solver converged within max_iter
        num_iterations: Number of iterations performed
        final_error: Final convergence error (sup norm of value change)
    """

    Q: torch.Tensor
    V: torch.Tensor
    policy: torch.Tensor
    converged: bool
    num_iterations: int
    final_error: float


def value_iteration(
    operator: SoftBellmanOperator,
    utility: torch.Tensor,
    V_init: torch.Tensor | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> SolverResult:
    """Solve for the fixed point using value iteration.

    Iteratively applies the soft Bellman operator until convergence:
        V_{k+1} = T(V_k)

    Convergence is guaranteed because T is a contraction with modulus β.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions)
        V_init: Initial value function guess. If None, starts at zeros.
        tol: Convergence tolerance (sup norm)
        max_iter: Maximum number of iterations

    Returns:
        SolverResult with converged value function and policy

    Example:
        >>> operator = SoftBellmanOperator(problem, transitions)
        >>> result = value_iteration(operator, utility)
        >>> if result.converged:
        ...     print(f"Converged in {result.num_iterations} iterations")
    """
    num_states = operator.problem.num_states

    if V_init is None:
        V = torch.zeros(num_states, dtype=utility.dtype, device=utility.device)
    else:
        V = V_init.clone()

    converged = False
    final_error = float("inf")

    for iteration in range(max_iter):
        result = operator.apply(utility, V)
        V_new = result.V

        # Check convergence using sup norm
        error = torch.abs(V_new - V).max().item()

        V = V_new

        if error < tol:
            converged = True
            final_error = error
            break

        final_error = error

    # Get final Q and policy
    final_result = operator.apply(utility, V)

    return SolverResult(
        Q=final_result.Q,
        V=final_result.V,
        policy=final_result.policy,
        converged=converged,
        num_iterations=iteration + 1,
        final_error=final_error,
    )


def policy_iteration(
    operator: SoftBellmanOperator,
    utility: torch.Tensor,
    V_init: torch.Tensor | None = None,
    tol: float = 1e-10,
    max_iter: int = 100,
    eval_method: Literal["matrix", "iterative"] = "matrix",
    eval_tol: float = 1e-12,
    eval_max_iter: int = 1000,
) -> SolverResult:
    """Solve for the fixed point using policy iteration.

    Policy iteration alternates between:
    1. Policy evaluation: Given policy π, solve for V^π
    2. Policy improvement: Update policy using new V

    For logit models, the "policy" is the choice probability distribution,
    and policy evaluation requires solving a linear system.

    This can converge faster than value iteration, especially for
    high discount factors.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions)
        V_init: Initial value function guess. If None, starts at zeros.
        tol: Convergence tolerance for policy probabilities
        max_iter: Maximum number of policy iterations
        eval_method: "matrix" for direct solve, "iterative" for fixed-point
        eval_tol: Tolerance for iterative policy evaluation
        eval_max_iter: Max iterations for iterative policy evaluation

    Returns:
        SolverResult with converged value function and policy
    """
    problem = operator.problem
    num_states = problem.num_states
    num_actions = problem.num_actions
    beta = problem.discount_factor
    sigma = problem.scale_parameter

    if V_init is None:
        V = torch.zeros(num_states, dtype=utility.dtype, device=utility.device)
    else:
        V = V_init.clone()

    # Initial policy
    result = operator.apply(utility, V)
    policy = result.policy

    converged = False
    final_error = float("inf")

    for iteration in range(max_iter):
        # Policy evaluation: solve for V given current policy
        if eval_method == "matrix":
            V = _policy_evaluation_matrix(
                utility, policy, operator.transitions, beta, sigma
            )
        else:
            V = _policy_evaluation_iterative(
                utility, policy, operator.transitions, beta, sigma, eval_tol, eval_max_iter
            )

        # Policy improvement
        new_result = operator.apply(utility, V)
        new_policy = new_result.policy

        # Check convergence of policy
        policy_error = torch.abs(new_policy - policy).max().item()

        policy = new_policy

        if policy_error < tol:
            converged = True
            final_error = policy_error
            break

        final_error = policy_error

    # Final values
    final_result = operator.apply(utility, V)

    return SolverResult(
        Q=final_result.Q,
        V=final_result.V,
        policy=final_result.policy,
        converged=converged,
        num_iterations=iteration + 1,
        final_error=final_error,
    )


def _policy_evaluation_matrix(
    utility: torch.Tensor,
    policy: torch.Tensor,
    transitions: torch.Tensor,
    beta: float,
    sigma: float,
) -> torch.Tensor:
    """Evaluate policy by solving linear system.

    Given policy π, solves:
        V^π(s) = Σ_a π(a|s) u(s,a) + σ·H(π(·|s)) + β Σ_a π(a|s) Σ_{s'} P(s'|s,a) V^π(s')

    where H(π) = -Σ_a π(a) log π(a) is Shannon entropy.

    This can be written as: (I - βP^π) V = r^π
    where P^π is the policy-weighted transition matrix.
    """
    num_states = utility.shape[0]
    device = utility.device
    dtype = utility.dtype

    # Expected flow utility under policy (including entropy bonus)
    # r^π(s) = Σ_a π(a|s) u(s,a) + σ·H(π(·|s))
    # where H(π) = -Σ_a π(a) log π(a) is Shannon entropy
    log_policy = torch.log(policy + 1e-10)
    expected_utility = (policy * utility).sum(dim=1)
    entropy_bonus = -sigma * (policy * log_policy).sum(dim=1)
    r_pi = expected_utility + entropy_bonus

    # Policy-weighted transition matrix P^π(s,s') = Σ_a π(a|s) P(s'|s,a)
    # transitions shape: (num_actions, num_states, num_states) = [a, from_s, to_s]
    # policy shape: (num_states, num_actions) = [from_s, a]
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    # Solve (I - βP^π) V = r^π
    A = torch.eye(num_states, device=device, dtype=dtype) - beta * P_pi
    V = torch.linalg.solve(A, r_pi)

    return V


def _policy_evaluation_iterative(
    utility: torch.Tensor,
    policy: torch.Tensor,
    transitions: torch.Tensor,
    beta: float,
    sigma: float,
    tol: float,
    max_iter: int,
) -> torch.Tensor:
    """Evaluate policy by fixed-point iteration.

    Useful when state space is large and matrix solve is expensive.
    """
    num_states = utility.shape[0]
    device = utility.device
    dtype = utility.dtype

    # Expected flow utility under policy (including entropy bonus)
    # r^π(s) = Σ_a π(a|s) u(s,a) + σ·H(π(·|s))
    # where H(π) = -Σ_a π(a) log π(a) is Shannon entropy
    log_policy = torch.log(policy + 1e-10)
    expected_utility = (policy * utility).sum(dim=1)
    entropy_bonus = -sigma * (policy * log_policy).sum(dim=1)
    r_pi = expected_utility + entropy_bonus

    # Policy-weighted transition matrix
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    # Iterate: V_{k+1} = r^π + β P^π V_k
    V = torch.zeros(num_states, device=device, dtype=dtype)

    for _ in range(max_iter):
        V_new = r_pi + beta * (P_pi @ V)
        if torch.abs(V_new - V).max() < tol:
            return V_new
        V = V_new

    return V


def _newton_kantorovich_step(
    V: torch.Tensor,
    utility: torch.Tensor,
    operator: SoftBellmanOperator,
) -> tuple[torch.Tensor, float]:
    """Perform a single Newton-Kantorovich iteration.

    The NK update is:
        V_{k+1} = V_k + (I - βP)⁻¹ [T(V_k) - V_k]

    where:
    - T is the soft Bellman operator
    - P is the policy-weighted transition matrix at current V
    - (I - βP)⁻¹ accelerates convergence via Newton's method

    This achieves quadratic convergence near the fixed point.

    Args:
        V: Current value function estimate
        utility: Flow utility matrix, shape (num_states, num_actions)
        operator: SoftBellmanOperator instance

    Returns:
        Tuple of (updated value function, post-update residual norm)

    Reference:
        Rust (2000) NFXP Manual, Section 3.2
    """
    # Apply Bellman operator to get T(V_k)
    result = operator.apply(utility, V)
    V_bellman = result.V

    # Compute residual: T(V_k) - V_k
    residual = V_bellman - V

    # Policy-weighted transition matrix P^π(s,s') = Σ_a π(a|s) P(s'|s,a)
    # transitions shape: (num_actions, num_states, num_states) = [a, from_s, to_s]
    # policy shape: (num_states, num_actions) = [from_s, a]
    P_pi = torch.einsum("sa,ast->st", result.policy, operator.transitions)

    # Solve (I - βP) Δ = residual for the Newton correction Δ
    beta = operator.problem.discount_factor
    num_states = len(V)
    A = torch.eye(num_states, device=V.device, dtype=V.dtype) - beta * P_pi
    delta = torch.linalg.solve(A, residual)

    # Apply Newton correction
    V_new = V + delta

    # Compute post-update residual for convergence check
    result_new = operator.apply(utility, V_new)
    post_residual_norm = torch.abs(result_new.V - V_new).max().item()

    return V_new, post_residual_norm


def hybrid_iteration(
    operator: SoftBellmanOperator,
    utility: torch.Tensor,
    V_init: torch.Tensor | None = None,
    tol: float = 1e-10,
    max_iter: int = 1000,
    switch_tol: float = 1e-3,
    max_nk_iter: int = 20,
) -> SolverResult:
    """Solve for the fixed point using hybrid contraction + Newton-Kantorovich.

    This implements the hybrid algorithm from Rust (1987, 2000):
    1. Run contraction iterations (value iteration) until error < switch_tol
    2. Switch to Newton-Kantorovich iterations for quadratic convergence
    3. Typically converges in just 1-2 NK iterations after the switch

    The NK iteration is:
        V_{k+1} = V_k + (I - βP)⁻¹ [T(V_k) - V_k]

    This exploits the structure of the Bellman fixed-point equation to
    achieve quadratic convergence, much faster than the linear convergence
    of pure contraction.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions)
        V_init: Initial value function guess. If None, starts at zeros.
        tol: Final convergence tolerance (sup norm)
        max_iter: Maximum total iterations (contraction + NK)
        switch_tol: Switch from contraction to NK when error < this
        max_nk_iter: Maximum NK iterations after switch

    Returns:
        SolverResult with converged value function and policy

    Example:
        >>> operator = SoftBellmanOperator(problem, transitions)
        >>> result = hybrid_iteration(operator, utility, switch_tol=1e-3)
        >>> print(f"Converged in {result.num_iterations} iterations")

    Reference:
        Rust, J. (2000). "NFXP Manual"
        Iskhakov et al. (2016). "The Endurance of First-Price Auctions"
    """
    num_states = operator.problem.num_states

    if V_init is None:
        V = torch.zeros(num_states, dtype=utility.dtype, device=utility.device)
    else:
        V = V_init.clone()

    converged = False
    final_error = float("inf")
    iteration = 0
    switch_to_nk = False

    # Phase 1: Contraction iterations until close to solution
    for iteration in range(max_iter):
        result = operator.apply(utility, V)
        V_new = result.V

        error = torch.abs(V_new - V).max().item()
        V = V_new
        final_error = error

        if error < tol:
            # Already converged during contraction phase
            converged = True
            break

        if error < switch_tol:
            # Switch to NK phase
            switch_to_nk = True
            break

    contraction_iters = iteration + 1

    # Phase 2: Newton-Kantorovich iterations for quadratic convergence
    if not converged and switch_to_nk:
        for nk_iter in range(max_nk_iter):
            V, error = _newton_kantorovich_step(V, utility, operator)

            if error < tol:
                converged = True
                final_error = error
                break

            final_error = error

        iteration = contraction_iters + nk_iter + 1
    else:
        iteration = contraction_iters

    # Get final Q and policy
    final_result = operator.apply(utility, V)

    return SolverResult(
        Q=final_result.Q,
        V=final_result.V,
        policy=final_result.policy,
        converged=converged,
        num_iterations=iteration,
        final_error=final_error,
    )


@dataclass
class FiniteHorizonResult:
    """Result from solving a finite-horizon dynamic programming problem.

    All tensors are time-indexed: index 0 is the first period, T-1 is the last.

    Attributes:
        Q: Action-value functions, shape (num_periods, num_states, num_actions)
        V: Value functions, shape (num_periods, num_states)
        policy: Choice probabilities, shape (num_periods, num_states, num_actions)
        num_periods: Number of time periods
    """

    Q: torch.Tensor
    V: torch.Tensor
    policy: torch.Tensor
    num_periods: int


def backward_induction(
    operator: SoftBellmanOperator,
    utility: torch.Tensor,
    num_periods: int,
    terminal_value: torch.Tensor | None = None,
) -> FiniteHorizonResult:
    """Solve finite-horizon DDC by backward induction.

    For t = T-1, T-2, ..., 0:
        Q_t(s,a) = u(s,a) + β Σ_{s'} P(s'|s,a) V_{t+1}(s')
        V_t(s) = σ log(Σ_a exp(Q_t(s,a)/σ))
        π_t(a|s) = softmax(Q_t(s,a)/σ)

    This is the standard finite-horizon solution for DDC models
    (Keane & Wolpin 1994, Ziebart 2008). No convergence criterion
    needed — the backward pass is deterministic.

    Args:
        operator: SoftBellmanOperator instance
        utility: Flow utility matrix, shape (num_states, num_actions).
            If time-varying, shape (num_periods, num_states, num_actions).
        num_periods: Number of decision periods T
        terminal_value: Terminal value function V_T(s), shape (num_states,).
            Defaults to zeros (no continuation value after horizon).

    Returns:
        FiniteHorizonResult with time-indexed Q, V, and policy tensors.

    Example:
        >>> operator = SoftBellmanOperator(problem, transitions)
        >>> result = backward_induction(operator, utility, num_periods=10)
        >>> print(result.policy.shape)  # (10, num_states, num_actions)
    """
    num_states = operator.problem.num_states
    num_actions = operator.problem.num_actions
    dtype = utility.dtype
    device = utility.device

    # Time-varying or stationary utility
    time_varying = utility.dim() == 3
    if time_varying and utility.shape[0] != num_periods:
        raise ValueError(
            f"Time-varying utility has {utility.shape[0]} periods, "
            f"expected {num_periods}"
        )

    # Allocate output tensors
    all_Q = torch.zeros(num_periods, num_states, num_actions, dtype=dtype, device=device)
    all_V = torch.zeros(num_periods, num_states, dtype=dtype, device=device)
    all_policy = torch.zeros(num_periods, num_states, num_actions, dtype=dtype, device=device)

    # Terminal value
    if terminal_value is None:
        V_next = torch.zeros(num_states, dtype=dtype, device=device)
    else:
        V_next = terminal_value.clone()

    # Backward pass: t = T-1, T-2, ..., 0
    for t in range(num_periods - 1, -1, -1):
        u_t = utility[t] if time_varying else utility
        result = operator.apply(u_t, V_next)

        all_Q[t] = result.Q
        all_V[t] = result.V
        all_policy[t] = result.policy

        V_next = result.V

    return FiniteHorizonResult(
        Q=all_Q,
        V=all_V,
        policy=all_policy,
        num_periods=num_periods,
    )


def solve(
    problem: DDCProblem,
    transitions: torch.Tensor,
    utility: torch.Tensor,
    method: Literal["value", "policy", "hybrid"] = "value",
    **kwargs,
) -> SolverResult:
    """Convenience function to solve a DDC problem.

    Args:
        problem: DDCProblem specification
        transitions: Transition matrices, shape (num_actions, num_states, num_states)
        utility: Flow utility matrix, shape (num_states, num_actions)
        method: "value" for value iteration, "policy" for policy iteration
        **kwargs: Additional arguments passed to the solver

    Returns:
        SolverResult with solution
    """
    operator = SoftBellmanOperator(problem, transitions)

    if method == "value":
        return value_iteration(operator, utility, **kwargs)
    elif method == "policy":
        return policy_iteration(operator, utility, **kwargs)
    elif method == "hybrid":
        return hybrid_iteration(operator, utility, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'value', 'policy', or 'hybrid'.")
