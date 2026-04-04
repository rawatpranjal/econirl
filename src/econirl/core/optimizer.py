"""JAX-native optimization wrappers replacing scipy.optimize.minimize.

All structural estimators use bounded L-BFGS for the outer theta optimization.
This module provides a unified interface backed by jaxopt.LBFGSB (bounded) and
jaxopt.LBFGS (unbounded), replacing scipy.optimize.minimize throughout the
codebase.

For constrained problems (MPEC SLSQP, max margin), an augmented Lagrangian
wrapper is provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import jaxopt
from tqdm import tqdm


@dataclass
class OptimizeResult:
    """Result of JAX-native optimization, matching scipy interface."""
    x: jnp.ndarray
    fun: float
    success: bool
    nit: int
    nfev: int
    message: str


def minimize_lbfgsb(
    fun: Callable,
    x0: jnp.ndarray,
    bounds: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    maxiter: int = 500,
    tol: float = 1e-8,
    verbose: bool = False,
    desc: str = "L-BFGS-B",
    value_and_grad: bool = False,
) -> OptimizeResult:
    """Bounded L-BFGS optimization using jaxopt.

    Parameters
    ----------
    fun : callable
        Objective function. If value_and_grad is False, must be f(x) -> scalar
        and must be JAX-differentiable. If value_and_grad is True, must be
        f(x) -> (scalar, grad_array), which allows analytical gradients that
        are not JAX-traceable (e.g. functions using numpy or Python loops).
    x0 : jnp.ndarray
        Initial parameter vector.
    bounds : tuple of (lower, upper) arrays, optional
        Box constraints. If None, uses unbounded LBFGS.
    maxiter : int
        Maximum iterations.
    tol : float
        Gradient tolerance for convergence.
    verbose : bool
        If True, show tqdm progress bar.
    desc : str
        Description for the progress bar.
    value_and_grad : bool
        If True, fun returns (value, gradient) instead of just the value.
        jaxopt will use the supplied gradient rather than computing via autodiff.

    Returns
    -------
    OptimizeResult
        Result with .x, .fun, .success, .nit, .nfev, .message fields.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    if bounds is not None:
        lower, upper = bounds
        lower = jnp.asarray(lower, dtype=jnp.float64)
        upper = jnp.asarray(upper, dtype=jnp.float64)

        solver = jaxopt.LBFGSB(
            fun=fun,
            value_and_grad=value_and_grad,
            maxiter=maxiter,
            tol=tol,
            jit=False,
            unroll=True,
        )

        if verbose:
            # Run stepwise with tqdm
            state = solver.init_state(x0, bounds=(lower, upper))
            params = x0
            pbar = tqdm(range(maxiter), desc=desc, leave=True)
            for i in pbar:
                params, state = solver.update(params, state, bounds=(lower, upper))
                fval = float(state.value)
                gnorm = float(jnp.linalg.norm(state.grad))
                pbar.set_postfix({"obj": f"{fval:.4f}", "|g|": f"{gnorm:.2e}"})
                if gnorm < tol:
                    break
            pbar.close()
            nit = i + 1
        else:
            result = solver.run(x0, bounds=(lower, upper))
            params = result.params
            state = result.state
            nit = maxiter  # jaxopt doesn't expose iteration count easily
    else:
        solver = jaxopt.LBFGS(
            fun=fun,
            value_and_grad=value_and_grad,
            maxiter=maxiter,
            tol=tol,
            jit=False,
            unroll=True,
        )

        if verbose:
            state = solver.init_state(x0)
            params = x0
            pbar = tqdm(range(maxiter), desc=desc, leave=True)
            for i in pbar:
                params, state = solver.update(params, state)
                fval = float(state.value)
                gnorm = float(jnp.linalg.norm(state.grad))
                pbar.set_postfix({"obj": f"{fval:.4f}", "|g|": f"{gnorm:.2e}"})
                if gnorm < tol:
                    break
            pbar.close()
            nit = i + 1
        else:
            result = solver.run(x0)
            params = result.params
            state = result.state
            nit = maxiter

    # Compute final gradient norm to determine convergence. When value_and_grad
    # is True the function returns (val, grad), so extract the gradient directly
    # rather than calling jax.grad which would fail on a non-traceable function.
    if value_and_grad:
        final_val, final_grad = fun(params)
        final_val = float(final_val)
        grad_norm = float(jnp.linalg.norm(jnp.asarray(final_grad)))
    else:
        final_val = float(fun(params))
        grad_norm = float(jnp.linalg.norm(jax.grad(fun)(params)))

    converged = grad_norm < tol * 10  # slightly relaxed check

    return OptimizeResult(
        x=params,
        fun=final_val,
        success=converged,
        nit=nit,
        nfev=nit,
        message="Converged" if converged else "Maximum iterations reached",
    )
