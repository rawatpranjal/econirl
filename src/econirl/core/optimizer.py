"""JAX-native optimization wrappers replacing scipy.optimize.minimize.

All structural estimators use bounded L-BFGS for the outer theta optimization.
This module provides a unified interface backed by jaxopt.LBFGSB (bounded) and
jaxopt.LBFGS (unbounded), replacing scipy.optimize.minimize throughout the
codebase.

The solver always runs step-by-step (no unrolling) to avoid catastrophic
compile times on complex objectives. When verbose=True a tqdm bar with
rich diagnostics is displayed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

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
    param_names: Sequence[str] | None = None,
    jit: bool = True,
    fun_args: tuple = (),
) -> OptimizeResult:
    """Bounded L-BFGS optimization using jaxopt.

    Always runs step-by-step to avoid unrolling the full iteration count
    at JAX compile time (which can take minutes for complex objectives).
    When verbose=True, a tqdm bar shows objective value, gradient norm,
    relative change, and parameter values.

    Parameters
    ----------
    fun : callable
        Objective function. If value_and_grad is False, must be f(x) -> scalar
        and must be JAX-differentiable. If value_and_grad is True, must be
        f(x) -> (scalar, grad_array).
    x0 : jnp.ndarray
        Initial parameter vector.
    bounds : tuple of (lower, upper) arrays, optional
        Box constraints. If None, uses unbounded LBFGS.
    maxiter : int
        Maximum iterations.
    tol : float
        Gradient tolerance for convergence.
    verbose : bool
        If True, show tqdm progress bar with rich diagnostics.
    desc : str
        Description for the progress bar.
    value_and_grad : bool
        If True, fun returns (value, gradient) instead of just the value.
    param_names : list of str, optional
        Names for the first few parameters. Shown in tqdm postfix.
    jit : bool
        If True (default), JIT-compile the solver steps. Set to False when the
        objective function uses Python control flow (float(), bool(), in-place
        assignment) that is not compatible with JAX tracing.
    fun_args : tuple, optional
        Extra positional arguments passed to fun as fun(x, *fun_args).
        Passing dynamic arrays here (rather than closing over them) allows
        jaxopt to JIT-compile the solver once and reuse it across calls with
        different concrete values of those arrays.

    Returns
    -------
    OptimizeResult
        Result with .x, .fun, .success, .nit, .nfev, .message fields.
    """
    x0 = jnp.asarray(x0, dtype=jnp.float64)

    # jaxopt.LBFGSB.init_state / update signature:
    #   init_state(init_params, bounds, *args)
    #   update(params, state, bounds, *args)
    # where *args are extra positional args forwarded to fun(params, *args).
    # jaxopt.LBFGS does not have a bounds arg.
    if bounds is not None:
        lower = jnp.asarray(bounds[0], dtype=jnp.float64)
        upper = jnp.asarray(bounds[1], dtype=jnp.float64)
        bounds_val = (lower, upper)
        solver = jaxopt.LBFGSB(
            fun=fun, value_and_grad=value_and_grad,
            maxiter=maxiter, tol=tol, jit=jit,
        )
    else:
        bounds_val = None
        solver = jaxopt.LBFGS(
            fun=fun, value_and_grad=value_and_grad,
            maxiter=maxiter, tol=tol, jit=jit,
        )

    # Step-by-step loop. Only the first iteration incurs JIT cost; subsequent
    # iterations reuse the compiled update.
    if bounds_val is not None:
        state = solver.init_state(x0, bounds_val, *fun_args)
    else:
        state = solver.init_state(x0, *fun_args)
    params = x0
    prev_val = float("inf")

    pbar = tqdm(range(maxiter), desc=desc, disable=not verbose, leave=True)
    nit = 0
    for i in pbar:
        if bounds_val is not None:
            params, state = solver.update(params, state, bounds_val, *fun_args)
        else:
            params, state = solver.update(params, state, *fun_args)
        nit = i + 1

        fval = float(state.value)
        gnorm = float(jnp.linalg.norm(state.grad))
        rel_change = abs(fval - prev_val) / max(abs(prev_val), 1e-12)
        prev_val = fval

        if verbose:
            postfix = {
                "obj": f"{fval:.4f}",
                "|g|": f"{gnorm:.1e}",
                "dobj": f"{rel_change:.1e}",
            }
            if param_names is not None:
                p = params.ravel()
                for j, name in enumerate(param_names[:3]):
                    if j < len(p):
                        postfix[name] = f"{float(p[j]):.5f}"
            pbar.set_postfix(postfix)

        if gnorm < tol:
            if verbose:
                pbar.set_postfix({**postfix, "status": "converged"})
            break

        # Stop if objective has plateaued (relative change tiny for 5 iters)
        if i >= 5 and rel_change < tol * 0.01:
            if verbose:
                pbar.set_postfix({**postfix, "status": "plateaued"})
            break

    pbar.close()

    # Final convergence check — call fun with extra args
    if value_and_grad:
        final_val, final_grad = fun(params, *fun_args)
        final_val = float(final_val)
        grad_norm = float(jnp.linalg.norm(jnp.asarray(final_grad)))
    else:
        _fun_no_args = lambda p: fun(p, *fun_args)  # noqa: E731
        final_val = float(_fun_no_args(params))
        grad_norm = float(jnp.linalg.norm(jax.grad(_fun_no_args)(params)))

    converged = grad_norm < tol * 10

    return OptimizeResult(
        x=params,
        fun=final_val,
        success=converged,
        nit=nit,
        nfev=nit,
        message="Converged" if converged else "Maximum iterations reached",
    )
