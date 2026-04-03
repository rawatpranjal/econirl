"""Metrics for comparing policies and outcomes in the identification study.

Provides CCP divergences (KL/TV/L1/Linf), occupancy weighting, and
expected-return regret for a candidate policy relative to oracle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np


def softmax(Q: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.softmax(Q, axis=1)


def policy_value(P: jnp.ndarray, r: jnp.ndarray, beta: float, pi: jnp.ndarray,
                 tol: float = 1e-10, max_iter: int = 5000) -> jnp.ndarray:
    """Evaluate a stochastic policy in a discounted MDP.

    Bellman expectation: V_pi(s) = E_a[r(s,a) + beta E_{s'}[V_pi(s')]].
    """
    nS, nA, _ = P.shape
    V = jnp.zeros(nS)
    for _ in range(max_iter):
        EV = jnp.einsum('ijk,k->ij', P, V)
        V_new = jnp.sum(pi * (r + beta * EV), axis=1)
        if float(jnp.max(jnp.abs(V_new - V))) < tol:
            V = V_new
            break
        V = V_new
    return V


def discounted_occupancy(P: jnp.ndarray, pi: jnp.ndarray, beta: float,
                         start: jnp.ndarray | None = None,
                         tol: float = 1e-10, max_iter: int = 10000) -> jnp.ndarray:
    """Compute discounted state occupancy d(s) = (1-beta) sum_t beta^t Pr(S_t=s).

    Uses forward propagation of distributions. Returns d that sums to 1.
    """
    nS = P.shape[0]
    if start is None:
        start = jnp.ones(nS) / nS
    d = jnp.zeros(nS)
    dist = start
    b = 1.0
    for _ in range(max_iter):
        d_new = d + (1 - beta) * b * dist
        if float(jnp.max(jnp.abs(d_new - d))) < tol:
            d = d_new
            break
        d = d_new
        # propagate: dist_{t+1} = dist_t * P_pi
        P_pi = jnp.einsum('ij,ijk->ik', pi, P)
        dist = jnp.einsum('i,ij->j', dist, P_pi)
        b *= beta
    # normalize for safety
    d = d / jnp.sum(d)
    return d


def _safe(p: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return jnp.clip(p, eps, 1.0)


def ccp_divergences(pi_true: jnp.ndarray, pi_hat: jnp.ndarray,
                    weights: jnp.ndarray | None = None) -> Dict[str, float]:
    """Compute per-state CCP divergences and aggregate with weights.

    - L1 (mean absolute difference across actions)
    - Linf (max absolute difference across actions)
    - TV = 0.5 * L1
    - KL(pi_true || pi_hat)
    Aggregation default is uniform over states if weights is None.
    """
    if weights is None:
        weights = jnp.ones(pi_true.shape[0]) / pi_true.shape[0]
    weights = weights / jnp.sum(weights)
    diff = jnp.abs(pi_true - pi_hat)
    l1 = jnp.sum(weights * jnp.sum(diff, axis=1))
    linf = jnp.max(jnp.sum(diff, axis=1))  # sup over states of L1 across actions
    tv = 0.5 * l1
    # KL with small epsilon
    p = _safe(pi_true)
    q = _safe(pi_hat)
    per_state_kl = jnp.sum(p * (jnp.log(p) - jnp.log(q)), axis=1)
    kl = jnp.sum(weights * per_state_kl)
    return {
        "l1": float(l1),
        "linf": float(linf),
        "tv": float(tv),
        "kl": float(kl),
    }


def metrics_for_counterfactual(P_cf: jnp.ndarray, r_true: jnp.ndarray, beta: float,
                               pi_oracle: jnp.ndarray, pi_hat: jnp.ndarray,
                               start_uniform_over: int | None = None) -> Dict[str, float]:
    """Compute complementary metrics for a counterfactual environment.

    Returns a dict with uniform- and occupancy-weighted divergences and
    expected-return regret (avg over a uniform start distribution over first
    `start_uniform_over` states if provided; else over all states).
    """
    nS = P_cf.shape[0]
    if start_uniform_over is None:
        start = jnp.ones(nS) / nS
    else:
        w = jnp.zeros(nS)
        w = w.at[:start_uniform_over].set(1.0 / start_uniform_over)
        start = w
    # Divergences
    div_uniform = ccp_divergences(pi_oracle, pi_hat)
    occ = discounted_occupancy(P_cf, pi_oracle, beta, start)
    div_occ = ccp_divergences(pi_oracle, pi_hat, weights=occ)
    # Regret via policy evaluation
    V_oracle = policy_value(P_cf, r_true, beta, pi_oracle)
    V_hat = policy_value(P_cf, r_true, beta, pi_hat)
    regret_uniform = float(jnp.dot(start, V_oracle - V_hat))
    regret_occ = float(jnp.dot(occ, V_oracle - V_hat))
    out = {}
    for k, v in div_uniform.items():
        out[f"uniform_{k}"] = v
    for k, v in div_occ.items():
        out[f"occupancy_{k}"] = v
    out["regret_uniform"] = regret_uniform
    out["regret_occupancy"] = regret_occ
    return out
