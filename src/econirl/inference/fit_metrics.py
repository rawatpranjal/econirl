"""Predictive fit metrics for dynamic discrete choice models.

Metrics for evaluating how well estimated models predict observed choices:
- Brier score: average squared prediction error
- KL divergence: information-theoretic distance between data and model CCPs
- Efron pseudo R-squared: variance-ratio measure of fit
- CCP consistency test: Pearson chi-squared comparing observed vs model CCPs
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats


def brier_score(
    policy: jnp.ndarray,
    obs_states: jnp.ndarray,
    obs_actions: jnp.ndarray,
) -> dict[str, float]:
    """Brier score for choice prediction accuracy.

    Computes BS = (1/N) sum_i sum_a (pi(a|s_i) - 1[a_i = a])^2.
    Lower is better. For binary choices, ranges from 0 (perfect) to 2 (worst).

    Args:
        policy: Model-implied CCPs, shape (num_states, num_actions).
        obs_states: Observed states, shape (N,), dtype int.
        obs_actions: Observed actions, shape (N,), dtype int.

    Returns:
        Dict with keys: brier_score, brier_score_per_action (list of floats).
    """
    num_actions = policy.shape[1]
    obs_states = jnp.asarray(obs_states, dtype=jnp.int32)
    obs_actions = jnp.asarray(obs_actions, dtype=jnp.int32)

    y_onehot = jax.nn.one_hot(obs_actions, num_actions)
    p_hat = policy[obs_states]  # (N, A)

    squared_errors = (p_hat - y_onehot) ** 2
    bs = float(jnp.mean(jnp.sum(squared_errors, axis=1)))
    bs_per_action = [float(x) for x in jnp.mean(squared_errors, axis=0)]

    return {"brier_score": bs, "brier_score_per_action": bs_per_action}


def kl_divergence(
    data_ccps: jnp.ndarray,
    model_ccps: jnp.ndarray,
    state_frequencies: jnp.ndarray | None = None,
) -> dict[str, float]:
    """KL divergence from model CCPs to data CCPs.

    Computes D_KL = sum_s mu(s) sum_a p_data(a|s) log(p_data(a|s) / p_model(a|s))
    weighted by empirical state frequencies mu(s).

    Args:
        data_ccps: Empirical choice probabilities, shape (num_states, num_actions).
        model_ccps: Model-implied CCPs, shape (num_states, num_actions).
        state_frequencies: State visitation frequencies, shape (num_states,).
            If None, uses uniform weighting.

    Returns:
        Dict with keys: kl_divergence, per_state_kl (list of floats).
    """
    eps = 1e-15
    data_safe = jnp.clip(data_ccps, eps, 1.0)
    model_safe = jnp.clip(model_ccps, eps, 1.0)

    per_state = jnp.sum(data_safe * jnp.log(data_safe / model_safe), axis=1)

    if state_frequencies is None:
        num_states = data_ccps.shape[0]
        state_frequencies = jnp.ones(num_states) / num_states

    kl = float(jnp.sum(state_frequencies * per_state))

    return {"kl_divergence": kl, "per_state_kl": [float(x) for x in per_state]}


def efron_pseudo_r_squared(
    policy: jnp.ndarray,
    obs_states: jnp.ndarray,
    obs_actions: jnp.ndarray,
) -> dict[str, float]:
    """Efron pseudo R-squared for discrete choice models.

    Computes 1 - SSR/SST where SSR and SST use the multinomial indicator
    across all actions. Equivalent to the ratio of the Brier score to
    the null model Brier score.

    Args:
        policy: Model-implied CCPs, shape (num_states, num_actions).
        obs_states: Observed states, shape (N,), dtype int.
        obs_actions: Observed actions, shape (N,), dtype int.

    Returns:
        Dict with keys: efron_r_squared, ssr, sst.
    """
    num_actions = policy.shape[1]
    obs_states = jnp.asarray(obs_states, dtype=jnp.int32)
    obs_actions = jnp.asarray(obs_actions, dtype=jnp.int32)

    y_onehot = jax.nn.one_hot(obs_actions, num_actions)  # (N, A)
    p_hat = policy[obs_states]  # (N, A)

    # y_bar is the mean of each action indicator across observations
    y_bar = jnp.mean(y_onehot, axis=0, keepdims=True)  # (1, A)

    ssr = float(jnp.sum((y_onehot - p_hat) ** 2))
    sst = float(jnp.sum((y_onehot - y_bar) ** 2))

    r_squared = 1.0 - ssr / sst if sst > 0 else 0.0

    return {"efron_r_squared": r_squared, "ssr": ssr, "sst": sst}


def ccp_consistency_test(
    data_ccps: jnp.ndarray,
    model_ccps: jnp.ndarray,
    state_counts: jnp.ndarray,
    num_estimated_params: int = 0,
) -> dict[str, float]:
    """Pearson chi-squared test comparing observed vs model CCPs.

    Tests whether the model-implied choice probabilities are consistent
    with the observed data. The test statistic is
    T = sum_s n(s) sum_a [p_data(a|s) - p_model(a|s)]^2 / p_model(a|s).

    Args:
        data_ccps: Empirical CCPs, shape (num_states, num_actions).
        model_ccps: Model-implied CCPs, shape (num_states, num_actions).
        state_counts: Number of observations per state, shape (num_states,).
        num_estimated_params: Number of estimated parameters for df adjustment.

    Returns:
        Dict with keys: statistic, df, p_value, per_state_statistic (list).
    """
    eps = 1e-15
    model_safe = jnp.clip(model_ccps, eps, 1.0)

    per_state = state_counts * jnp.sum(
        (data_ccps - model_ccps) ** 2 / model_safe, axis=1
    )

    statistic = float(jnp.sum(per_state))

    num_actions = data_ccps.shape[1]
    states_with_obs = int(jnp.sum(state_counts > 0))
    df = states_with_obs * (num_actions - 1) - num_estimated_params
    df = max(df, 1)

    p_value = 1.0 - stats.chi2.cdf(statistic, df)

    return {
        "statistic": statistic,
        "df": df,
        "p_value": float(p_value),
        "per_state_statistic": [float(x) for x in per_state],
    }
