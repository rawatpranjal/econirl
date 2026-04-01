"""Hypothesis tests for model comparison and parameter restrictions.

Provides classical hypothesis tests for structural estimation:
- Likelihood Ratio (LR) test for nested model comparison
- Score (Lagrange Multiplier) test for restrictions at the restricted estimate
- Vuong test for non-nested model comparison

Each function returns a dict with test statistic, degrees of freedom,
and p-value, matching the pattern of EstimationSummary.wald_test().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from scipy import stats

if TYPE_CHECKING:
    from econirl.inference.results import EstimationSummary


def likelihood_ratio_test(
    restricted: EstimationSummary,
    unrestricted: EstimationSummary,
) -> dict[str, float]:
    """Likelihood ratio test for nested models.

    Tests H0 (restricted model is correct) against H1 (unrestricted model).
    The test statistic LR = -2 * [l(theta_R) - l(theta_U)] is
    asymptotically chi-squared with degrees of freedom equal to the
    difference in the number of parameters.

    Args:
        restricted: EstimationSummary from the restricted (null) model.
        unrestricted: EstimationSummary from the unrestricted (alternative) model.

    Returns:
        Dict with keys: statistic, df, p_value.

    Raises:
        ValueError: If log-likelihood is missing from either model or if
            the restricted model has more parameters than the unrestricted.
    """
    if restricted.log_likelihood is None or unrestricted.log_likelihood is None:
        raise ValueError("Both models must have log_likelihood computed")

    df = unrestricted.num_parameters - restricted.num_parameters
    if df <= 0:
        raise ValueError(
            f"Unrestricted model must have more parameters than restricted. "
            f"Got {unrestricted.num_parameters} vs {restricted.num_parameters}"
        )

    statistic = -2.0 * (restricted.log_likelihood - unrestricted.log_likelihood)
    if statistic < 0:
        statistic = 0.0  # Numerical noise can produce tiny negatives

    p_value = 1.0 - stats.chi2.cdf(statistic, df)

    return {"statistic": float(statistic), "df": df, "p_value": float(p_value)}


def score_test(
    score: jnp.ndarray,
    information_matrix: jnp.ndarray,
    df: int,
) -> dict[str, float]:
    """Score (Lagrange Multiplier) test at the restricted estimate.

    Tests H0 (restrictions are valid) using only the restricted estimate.
    The test statistic LM = S' I^{-1} S is asymptotically chi-squared.

    The score vector is the gradient of the unrestricted log-likelihood
    evaluated at the restricted parameter values. Compute it as
    jax.grad(unrestricted_ll_fn)(restricted_params).

    The information matrix is the negative Hessian of the restricted model,
    available as -restricted_summary.hessian.

    Args:
        score: Score vector S(theta_R) of shape (K,). The gradient of
            the unrestricted log-likelihood at the restricted estimates.
        information_matrix: Information matrix I(theta_R) of shape (K, K).
            Typically -H where H is the Hessian of the restricted model.
        df: Degrees of freedom (number of restrictions being tested).

    Returns:
        Dict with keys: statistic, df, p_value.
    """
    score = jnp.asarray(score)
    information_matrix = jnp.asarray(information_matrix)

    # LM = S' I^{-1} S
    I_inv = jnp.linalg.inv(information_matrix)
    statistic = float(score @ I_inv @ score)

    p_value = 1.0 - stats.chi2.cdf(statistic, df)

    return {"statistic": statistic, "df": df, "p_value": float(p_value)}


def vuong_test(
    policy_1: jnp.ndarray,
    policy_2: jnp.ndarray,
    obs_states: jnp.ndarray,
    obs_actions: jnp.ndarray,
    num_params_1: int | None = None,
    num_params_2: int | None = None,
) -> dict[str, float | str]:
    """Vuong test for non-nested model comparison.

    Tests H0 (both models are equally close to the true DGP) against
    H1 (one model is closer). The test statistic is asymptotically
    standard normal under H0.

    Per-observation log-likelihoods are computed from the policy matrices
    as log pi(a_i | s_i) for each model.

    When num_params_1 and num_params_2 are provided, applies the Schwarz
    (BIC-type) correction for different model complexity.

    Args:
        policy_1: Choice probabilities from model 1, shape (num_states, num_actions).
        policy_2: Choice probabilities from model 2, shape (num_states, num_actions).
        obs_states: Observed states, shape (N,), dtype int.
        obs_actions: Observed actions, shape (N,), dtype int.
        num_params_1: Number of parameters in model 1 (for Schwarz correction).
        num_params_2: Number of parameters in model 2 (for Schwarz correction).

    Returns:
        Dict with keys: statistic, p_value, direction.
        direction is 'model_1' if model 1 fits better, 'model_2' if model 2
        fits better, or 'indistinguishable' if the test cannot distinguish them
        at the 5 percent level.
        If Schwarz correction is applied, also includes corrected_statistic
        and corrected_p_value.
    """
    obs_states = jnp.asarray(obs_states, dtype=jnp.int32)
    obs_actions = jnp.asarray(obs_actions, dtype=jnp.int32)

    eps = 1e-15
    log_pi_1 = jnp.log(jnp.clip(policy_1[obs_states, obs_actions], eps, 1.0))
    log_pi_2 = jnp.log(jnp.clip(policy_2[obs_states, obs_actions], eps, 1.0))

    m = log_pi_1 - log_pi_2
    n = len(m)
    omega = float(jnp.std(m, ddof=1))

    if omega < 1e-12:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "direction": "indistinguishable",
        }

    z = float(jnp.sum(m)) / (np.sqrt(n) * omega)
    p_value = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

    if abs(z) < 1.96:
        direction = "indistinguishable"
    elif z > 0:
        direction = "model_1"
    else:
        direction = "model_2"

    result: dict[str, float | str] = {
        "statistic": z,
        "p_value": p_value,
        "direction": direction,
    }

    # Schwarz (BIC-type) correction for different complexity
    if num_params_1 is not None and num_params_2 is not None:
        correction = (num_params_1 - num_params_2) * np.log(n) / (
            2.0 * np.sqrt(n) * omega
        )
        z_corrected = z - correction
        p_corrected = 2.0 * (1.0 - stats.norm.cdf(abs(z_corrected)))
        result["corrected_statistic"] = z_corrected
        result["corrected_p_value"] = p_corrected

    return result
