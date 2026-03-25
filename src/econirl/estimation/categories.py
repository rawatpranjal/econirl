"""Estimator taxonomy and problem-space categorization.

Provides a principled decomposition of estimators along two orthogonal
dimensions: estimation paradigm (category) and structural assumptions
(capabilities). Inspired by EconML's class hierarchy and scikit-learn's
tags system.

Usage:
    >>> from econirl.estimation.categories import (
    ...     EstimatorCategory, get_estimators_by_category, get_estimators_with_capability
    ... )
    >>> get_estimators_by_category(EstimatorCategory.ADVERSARIAL_IRL)
    ['GAIL', 'AIRL', 'GCL']
    >>> get_estimators_with_capability(has_inner_bellman_solve=False)
    ['CCP', 'SEES', 'IQ-Learn', 'GLADIUS', 'BC']
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal


class EstimatorCategory(str, Enum):
    """Estimation paradigm — what kind of method is this?"""

    STRUCTURAL = "structural"
    """Forward MLE with parametric reward. Solves the Bellman equation
    exactly (NFXP) or via CCP inversion (CCP). The econometrics approach."""

    STRUCTURAL_APPROX = "structural_approx"
    """Structural MLE with approximate value function. Uses neural networks
    (NNES, TD-CCP) or sieve bases (SEES) to approximate V(s), enabling
    scalability to large state spaces."""

    ENTROPY_IRL = "entropy_irl"
    """Maximum entropy / maximum causal entropy IRL. Recovers reward by
    matching feature expectations under the maximum entropy principle."""

    MARGIN_IRL = "margin_irl"
    """Margin-based IRL. Recovers reward that makes expert behavior optimal
    with maximum margin over alternatives."""

    ADVERSARIAL_IRL = "adversarial_irl"
    """Adversarial IRL. Uses a discriminator to distinguish expert from
    generated behavior; reward emerges from the discriminator."""

    BAYESIAN_IRL = "bayesian_irl"
    """Bayesian IRL. Places a prior over reward parameters and computes
    the posterior via MCMC sampling."""

    DISTRIBUTION_IRL = "distribution_irl"
    """Distribution-matching IRL. Minimizes statistical divergence between
    expert and policy state-action distributions."""

    Q_LEARNING_IRL = "q_learning_irl"
    """Q-function-based IRL. Learns Q-values directly and recovers reward
    via the inverse Bellman operator. Avoids inner-loop policy optimization."""

    IMITATION = "imitation"
    """Direct imitation. Matches expert behavior without reward modeling
    or dynamic programming."""


@dataclass(frozen=True)
class ProblemCapabilities:
    """Structural assumptions and capabilities of an estimator.

    Each field represents an axis of the problem space that determines
    which estimators are applicable.

    Attributes:
        reward_type: Form of reward function the estimator uses.
            "linear" = R(s,a) = θ·φ(s,a), parametric with features.
            "tabular" = R(s,a) free matrix, no parametric structure.
            "neural" = R(s,a) from a neural network.
            "none" = no reward modeling.
        requires_transitions: Whether known P(s'|s,a) is needed.
        recovers_structural_params: Whether interpretable θ is returned.
        recovers_reward: Whether an R(s,a) matrix is produced.
        has_inner_bellman_solve: Whether a full Bellman solve runs inside.
        supports_finite_horizon: Whether finite-horizon (T < ∞) works.
        supports_continuous_states: Whether function approximation enables
            scalability to large/continuous state spaces.
    """

    reward_type: Literal["linear", "tabular", "neural", "none"]
    requires_transitions: bool
    recovers_structural_params: bool
    recovers_reward: bool
    has_inner_bellman_solve: bool
    supports_finite_horizon: bool
    supports_continuous_states: bool


# ---------------------------------------------------------------------------
# Registry: maps estimator benchmark name -> (category, capabilities)
# ---------------------------------------------------------------------------

ESTIMATOR_REGISTRY: dict[str, tuple[EstimatorCategory, ProblemCapabilities]] = {
    "NFXP": (
        EstimatorCategory.STRUCTURAL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=True,
            supports_continuous_states=False,
        ),
    ),
    "CCP": (
        EstimatorCategory.STRUCTURAL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "SEES": (
        EstimatorCategory.STRUCTURAL_APPROX,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=True,
        ),
    ),
    "NNES": (
        EstimatorCategory.STRUCTURAL_APPROX,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=True,
        ),
    ),
    "TD-CCP": (
        EstimatorCategory.STRUCTURAL_APPROX,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=True,
        ),
    ),
    "MCE IRL": (
        EstimatorCategory.ENTROPY_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=True,
            supports_continuous_states=False,
        ),
    ),
    "MaxEnt IRL": (
        EstimatorCategory.ENTROPY_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "Deep MaxEnt": (
        EstimatorCategory.ENTROPY_IRL,
        ProblemCapabilities(
            reward_type="neural",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "Max Margin": (
        EstimatorCategory.MARGIN_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "Max Margin IRL": (
        EstimatorCategory.MARGIN_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "GAIL": (
        EstimatorCategory.ADVERSARIAL_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=False,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "AIRL": (
        EstimatorCategory.ADVERSARIAL_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=True,
            supports_continuous_states=False,
        ),
    ),
    "GCL": (
        EstimatorCategory.ADVERSARIAL_IRL,
        ProblemCapabilities(
            reward_type="neural",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "BIRL": (
        EstimatorCategory.BAYESIAN_IRL,
        ProblemCapabilities(
            reward_type="linear",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "f-IRL": (
        EstimatorCategory.DISTRIBUTION_IRL,
        ProblemCapabilities(
            reward_type="tabular",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=True,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "IQ-Learn": (
        EstimatorCategory.Q_LEARNING_IRL,
        ProblemCapabilities(
            reward_type="tabular",
            requires_transitions=True,
            recovers_structural_params=False,
            recovers_reward=True,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
    "GLADIUS": (
        EstimatorCategory.Q_LEARNING_IRL,
        ProblemCapabilities(
            reward_type="neural",
            requires_transitions=True,
            recovers_structural_params=True,
            recovers_reward=True,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=True,
        ),
    ),
    "BC": (
        EstimatorCategory.IMITATION,
        ProblemCapabilities(
            reward_type="none",
            requires_transitions=False,
            recovers_structural_params=False,
            recovers_reward=False,
            has_inner_bellman_solve=False,
            supports_finite_horizon=False,
            supports_continuous_states=False,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_estimators_by_category(category: EstimatorCategory) -> list[str]:
    """Return estimator names belonging to a category.

    >>> get_estimators_by_category(EstimatorCategory.ADVERSARIAL_IRL)
    ['GAIL', 'AIRL', 'GCL']
    """
    return [name for name, (cat, _) in ESTIMATOR_REGISTRY.items() if cat == category]


def get_estimators_with_capability(**kwargs) -> list[str]:
    """Return estimator names matching all given capability filters.

    >>> get_estimators_with_capability(has_inner_bellman_solve=False)
    ['CCP', 'SEES', 'NNES', 'TD-CCP', 'IQ-Learn', 'GLADIUS', 'BC']
    >>> get_estimators_with_capability(recovers_structural_params=True, supports_continuous_states=True)
    ['SEES', 'NNES', 'TD-CCP', 'GLADIUS']
    """
    results = []
    for name, (_, caps) in ESTIMATOR_REGISTRY.items():
        if all(getattr(caps, k) == v for k, v in kwargs.items()):
            results.append(name)
    return results


def get_category(estimator_name: str) -> EstimatorCategory:
    """Get the category of an estimator by name."""
    return ESTIMATOR_REGISTRY[estimator_name][0]


def get_capabilities(estimator_name: str) -> ProblemCapabilities:
    """Get the capabilities of an estimator by name."""
    return ESTIMATOR_REGISTRY[estimator_name][1]
