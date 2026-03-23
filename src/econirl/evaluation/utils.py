"""Shared utilities for evaluation metrics."""

from __future__ import annotations

import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem


def compute_policy(
    theta: torch.Tensor,
    problem: DDCProblem,
    transitions: torch.Tensor,
    feature_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute optimal policy given parameters and environment.

    Args:
        theta: Parameter vector
        problem: DDC problem specification
        transitions: Transition tensor (n_actions, n_states, n_states)
        feature_matrix: Feature tensor (n_states, n_actions, n_features)

    Returns:
        Policy tensor of shape (n_states, n_actions)
    """
    # Compute utility matrix
    utility = torch.einsum("sak,k->sa", feature_matrix, theta)

    # Create Bellman operator and solve
    operator = SoftBellmanOperator(problem=problem, transitions=transitions)
    result = value_iteration(operator, utility)

    return result.policy
