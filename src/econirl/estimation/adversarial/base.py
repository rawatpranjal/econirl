"""Base class for adversarial imitation learning methods.

This module provides shared functionality for GAIL, AIRL, and similar
adversarial methods.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Literal

import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration, value_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.base import BaseEstimator


class AdversarialEstimatorBase(BaseEstimator):
    """Base class for adversarial imitation learning estimators.

    Provides shared utilities for sampling, policy computation, and
    initial state distribution estimation.
    """

    def _sample_from_panel(
        self,
        panel: Panel,
        batch_size: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample state-action pairs from expert demonstrations."""
        states = panel.get_all_states()
        actions = panel.get_all_actions()

        if batch_size is not None and batch_size > 0 and batch_size < len(states):
            indices = torch.randperm(len(states))[:batch_size]
            return states[indices], actions[indices]

        return states, actions

    def _sample_transitions_from_panel(
        self,
        panel: Panel,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (s, a, s') transitions from expert demonstrations."""
        return (
            panel.get_all_states(),
            panel.get_all_actions(),
            panel.get_all_next_states(),
        )

    def _sample_from_policy(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        n_samples: int,
        initial_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample state-action pairs from current policy."""
        n_states, n_actions = policy.shape
        states = []
        actions = []

        state = torch.multinomial(initial_dist, 1).item()

        for _ in range(n_samples):
            action = torch.multinomial(policy[state], 1).item()
            states.append(state)
            actions.append(action)

            next_state_dist = transitions[action, state, :]
            state = torch.multinomial(next_state_dist, 1).item()

        return torch.tensor(states, dtype=torch.long), torch.tensor(
            actions, dtype=torch.long
        )

    def _sample_transitions_from_policy(
        self,
        policy: torch.Tensor,
        transitions: torch.Tensor,
        n_samples: int,
        initial_dist: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (s, a, s') transitions from current policy."""
        n_states, n_actions = policy.shape
        states = []
        actions = []
        next_states_list = []

        state = torch.multinomial(initial_dist, 1).item()

        for _ in range(n_samples):
            action = torch.multinomial(policy[state], 1).item()
            next_state_dist = transitions[action, state, :]
            next_state = torch.multinomial(next_state_dist, 1).item()

            states.append(state)
            actions.append(action)
            next_states_list.append(next_state)

            state = next_state

        return (
            torch.tensor(states, dtype=torch.long),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(next_states_list, dtype=torch.long),
        )

    def _compute_initial_distribution(
        self,
        panel: Panel,
        n_states: int,
    ) -> torch.Tensor:
        """Compute initial state distribution from data."""
        counts = torch.zeros(n_states, dtype=torch.float32)
        init_states = torch.tensor(
            [traj.states[0].item() for traj in panel.trajectories if len(traj) > 0],
            dtype=torch.long,
        )
        counts.scatter_add_(0, init_states, torch.ones_like(init_states, dtype=torch.float32))

        if counts.sum() > 0:
            return counts / counts.sum()
        return torch.ones(n_states) / n_states

    def _compute_policy(
        self,
        reward_matrix: torch.Tensor,
        operator: SoftBellmanOperator,
        solver: Literal["value", "hybrid"] = "hybrid",
        tol: float = 1e-8,
        max_iter: int = 5000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal policy given reward matrix."""
        if solver == "hybrid":
            result = hybrid_iteration(operator, reward_matrix, tol=tol, max_iter=max_iter)
        else:
            result = value_iteration(operator, reward_matrix, tol=tol, max_iter=max_iter)
        return result.policy, result.V
