"""Soft Bellman operator for logit discrete choice models.

This module implements the contraction mapping at the heart of dynamic
discrete choice estimation. The soft (logit) Bellman operator accounts
for the extreme value distribution of preference shocks.

Key equations (following Rust 1987):
- Q(s,a) = u(s,a) + β Σ_{s'} P(s'|s,a) V(s')
- V(s) = σ log(Σ_a exp(Q(s,a)/σ))  [log-sum-exp / social surplus]

The operator Λ_σ is a contraction with modulus β, guaranteeing
convergence of value iteration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from econirl.core.types import DDCProblem


class BellmanResult(NamedTuple):
    """Result of applying the Bellman operator.

    Attributes:
        Q: Action-value function, shape (num_states, num_actions)
        V: Value function, shape (num_states,)
        policy: Choice probabilities, shape (num_states, num_actions)
    """

    Q: torch.Tensor
    V: torch.Tensor
    policy: torch.Tensor


@dataclass
class SoftBellmanOperator:
    """Soft Bellman operator for logit discrete choice models.

    Implements the fixed-point operator for solving the dynamic programming
    problem with extreme value (Type I) preference shocks. This is the
    inner loop of NFXP estimation.

    The operator computes:
        Q(s,a) = u(s,a) + β Σ_{s'} P(s'|s,a) V(s')
        V(s) = σ log(Σ_a exp(Q(s,a)/σ))

    where σ is the scale parameter of the extreme value distribution.

    Attributes:
        problem: DDCProblem specification
        transitions: Transition probability matrices, shape (num_actions, num_states, num_states)
                    where transitions[a, s, s'] = P(s' | s, a)

    Example:
        >>> operator = SoftBellmanOperator(problem, transitions)
        >>> result = operator.solve(utility_matrix)
        >>> print(result.policy)  # Optimal choice probabilities
    """

    problem: DDCProblem
    transitions: torch.Tensor

    def __post_init__(self) -> None:
        expected_shape = (
            self.problem.num_actions,
            self.problem.num_states,
            self.problem.num_states,
        )
        if self.transitions.shape != expected_shape:
            raise ValueError(
                f"transitions must have shape {expected_shape}, "
                f"got {self.transitions.shape}"
            )

    def apply(self, utility: torch.Tensor, V: torch.Tensor) -> BellmanResult:
        """Apply the Bellman operator once.

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Current value function, shape (num_states,)

        Returns:
            BellmanResult with updated Q, V, and policy
        """
        beta = self.problem.discount_factor
        sigma = self.problem.scale_parameter

        # Q(s,a) = u(s,a) + β Σ_{s'} P(s'|s,a) V(s')
        # transitions[a, s, t] @ V[t] gives expected continuation value
        # where t indexes the next state s'
        # Result shape: (num_actions, num_states)
        EV = torch.einsum("ast,t->as", self.transitions, V)

        # Q shape: (num_states, num_actions)
        Q = utility + beta * EV.T

        # V(s) = σ log(Σ_a exp(Q(s,a)/σ)) using log-sum-exp for stability
        V_new = sigma * torch.logsumexp(Q / sigma, dim=1)

        # Choice probabilities via softmax
        policy = F.softmax(Q / sigma, dim=1)

        return BellmanResult(Q=Q, V=V_new, policy=policy)

    def compute_expected_value(self, V: torch.Tensor) -> torch.Tensor:
        """Compute expected continuation value E[V(s') | s, a].

        Args:
            V: Value function, shape (num_states,)

        Returns:
            Expected values, shape (num_states, num_actions)
        """
        # transitions[a, s, t] @ V[t] -> (num_actions, num_states)
        # where t indexes the next state s'
        EV = torch.einsum("ast,t->as", self.transitions, V)
        return EV.T  # (num_states, num_actions)

    def compute_choice_probabilities(
        self, utility: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Compute choice probabilities given utility and value function.

        This implements the logit choice rule:
            P(a|s) = exp(Q(s,a)/σ) / Σ_{a'} exp(Q(s,a')/σ)

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Value function, shape (num_states,)

        Returns:
            Choice probabilities, shape (num_states, num_actions)
        """
        result = self.apply(utility, V)
        return result.policy

    def compute_log_choice_probabilities(
        self, utility: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """Compute log choice probabilities (numerically stable).

        Args:
            utility: Flow utility matrix, shape (num_states, num_actions)
            V: Value function, shape (num_states,)

        Returns:
            Log choice probabilities, shape (num_states, num_actions)
        """
        beta = self.problem.discount_factor
        sigma = self.problem.scale_parameter

        EV = self.compute_expected_value(V)
        Q = utility + beta * EV

        # log P(a|s) = Q(s,a)/σ - log(Σ_{a'} exp(Q(s,a')/σ))
        return F.log_softmax(Q / sigma, dim=1)


def compute_flow_utility(
    utility_params: torch.Tensor,
    feature_matrix: torch.Tensor,
) -> torch.Tensor:
    """Compute flow utility from parameters and features.

    Implements the linear utility specification:
        u(s,a) = θ · φ(s,a)

    where θ are the utility parameters and φ(s,a) are the features.

    Args:
        utility_params: Parameter vector, shape (num_params,)
        feature_matrix: Feature tensor, shape (num_states, num_actions, num_params)

    Returns:
        Flow utility matrix, shape (num_states, num_actions)
    """
    return torch.einsum("sak,k->sa", feature_matrix, utility_params)


def compute_social_surplus(Q: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """Compute the social surplus (expected max utility) at each state.

    The social surplus is the expected value of the maximum utility
    across actions, accounting for the preference shocks:
        W(s) = E[max_a {Q(s,a) + σε_a}] = σ log(Σ_a exp(Q(s,a)/σ))

    This is also known as the "inclusive value" in discrete choice.

    Args:
        Q: Action-value function, shape (num_states, num_actions)
        sigma: Scale parameter of extreme value distribution

    Returns:
        Social surplus, shape (num_states,)
    """
    return sigma * torch.logsumexp(Q / sigma, dim=1)
