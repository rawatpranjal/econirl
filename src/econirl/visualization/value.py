"""Visualization of value functions and Q-values.

This module provides plotting functions for visualizing value functions,
action values (Q-functions), and related quantities from dynamic
discrete choice models.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.core.types import DDCProblem
from econirl.inference.results import EstimationSummary
from econirl.preferences.base import UtilityFunction


def plot_value_function(
    result: EstimationSummary,
    xlabel: str = "State",
    ylabel: str = "Value",
    title: str = "Value Function V(s)",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the estimated value function.

    Args:
        result: Estimation result with value function
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    V = result.value_function
    num_states = len(V)
    states = np.arange(num_states)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(states, V.numpy(), linewidth=2, color="blue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig


def plot_q_values(
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
    parameters: torch.Tensor,
    action_labels: list[str] | None = None,
    xlabel: str = "State",
    ylabel: str = "Q-Value",
    title: str = "Action Values Q(s, a)",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot Q-values for each action.

    Shows Q(s, a) = U(s, a) + β E[V(s') | s, a] for each action.

    Args:
        utility: Utility specification
        problem: Problem specification
        transitions: Transition matrices
        parameters: Parameter values
        action_labels: Labels for actions
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    # Compute value function and Q-values
    operator = SoftBellmanOperator(problem, transitions)
    flow_utility = utility.compute(parameters)
    result = value_iteration(operator, flow_utility)

    Q = result.Q
    num_states, num_actions = Q.shape
    states = np.arange(num_states)

    if action_labels is None:
        action_labels = [f"Action {a}" for a in range(num_actions)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for a in range(num_actions):
        ax.plot(states, Q[:, a].numpy(), label=action_labels[a], linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_value_decomposition(
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
    parameters: torch.Tensor,
    action_idx: int = 0,
    action_label: str | None = None,
    xlabel: str = "State",
    title: str = "Value Decomposition",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Decompose Q-value into flow utility and continuation value.

    Shows Q(s,a) = U(s,a) + β E[V(s') | s,a] with both components.

    Args:
        utility: Utility specification
        problem: Problem specification
        transitions: Transition matrices
        parameters: Parameter values
        action_idx: Which action to decompose
        action_label: Label for action
        xlabel: X-axis label
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    operator = SoftBellmanOperator(problem, transitions)
    flow_utility = utility.compute(parameters)
    result = value_iteration(operator, flow_utility)

    num_states = problem.num_states
    states = np.arange(num_states)

    if action_label is None:
        action_label = f"Action {action_idx}"

    # Components
    U = flow_utility[:, action_idx]
    EV = operator.compute_expected_value(result.V)[:, action_idx]
    continuation = problem.discount_factor * EV
    Q = result.Q[:, action_idx]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(states, U.numpy(), label="Flow Utility U(s,a)", linewidth=2)
    ax.plot(states, continuation.numpy(), label=f"β E[V(s')|s,a]", linewidth=2)
    ax.plot(states, Q.numpy(), label="Q(s,a) = U + βEV", linewidth=2, linestyle="--")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title(f"{title}: {action_label}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_value_comparison(
    values: list[torch.Tensor],
    labels: list[str],
    xlabel: str = "State",
    ylabel: str = "Value",
    title: str = "Value Function Comparison",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Compare multiple value functions.

    Useful for comparing baseline vs counterfactual or
    different estimation methods.

    Args:
        values: List of value function tensors
        labels: Labels for each value function
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    if len(values) != len(labels):
        raise ValueError("Number of values must match number of labels")

    num_states = len(values[0])
    states = np.arange(num_states)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for V, label in zip(values, labels):
        ax.plot(states, V.numpy(), label=label, linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_advantage_function(
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
    parameters: torch.Tensor,
    action_labels: list[str] | None = None,
    xlabel: str = "State",
    ylabel: str = "Advantage",
    title: str = "Advantage Function A(s, a) = Q(s, a) - V(s)",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot the advantage function A(s,a) = Q(s,a) - V(s).

    The advantage shows the relative value of each action compared
    to the expected value under the optimal policy.

    Args:
        utility: Utility specification
        problem: Problem specification
        transitions: Transition matrices
        parameters: Parameter values
        action_labels: Labels for actions
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    operator = SoftBellmanOperator(problem, transitions)
    flow_utility = utility.compute(parameters)
    result = value_iteration(operator, flow_utility)

    # Advantage = Q - V
    V_expanded = result.V.unsqueeze(1)  # (num_states, 1)
    advantage = result.Q - V_expanded  # (num_states, num_actions)

    num_states, num_actions = advantage.shape
    states = np.arange(num_states)

    if action_labels is None:
        action_labels = [f"Action {a}" for a in range(num_actions)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for a in range(num_actions):
        ax.plot(states, advantage[:, a].numpy(), label=action_labels[a], linewidth=2)

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_flow_utility(
    utility: UtilityFunction,
    parameters: torch.Tensor,
    action_labels: list[str] | None = None,
    xlabel: str = "State",
    ylabel: str = "Flow Utility",
    title: str = "Flow Utility U(s, a)",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot flow utility for each action (without continuation value).

    Args:
        utility: Utility specification
        parameters: Parameter values
        action_labels: Labels for actions
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    U = utility.compute(parameters)
    num_states, num_actions = U.shape
    states = np.arange(num_states)

    if action_labels is None:
        action_labels = [f"Action {a}" for a in range(num_actions)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for a in range(num_actions):
        ax.plot(states, U[:, a].numpy(), label=action_labels[a], linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def plot_value_iteration_convergence(
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
    parameters: torch.Tensor,
    max_iter: int = 100,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot convergence of value iteration.

    Shows how the value function converges over iterations.

    Args:
        utility: Utility specification
        problem: Problem specification
        transitions: Transition matrices
        parameters: Parameter values
        max_iter: Maximum iterations to run
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    operator = SoftBellmanOperator(problem, transitions)
    flow_utility = utility.compute(parameters)

    # Track convergence
    V = torch.zeros(problem.num_states)
    errors = []

    for i in range(max_iter):
        result = operator.apply(flow_utility, V)
        error = torch.abs(result.V - V).max().item()
        errors.append(error)

        if error < 1e-12:
            break
        V = result.V

    fig, ax = plt.subplots(figsize=figsize)

    ax.semilogy(range(len(errors)), errors, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Max |V_{k+1} - V_k| (log scale)")
    ax.set_title("Value Iteration Convergence")
    ax.grid(True, alpha=0.3)

    return fig


def create_value_summary_figure(
    result: EstimationSummary,
    utility: UtilityFunction,
    problem: DDCProblem,
    transitions: torch.Tensor,
    action_labels: list[str] | None = None,
    figsize: tuple[float, float] = (14, 10),
) -> plt.Figure:
    """Create a comprehensive summary figure with multiple value plots.

    Includes:
    - Value function V(s)
    - Q-values for each action
    - Advantage function
    - Flow utility

    Args:
        result: Estimation result
        utility: Utility specification
        problem: Problem specification
        transitions: Transition matrices
        action_labels: Labels for actions
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Value function
    plot_value_function(result, ax=axes[0, 0])

    # Q-values
    plot_q_values(
        utility, problem, transitions, result.parameters,
        action_labels=action_labels, ax=axes[0, 1]
    )

    # Advantage
    plot_advantage_function(
        utility, problem, transitions, result.parameters,
        action_labels=action_labels, ax=axes[1, 0]
    )

    # Flow utility
    plot_flow_utility(
        utility, result.parameters,
        action_labels=action_labels, ax=axes[1, 1]
    )

    plt.tight_layout()
    return fig
