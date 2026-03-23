"""Visualization of choice probabilities and policies.

This module provides plotting functions for visualizing estimated
choice probabilities (CCPs) and comparing policies across different
scenarios (baseline vs counterfactual).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from econirl.core.types import Panel
from econirl.inference.results import EstimationSummary
from econirl.simulation.counterfactual import CounterfactualResult


def plot_choice_probabilities(
    result: EstimationSummary,
    action_labels: list[str] | None = None,
    state_labels: list[str] | None = None,
    xlabel: str = "State",
    ylabel: str = "Choice Probability",
    title: str = "Conditional Choice Probabilities",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot estimated choice probabilities by state.

    Creates a line plot showing P(action | state) for each action.

    Args:
        result: Estimation result with policy
        action_labels: Labels for each action (default: Action 0, 1, ...)
        state_labels: Labels for x-axis states (default: 0, 1, 2, ...)
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes to plot on (optional)

    Returns:
        matplotlib Figure

    Example:
        >>> fig = plot_choice_probabilities(result, action_labels=["Keep", "Replace"])
        >>> fig.savefig("ccps.png")
    """
    policy = result.policy
    num_states, num_actions = policy.shape

    if action_labels is None:
        action_labels = [f"Action {a}" for a in range(num_actions)]

    states = np.arange(num_states)
    if state_labels is None:
        state_labels = states

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    for a in range(num_actions):
        ax.plot(states, policy[:, a].numpy(), label=action_labels[a], linewidth=2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig


def plot_policy_comparison(
    baseline_policy: torch.Tensor,
    counterfactual_policy: torch.Tensor,
    action_idx: int = 0,
    baseline_label: str = "Baseline",
    counterfactual_label: str = "Counterfactual",
    action_label: str | None = None,
    xlabel: str = "State",
    ylabel: str = "Choice Probability",
    title: str = "Policy Comparison",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Compare two policies side by side.

    Useful for counterfactual analysis to show how policy changes.

    Args:
        baseline_policy: Original policy
        counterfactual_policy: New policy
        action_idx: Which action to plot
        baseline_label: Label for baseline policy
        counterfactual_label: Label for counterfactual
        action_label: Label for the action being plotted
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    num_states = baseline_policy.shape[0]
    states = np.arange(num_states)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if action_label is None:
        action_label = f"Action {action_idx}"

    ax.plot(
        states,
        baseline_policy[:, action_idx].numpy(),
        label=baseline_label,
        linewidth=2,
        linestyle="-",
    )
    ax.plot(
        states,
        counterfactual_policy[:, action_idx].numpy(),
        label=counterfactual_label,
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: P({action_label} | state)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig


def plot_counterfactual_comparison(
    counterfactual: CounterfactualResult,
    action_labels: list[str] | None = None,
    xlabel: str = "State",
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """Create a multi-panel comparison of baseline vs counterfactual.

    Shows:
    - Top row: Policy for each action
    - Bottom row: Value functions and policy change

    Args:
        counterfactual: Counterfactual result
        action_labels: Labels for actions
        xlabel: X-axis label
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    num_actions = counterfactual.baseline_policy.shape[1]
    num_states = counterfactual.baseline_policy.shape[0]
    states = np.arange(num_states)

    if action_labels is None:
        action_labels = [f"Action {a}" for a in range(num_actions)]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Top left: Baseline policy
    ax = axes[0, 0]
    for a in range(num_actions):
        ax.plot(states, counterfactual.baseline_policy[:, a].numpy(),
                label=action_labels[a], linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_title("Baseline Policy")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Top right: Counterfactual policy
    ax = axes[0, 1]
    for a in range(num_actions):
        ax.plot(states, counterfactual.counterfactual_policy[:, a].numpy(),
                label=action_labels[a], linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_title("Counterfactual Policy")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Bottom left: Value functions
    ax = axes[1, 0]
    ax.plot(states, counterfactual.baseline_value.numpy(),
            label="Baseline", linewidth=2)
    ax.plot(states, counterfactual.counterfactual_value.numpy(),
            label="Counterfactual", linewidth=2, linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title("Value Functions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Policy change
    ax = axes[1, 1]
    for a in range(num_actions):
        ax.plot(states, counterfactual.policy_change[:, a].numpy(),
                label=action_labels[a], linewidth=2)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Change in Probability")
    ax.set_title("Policy Change (CF - Baseline)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_empirical_vs_predicted(
    panel: Panel,
    result: EstimationSummary,
    action_idx: int = 0,
    action_label: str | None = None,
    xlabel: str = "State",
    ylabel: str = "Choice Probability",
    title: str = "Model Fit: Empirical vs Predicted",
    figsize: tuple[float, float] = (10, 6),
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Compare empirical choice frequencies to model predictions.

    This is a key diagnostic for model fit.

    Args:
        panel: Observed data
        result: Estimation result
        action_idx: Which action to plot
        action_label: Label for action
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        ax: Existing axes

    Returns:
        matplotlib Figure
    """
    num_states = result.policy.shape[0]
    num_actions = result.policy.shape[1]

    # Compute empirical CCPs
    empirical = panel.compute_choice_frequencies(num_states, num_actions)

    # Predicted
    predicted = result.policy

    states = np.arange(num_states)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if action_label is None:
        action_label = f"Action {action_idx}"

    ax.scatter(
        states,
        empirical[:, action_idx].numpy(),
        label="Empirical",
        alpha=0.6,
        s=30,
    )
    ax.plot(
        states,
        predicted[:, action_idx].numpy(),
        label="Predicted",
        linewidth=2,
        color="red",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}: P({action_label} | state)")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig


def plot_policy_heatmap(
    policy: torch.Tensor,
    action_labels: list[str] | None = None,
    xlabel: str = "Action",
    ylabel: str = "State",
    title: str = "Choice Probabilities",
    figsize: tuple[float, float] = (8, 10),
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot policy as a heatmap.

    Useful when there are many actions or for dense visualization.

    Args:
        policy: Policy tensor (num_states, num_actions)
        action_labels: Labels for actions
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        figsize: Figure size
        cmap: Colormap

    Returns:
        matplotlib Figure
    """
    num_states, num_actions = policy.shape

    if action_labels is None:
        action_labels = [f"A{a}" for a in range(num_actions)]

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(policy.numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(num_actions))
    ax.set_xticklabels(action_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    return fig


def plot_action_threshold(
    result: EstimationSummary,
    threshold_prob: float = 0.5,
    action_idx: int = 1,
    action_label: str | None = None,
    xlabel: str = "State",
    title: str = "Action Threshold Analysis",
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Plot where action probability crosses a threshold.

    Useful for identifying "switching points" in the state space.

    Args:
        result: Estimation result
        threshold_prob: Probability threshold to mark
        action_idx: Which action to analyze
        action_label: Label for action
        xlabel: X-axis label
        title: Plot title
        figsize: Figure size

    Returns:
        matplotlib Figure
    """
    policy = result.policy
    num_states = policy.shape[0]
    states = np.arange(num_states)

    if action_label is None:
        action_label = f"Action {action_idx}"

    fig, ax = plt.subplots(figsize=figsize)

    probs = policy[:, action_idx].numpy()
    ax.plot(states, probs, linewidth=2, label=f"P({action_label})")

    # Mark threshold
    ax.axhline(y=threshold_prob, color="red", linestyle="--",
               label=f"Threshold = {threshold_prob}")

    # Find crossing points
    above = probs >= threshold_prob
    crossings = np.where(np.diff(above.astype(int)) != 0)[0]

    for cross in crossings:
        ax.axvline(x=cross, color="green", linestyle=":", alpha=0.7)
        ax.annotate(
            f"s={cross}",
            (cross, threshold_prob),
            textcoords="offset points",
            xytext=(5, 5),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    return fig
