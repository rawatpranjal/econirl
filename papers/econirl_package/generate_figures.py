#!/usr/bin/env python3
"""Generate all figures for the EconIRL package paper.

Reads existing benchmark results from the repo and produces three PDF figures:
  fig/fig1_tabular_convergence.pdf  — Case Study 1: cosine similarity bar chart
  fig/fig2_scaling.pdf              — Case Study 2: SEES basis sweep + timing
  fig/fig3_nonlinear.pdf            — Case Study 3: policy accuracy comparison

Usage:
    python papers/econirl_package/generate_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FIG_DIR = Path(__file__).resolve().parent / "fig"
FIG_DIR.mkdir(exist_ok=True)

# Use a clean style
plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def fig1_tabular_convergence():
    """Case Study 1: Rust bus benchmark — cosine similarity vs speed."""

    # Data from examples/rust-bus-engine/benchmark_results.csv
    estimators = [
        # name, cosine_sim, time_seconds, group
        ("BC", 0.0, 0.01, "Baseline"),
        ("IQ-Learn", 0.9993, 0.01, "IRL"),
        ("AIRL", 1.0000, 607.7, "IRL"),
        ("GLADIUS", 1.0000, 10.4, "Neural"),
        ("SEES", 1.0000, 1.5, "Neural"),
        ("TD-CCP", 1.0000, 129.6, "Neural"),
        ("NNES", 0.9999, 16.0, "Neural"),
        ("MCE-IRL", 1.0000, 0.1, "Classical"),
        ("CCP-NPL", 1.0000, 0.1, "Classical"),
        ("NFXP-NK", 1.0000, 0.1, "Classical"),
    ]

    names = [e[0] for e in estimators]
    cosines = [e[1] for e in estimators]
    times = [e[2] for e in estimators]
    groups = [e[3] for e in estimators]

    # Color by group
    group_colors = {
        "Classical": "#2196F3",
        "Neural": "#4CAF50",
        "IRL": "#FF9800",
        "Baseline": "#9E9E9E",
    }
    colors = [group_colors[g] for g in groups]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, cosines, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Cosine Similarity to True Parameters")
    ax.set_xlim(0.0, 1.05)
    ax.set_title("(a) Parameter Recovery on Rust Bus Benchmark")

    # Annotate with time
    for i, (c, t) in enumerate(zip(cosines, times)):
        if c > 0:
            label = f"{t:.0f}s" if t >= 1 else f"<1s"
            ax.text(c + 0.01, i, label, va="center", fontsize=7, color="gray")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=group_colors["Classical"], label="Classical Structural"),
        Patch(facecolor=group_colors["Neural"], label="Neural Structural"),
        Patch(facecolor=group_colors["IRL"], label="Inverse RL"),
        Patch(facecolor=group_colors["Baseline"], label="Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=7)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig1_tabular_convergence.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved fig1_tabular_convergence.pdf")


def fig2_scaling():
    """Case Study 2: SEES basis sweep + timing comparison."""

    # Data from examples/multi-component-bus/sees_results.json
    sees_path = ROOT / "examples" / "multi-component-bus" / "sees_results.json"
    if sees_path.exists():
        with open(sees_path) as f:
            data = json.load(f)
        basis_dims = []
        rmses = []
        times_sees = []
        for k in ["4", "6", "8", "12", "16", "20"]:
            if k in data.get("basis_sweep", {}):
                basis_dims.append(int(k))
                rmses.append(data["basis_sweep"][k]["rmse_vs_nfxp"])
                times_sees.append(data["basis_sweep"][k]["time"])
        nfxp_time = data.get("nfxp", {}).get("time", 7.6)
        nnes_time = data.get("nnes", {}).get("time", 24.6)
    else:
        # Fallback hardcoded values
        basis_dims = [4, 6, 8, 12, 16, 20]
        rmses = [0.039, 0.044, 0.044, 0.042, 0.043, 0.040]
        times_sees = [1.6, 1.7, 1.3, 2.0, 2.2, 2.5]
        nfxp_time = 7.6
        nnes_time = 24.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Left: RMSE vs basis dimension
    ax1.plot(basis_dims, rmses, "o-", color="#4CAF50", linewidth=2, markersize=6)
    ax1.set_xlabel("SEES Basis Dimension ($K$)")
    ax1.set_ylabel("RMSE vs NFXP Parameters")
    ax1.set_title("(a) Accuracy vs Compression")
    ax1.set_ylim(0, 0.06)

    # Right: timing comparison
    methods = ["NFXP", "SEES\n($K$=8)", "NNES"]
    method_times = [nfxp_time, times_sees[2] if len(times_sees) > 2 else 1.3, nnes_time]
    method_colors = ["#2196F3", "#4CAF50", "#4CAF50"]
    ax2.bar(methods, method_times, color=method_colors, edgecolor="white")
    ax2.set_ylabel("Computation Time (s)")
    ax2.set_title("(b) Speed Comparison (400 states)")

    # Annotate bars
    for i, t in enumerate(method_times):
        ax2.text(i, t + 0.5, f"{t:.1f}s", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig2_scaling.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved fig2_scaling.pdf")


def fig3_nonlinear():
    """Case Study 3: Objectworld policy accuracy comparison."""

    # Data from examples/wulfmeier-deep-maxent/f_irl_results.json
    estimators = ["MCE-IRL\n(linear)", "f-IRL\n($\\chi^2$)", "f-IRL\n(KL)"]
    accuracies = [100.0, 89.1, 79.7]
    times = [415.7, 23.8, 22.4]
    colors = ["#2196F3", "#FF9800", "#FF9800"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: policy accuracy
    bars = ax1.bar(estimators, accuracies, color=colors, edgecolor="white")
    ax1.set_ylabel("Policy Accuracy (%)")
    ax1.set_title("(a) Policy Recovery")
    ax1.set_ylim(0, 110)
    for i, v in enumerate(accuracies):
        ax1.text(i, v + 2, f"{v:.1f}%", ha="center", fontsize=8)

    # Right: computation time
    bars2 = ax2.bar(estimators, times, color=colors, edgecolor="white")
    ax2.set_ylabel("Computation Time (s)")
    ax2.set_title("(b) Speed")
    for i, v in enumerate(times):
        ax2.text(i, v + 10, f"{v:.0f}s", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig3_nonlinear.pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved fig3_nonlinear.pdf")


if __name__ == "__main__":
    print("Generating figures for EconIRL package paper...")
    fig1_tabular_convergence()
    fig2_scaling()
    fig3_nonlinear()
    print("Done.")
