"""Plotting functions for identification experiments.

Generates publication-quality figures from JSON result files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"


def _load_full_simulation():
    path = RESULTS_DIR / "full_simulation.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def plot_shaping_sweep():
    """Plot Type II error vs shaping magnitude alpha.

    Preference order:
    1) Use population-level results from full_simulation.json (section "C").
    2) Fallback to legacy shaping_sweep.json.
    """
    data_full = _load_full_simulation()
    if data_full and "C" in data_full:
        # Keys are strings of alphas; sort numerically
        items = sorted(((float(k), v["error"]) for k, v in data_full["C"].items()), key=lambda x: x[0])
        alphas = [a for a, _ in items]
        errors = [e for _, e in items]
    else:
        with open(RESULTS_DIR / "shaping_sweep.json") as f:
            data = json.load(f)
        alphas = [d["alpha"] for d in data]
        errors = [d["type_ii_error"] for d in data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(alphas, errors, "o-", color="C0", linewidth=2, markersize=6)
    ax.set_xlabel(r"Shaping magnitude $\alpha$ ($\delta = \alpha \cdot V^*$)", fontsize=12)
    ax.set_ylabel("Type II CCP Error", fontsize=12)
    ax.set_title("Counterfactual Accuracy vs Shaping Magnitude", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "shaping_sweep.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'shaping_sweep.png'}")
    plt.close(fig)


def plot_anchor_misspec():
    """Plot Type II error vs anchor misspecification epsilon.

    Preference order:
    1) Use population-level results from full_simulation.json (section "E").
    2) Fallback to legacy anchor_misspec.json.
    """
    data_full = _load_full_simulation()
    if data_full and "E" in data_full:
        items = sorted(((float(k), v["type2_err"]) for k, v in data_full["E"].items()), key=lambda x: x[0])
        epsilons = [e for e, _ in items]
        errors = [err for _, err in items]
    else:
        with open(RESULTS_DIR / "anchor_misspec.json") as f:
            data = json.load(f)
        epsilons = [d["epsilon"] for d in data]
        errors = [d["type_ii_error"] for d in data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epsilons, errors, "s-", color="C1", linewidth=2, markersize=6)
    ax.set_xlabel(r"True exit payoff $\varepsilon$ (anchor assumes 0)", fontsize=12)
    ax.set_ylabel("Type II CCP Error", fontsize=12)
    ax.set_title("Robustness to Anchor Misspecification", fontsize=13)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "anchor_misspec.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'anchor_misspec.png'}")
    plt.close(fig)


def plot_type_ii_by_k():
    """Plot Type II error vs skip magnitude k for all methods.

    Preference order:
    1) Use population-level results from full_simulation.json (section "B").
    2) Fallback to finite-sample main_results.json (field "type_ii").
    """
    data_full = _load_full_simulation()
    if data_full and "B" in data_full:
        # Convert dict of ks -> {method: err}
        ks = sorted(map(int, data_full["B"].keys()))
        # Methods: take keys from first k
        first = data_full["B"][str(ks[0])]
        methods = list(first.keys())
        series = {
            m: [data_full["B"][str(k)][m] for k in ks]
            for m in methods
        }
    else:
        with open(RESULTS_DIR / "main_results.json") as f:
            data = json.load(f)
        type_ii = data.get("type_ii", [])
        if not type_ii:
            print("No Type II results found in main_results.json")
            return
        ks = [r["skip_k"] for r in type_ii]
        methods = [k for k in type_ii[0].keys() if k != "skip_k"]
        series = {m: [r[m] for r in type_ii] for m in methods}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ["o", "s", "^", "D", "v"]
    for i, method in enumerate(methods):
        errors = series[method]
        marker = markers[i % len(markers)]
        ax.plot(ks, errors, f"{marker}-", label=method, linewidth=2, markersize=6)

    ax.set_xlabel("Buy skip magnitude k", fontsize=12)
    ax.set_ylabel("Type II CCP Error", fontsize=12)
    ax.set_title("Counterfactual Accuracy Under Transition Changes", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "type_ii_by_k.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'type_ii_by_k.png'}")
    plt.close(fig)


def plot_sample_size():
    """Plot Type II error vs sample size for each method."""
    path = RESULTS_DIR / "sample_size.json"
    if not path.exists():
        print("sample_size.json not found. Run run_sample_size.py first.")
        return

    with open(path) as f:
        data = json.load(f)

    Ns = [d["N"] for d in data]
    rf_errors = [d["rf_q_error"] for d in data]
    iq_errors = [d["iq_learn_error"] for d in data]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(Ns, rf_errors, "o-", label="Reduced-Form Q", linewidth=2, markersize=6)
    ax.plot(Ns, iq_errors, "s-", label="IQ-Learn", linewidth=2, markersize=6)
    ax.set_xlabel("Number of Individuals (N)", fontsize=12)
    ax.set_ylabel("Type II CCP Error", fontsize=12)
    ax.set_title("Estimation Error vs Sample Size", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "sample_size.png", dpi=150)
    print(f"Saved {RESULTS_DIR / 'sample_size.png'}")
    plt.close(fig)


def main():
    """Generate all available plots."""
    print("Generating plots from identification experiment results...\n")

    if (RESULTS_DIR / "shaping_sweep.json").exists():
        plot_shaping_sweep()

    if (RESULTS_DIR / "anchor_misspec.json").exists():
        plot_anchor_misspec()

    if (RESULTS_DIR / "main_results.json").exists():
        plot_type_ii_by_k()

    if (RESULTS_DIR / "sample_size.json").exists():
        plot_sample_size()

    print("\nDone.")


if __name__ == "__main__":
    main()
