"""Generate scaling benchmark figures: time, accuracy, transfer, and Pareto front."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

df = pd.read_csv("docs/scaling_benchmark.csv")

# Drop skipped rows
df_active = df[~df["skipped"]].copy()

# Color scheme by family
FAMILY_COLORS = {
    # Forward (blues)
    "NFXP": "#1f77b4",
    "CCP": "#4a9bd9",
    # IRL (greens)
    "MCE IRL": "#2ca02c",
    "MaxEnt IRL": "#5fd35f",
    "Max Margin": "#8fbc8f",
    "Max Margin IRL": "#b2dfb2",
    "f-IRL": "#006400",
    # Neural (oranges/reds)
    "TD-CCP": "#ff7f0e",
    "GLADIUS": "#d62728",
    "NNES": "#e377c2",
    "Deep MaxEnt": "#ff9896",
    # Adversarial (purples)
    "GAIL": "#9467bd",
    "AIRL": "#c5b0d5",
    "GCL": "#8c564b",
    # Other
    "BC": "#7f7f7f",
    "SEES": "#bcbd22",
    "BIRL": "#17becf",
}

FAMILY_MARKERS = {
    "NFXP": "o", "CCP": "s",
    "MCE IRL": "^", "MaxEnt IRL": "v", "Max Margin": "<", "Max Margin IRL": ">", "f-IRL": "D",
    "TD-CCP": "P", "GLADIUS": "*", "NNES": "X", "Deep MaxEnt": "h",
    "GAIL": "p", "AIRL": "H", "GCL": "8",
    "BC": ".", "SEES": "d", "BIRL": "+",
}

# Only plot estimators that appear at >= 3 state sizes
estimator_counts = df_active.groupby("estimator")["n_states"].nunique()
scalable = estimator_counts[estimator_counts >= 3].index.tolist()

# ── Figure 1: Time + Accuracy (2 panels) ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

for est in scalable:
    sub = df_active[df_active["estimator"] == est].sort_values("n_states")
    ax1.plot(
        sub["n_states"], sub["time_seconds"],
        marker=FAMILY_MARKERS.get(est, "o"),
        color=FAMILY_COLORS.get(est, "#333"),
        label=est, linewidth=1.8, markersize=7,
    )

ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("Number of States", fontsize=13)
ax1.set_ylabel("Wall-Clock Time (seconds)", fontsize=13)
ax1.set_title("Scaling: Estimation Time vs State Space Size", fontsize=14)
ax1.set_xticks([5, 10, 20, 50, 100, 200, 500])
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax1.legend(fontsize=8, ncol=2, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.axhline(y=900, color="red", linestyle="--", alpha=0.5, label="15min timeout")

for est in scalable:
    sub = df_active[df_active["estimator"] == est].sort_values("n_states")
    ax2.plot(
        sub["n_states"], sub["pct_optimal"],
        marker=FAMILY_MARKERS.get(est, "o"),
        color=FAMILY_COLORS.get(est, "#333"),
        label=est, linewidth=1.8, markersize=7,
    )

ax2.set_xscale("log")
ax2.set_xlabel("Number of States", fontsize=13)
ax2.set_ylabel("% of Optimal Value", fontsize=13)
ax2.set_title("Scaling: Policy Quality vs State Space Size", fontsize=14)
ax2.set_xticks([5, 10, 20, 50, 100, 200, 500])
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8, ncol=2, loc="lower left")
ax2.grid(True, alpha=0.3)
ax2.axhline(y=90, color="green", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("docs/scaling_benchmark.png", dpi=150, bbox_inches="tight")
print("Saved docs/scaling_benchmark.png")

# ── Figure 2: Transfer Performance ──
has_transfer = "pct_optimal_transfer" in df_active.columns
if has_transfer:
    df_transfer = df_active.dropna(subset=["pct_optimal_transfer"])
    transfer_counts = df_transfer.groupby("estimator")["n_states"].nunique()
    scalable_transfer = transfer_counts[transfer_counts >= 3].index.tolist()

    if scalable_transfer:
        fig_t, (ax_t1, ax_t2) = plt.subplots(1, 2, figsize=(16, 7))

        for est in scalable_transfer:
            sub = df_transfer[df_transfer["estimator"] == est].sort_values("n_states")
            ax_t1.plot(
                sub["n_states"], sub["pct_optimal"],
                marker=FAMILY_MARKERS.get(est, "o"),
                color=FAMILY_COLORS.get(est, "#333"),
                label=est, linewidth=1.8, markersize=7,
            )

        ax_t1.set_xscale("log")
        ax_t1.set_xlabel("Number of States", fontsize=13)
        ax_t1.set_ylabel("% of Optimal Value", fontsize=13)
        ax_t1.set_title("In-Sample Policy Quality", fontsize=14)
        ax_t1.set_xticks([5, 10, 20, 50, 100, 200, 500])
        ax_t1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_t1.set_ylim(0, 105)
        ax_t1.legend(fontsize=8, ncol=2, loc="lower left")
        ax_t1.grid(True, alpha=0.3)
        ax_t1.axhline(y=90, color="green", linestyle="--", alpha=0.3)

        for est in scalable_transfer:
            sub = df_transfer[df_transfer["estimator"] == est].sort_values("n_states")
            ax_t2.plot(
                sub["n_states"], sub["pct_optimal_transfer"],
                marker=FAMILY_MARKERS.get(est, "o"),
                color=FAMILY_COLORS.get(est, "#333"),
                label=est, linewidth=1.8, markersize=7,
            )

        ax_t2.set_xscale("log")
        ax_t2.set_xlabel("Number of States", fontsize=13)
        ax_t2.set_ylabel("% of Optimal (Transfer)", fontsize=13)
        ax_t2.set_title("Transfer Performance (Different Dynamics)", fontsize=14)
        ax_t2.set_xticks([5, 10, 20, 50, 100, 200, 500])
        ax_t2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax_t2.set_ylim(0, 105)
        ax_t2.legend(fontsize=8, ncol=2, loc="lower left")
        ax_t2.grid(True, alpha=0.3)
        ax_t2.axhline(y=90, color="green", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig("docs/scaling_transfer.png", dpi=150, bbox_inches="tight")
        print("Saved docs/scaling_transfer.png")

# ── Figure 3: Speed-Accuracy Pareto at n=100 ──
fig2, ax3 = plt.subplots(figsize=(10, 7))

sub100 = df_active[df_active["n_states"] == 100].copy()
for _, row in sub100.iterrows():
    est = row["estimator"]
    ax3.scatter(
        row["time_seconds"], row["pct_optimal"],
        color=FAMILY_COLORS.get(est, "#333"),
        marker=FAMILY_MARKERS.get(est, "o"),
        s=120, zorder=5,
    )
    ax3.annotate(
        est, (row["time_seconds"], row["pct_optimal"]),
        textcoords="offset points", xytext=(8, 4), fontsize=9,
    )

ax3.set_xlabel("Wall-Clock Time (seconds)", fontsize=13)
ax3.set_ylabel("% of Optimal Value", fontsize=13)
ax3.set_title("Speed vs Accuracy Pareto Front (n_states=100)", fontsize=14)
ax3.set_xscale("log")
ax3.set_ylim(50, 105)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=90, color="green", linestyle="--", alpha=0.3, label="90% threshold")
ax3.axvline(x=60, color="orange", linestyle="--", alpha=0.3, label="1 min threshold")
ax3.legend(fontsize=10)

plt.tight_layout()
plt.savefig("docs/scaling_pareto.png", dpi=150, bbox_inches="tight")
print("Saved docs/scaling_pareto.png")
