#!/usr/bin/env python3
"""Generate figures for the RDW scrappage docs page."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / "docs" / "_static"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Figure 1: Scrappage rate by age (from real data)
# ---------------------------------------------------------------------------

ages =     [3,    4,    5,    6,    7,    8,    9,   10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20]
rates =    [0.0003, 0.001, 0.003, 0.0001, 0.0001, 0.0004, 0.002, 0.012, 0.016, 0.016, 0.013, 0.018, 0.024, 0.034, 0.033, 0.043, 0.060, 0.082]
n_obs =    [170852,170799,170643,170202,170177,170158,170096,169784,144332,120934,99199,84478,67364,50516,33813,23314,15136,7347]

fig, ax1 = plt.subplots(figsize=(10, 5))

color_rate = "#2166ac"
color_obs = "#b2182b"

ax1.bar(ages, [r * 100 for r in rates], color=color_rate, alpha=0.8, label="Scrappage rate")
ax1.set_xlabel("Vehicle Age (years)", fontsize=12)
ax1.set_ylabel("Annual Scrappage Rate (%)", fontsize=12, color=color_rate)
ax1.tick_params(axis="y", labelcolor=color_rate)
ax1.set_ylim(0, 10)

ax2 = ax1.twinx()
ax2.plot(ages, [n / 1000 for n in n_obs], color=color_obs, linewidth=2, marker="o",
         markersize=4, label="Vehicles observed")
ax2.set_ylabel("Vehicles Observed (thousands)", fontsize=12, color=color_obs)
ax2.tick_params(axis="y", labelcolor=color_obs)

fig.suptitle("Scrappage Rate by Vehicle Age\n170,852 VW Golfs (2005–2015), RDW Open Data", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "rdw_scrappage_by_age.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'rdw_scrappage_by_age.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Defect distribution by age (from real data)
# ---------------------------------------------------------------------------

age_labels = [5, 10, 15, 20]
pass_pct =  [100.0, 89.7, 72.9, 51.0]
minor_pct = [0.0,    7.0, 18.2, 30.1]
major_pct = [0.0,    3.3,  9.0, 18.9]

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(age_labels))
w = 0.25

ax.bar(x - w, pass_pct, w, label="Pass (no defects)", color="#4daf4a")
ax.bar(x, minor_pct, w, label="Minor defects", color="#ff7f00")
ax.bar(x + w, major_pct, w, label="Major defects", color="#e41a1c")

ax.set_xlabel("Vehicle Age (years)", fontsize=12)
ax.set_ylabel("Percentage of Vehicles (%)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(age_labels)
ax.legend()
ax.set_title("APK Inspection Defect Distribution by Age\nRDW Open Data", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "rdw_defect_by_age.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'rdw_defect_by_age.png'}")


# ---------------------------------------------------------------------------
# Figure 3: Counterfactual — subsidy effect on scrappage probability
# From showcase run: baseline vs 30% subsidy
# ---------------------------------------------------------------------------

cf_ages = [5, 10, 15, 20]

# Values from the 10K-vehicle showcase run on real data (defect=0 / pass state)
baseline_pass =      [0.0001, 0.0042, 0.0247, 0.0095]
subsidy_pass =       [0.0014, 0.0399, 0.1844, 0.0962]
baseline_major =     [0.0028, 0.0070, 0.0391, 0.0141]
subsidy_major =      [0.0296, 0.0642, 0.2628, 0.1369]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

x = np.arange(len(cf_ages))
w = 0.3

# Left panel: pass (no defects)
ax1.bar(x - w/2, [b*100 for b in baseline_pass], w, label="Baseline", color="#2166ac", alpha=0.8)
ax1.bar(x + w/2, [s*100 for s in subsidy_pass], w, label="30% Subsidy", color="#b2182b", alpha=0.8)
ax1.set_xlabel("Vehicle Age (years)", fontsize=12)
ax1.set_ylabel("Scrappage Probability (%)", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(cf_ages)
ax1.legend()
ax1.set_title("No Defects (d = 0)", fontsize=12, fontweight="bold")

# Right panel: major defects
ax2.bar(x - w/2, [b*100 for b in baseline_major], w, label="Baseline", color="#2166ac", alpha=0.8)
ax2.bar(x + w/2, [s*100 for s in subsidy_major], w, label="30% Subsidy", color="#b2182b", alpha=0.8)
ax2.set_xlabel("Vehicle Age (years)", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(cf_ages)
ax2.legend()
ax2.set_title("Major Defects (d = 2)", fontsize=12, fontweight="bold")

fig.suptitle("Scrappage Subsidy Counterfactual\n30% Reduction in Replacement Cost", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "rdw_subsidy_counterfactual.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'rdw_subsidy_counterfactual.png'}")


# ---------------------------------------------------------------------------
# Figure 4: Elasticity curve — welfare change vs RC perturbation
# From showcase run
# ---------------------------------------------------------------------------

pct_changes = [-50, -30, -10, 10, 30, 50]
welfare_changes = [7.9027, 2.5603, 0.3824, -0.1763, -0.2891, -0.3105]
policy_changes = [0.2413, 0.0854, 0.0135, 0.0063, 0.0104, 0.0112]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(pct_changes, welfare_changes, "o-", color="#2166ac", linewidth=2, markersize=8)
ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8)
ax1.set_xlabel("Change in Replacement Cost (%)", fontsize=12)
ax1.set_ylabel("Welfare Change", fontsize=12)
ax1.set_title("Welfare Elasticity", fontsize=12, fontweight="bold")

ax2.bar(pct_changes, [p*100 for p in policy_changes], width=8, color="#b2182b", alpha=0.8)
ax2.set_xlabel("Change in Replacement Cost (%)", fontsize=12)
ax2.set_ylabel("Average Policy Change (%)", fontsize=12)
ax2.set_title("Policy Sensitivity", fontsize=12, fontweight="bold")

fig.suptitle("Elasticity of Scrappage to Replacement Cost", fontsize=13, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(OUT / "rdw_elasticity.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved {OUT / 'rdw_elasticity.png'}")

print("\nAll figures generated.")
