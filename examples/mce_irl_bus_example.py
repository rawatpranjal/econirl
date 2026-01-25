"""
Example: Maximum Causal Entropy IRL on Bus Engine Data

This example demonstrates how to:
1. Load the Rust bus engine replacement data
2. Fit MCE IRL to recover reward parameters
3. Compute standard errors and confidence intervals
4. Make predictions using the learned model

MCE IRL (Ziebart 2010) is a principled approach for inverse reinforcement
learning that resolves ambiguity in reward recovery by maximizing the
causal entropy of the induced trajectory distribution.
"""

import numpy as np
from econirl.datasets import load_rust_bus
from econirl.estimators import MCEIRL

# =============================================================================
# Load Data
# =============================================================================
print("Loading Rust Bus Engine Data")
print("=" * 50)
df = load_rust_bus()
print(f"Observations: {len(df):,}")
print(f"Buses: {df['bus_id'].nunique()}")
print(f"Replacement rate: {df['replaced'].mean():.2%}")
print()

# Show replacement pattern by mileage
print("Replacement rate by mileage bin:")
for b in range(0, 90, 15):
    subset = df[(df["mileage_bin"] >= b) & (df["mileage_bin"] < b + 15)]
    if len(subset) > 0:
        print(f"  Bins {b:2d}-{b+14}: {subset['replaced'].mean():.2%} ({len(subset):4d} obs)")
print()

# =============================================================================
# Define Features
# =============================================================================
n_states = 90
features = np.arange(n_states).reshape(-1, 1) / 100  # Normalized mileage

# =============================================================================
# Fit Model
# =============================================================================
print("Fitting MCE IRL")
print("=" * 50)
model = MCEIRL(
    n_states=n_states,
    n_actions=2,
    discount=0.99,
    feature_matrix=features,
    feature_names=["mileage_cost"],
    se_method="hessian",
    inner_max_iter=500,
    verbose=True,
)

model.fit(
    data=df,
    state="mileage_bin",
    action="replaced",
    id="bus_id",
)

# =============================================================================
# Results Summary
# =============================================================================
print()
print(model.summary())

# =============================================================================
# Predictions
# =============================================================================
print()
print("Predictions")
print("=" * 50)
test_states = np.array([0, 20, 40, 60, 80])
proba = model.predict_proba(test_states)
print(f"{'State':>6} {'P(keep)':>10} {'P(replace)':>12}")
for i, s in enumerate(test_states):
    print(f"{s:>6} {proba[i, 0]:>10.4f} {proba[i, 1]:>12.4f}")

# =============================================================================
# Compare to Empirical Frequencies
# =============================================================================
print()
print("Model Fit: Empirical vs. Predicted")
print("=" * 50)
print(f"{'State':>6} {'Emp P(repl)':>14} {'Model P(repl)':>16}")
print("-" * 40)
for s in [0, 10, 20, 30, 40]:
    subset = df[df["mileage_bin"] == s]
    model_p = model.predict_proba(np.array([s]))[0, 1]
    if len(subset) > 0:
        emp_p = subset["replaced"].mean()
        print(f"{s:>6} {emp_p:>14.4f} {model_p:>16.4f}")
    else:
        print(f"{s:>6} {'N/A':>14} {model_p:>16.4f}")

print()
print("Done!")
