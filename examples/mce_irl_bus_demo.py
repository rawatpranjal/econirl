"""
Maximum Causal Entropy IRL on Rust Bus Engine Data

Demonstrates:
1. Running MCE IRL on the bus engine replacement problem
2. Inference on reward parameters
3. Prediction using the learned reward function
"""

import numpy as np
import pandas as pd

from econirl.datasets import load_rust_bus
from econirl.estimators.mce_irl import MCEIRL

# Load the bus data
print("=" * 70)
print("Loading Rust Bus Engine Data")
print("=" * 70)
df = load_rust_bus()
print(f"Observations: {len(df):,}")
print(f"Buses: {df['bus_id'].nunique()}")
print(f"Replacement rate: {df['replaced'].mean():.2%}")
print(f"Mean mileage bin: {df['mileage_bin'].mean():.2f}")
print()

# Analyze the data pattern
print("Replacement rate by mileage bin:")
for b in range(0, 90, 10):
    subset = df[(df['mileage_bin'] >= b) & (df['mileage_bin'] < b + 10)]
    if len(subset) > 0:
        print(f"  Bins {b:2d}-{b+9}: {subset['replaced'].mean():.2%} ({len(subset):4d} obs)")
print()

# Define state features
n_states = 90
n_actions = 2
s = np.arange(n_states)
features = np.column_stack([
    s / 100.0,            # Linear: normalized mileage
    (s / 100.0) ** 2,     # Quadratic
])

print("=" * 70)
print("Running Maximum Causal Entropy IRL")
print("=" * 70)
print(f"State features: {features.shape}")
print(f"Discount factor: 0.99")
print()

# Create and fit (no bootstrap for speed)
model = MCEIRL(
    n_states=n_states,
    n_actions=n_actions,
    discount=0.99,
    feature_matrix=features,
    feature_names=["linear_cost", "quadratic_cost"],
    se_method="hessian",  # Fast - numerical Hessian instead of bootstrap
    inner_max_iter=5000,
    verbose=True,
)

model.fit(
    data=df,
    state="mileage_bin",
    action="replaced",
    id="bus_id",
)

# Results
print()
print(model.summary())
print()

# Recovered reward function
print("=" * 70)
print("RECOVERED REWARD FUNCTION")
print("=" * 70)
print()
if model.reward_ is not None:
    r0 = model.reward_[0]
    print(f"{'State':>6} {'Mileage':>10} {'R(s)':>12} {'R(s) - R(0)':>14}")
    print("-" * 50)
    for s in [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]:
        r = model.reward_[s]
        print(f"{s:>6} {s*5:>10}k {r:>12.6f} {r - r0:>14.6f}")
print()

# Prediction
print("=" * 70)
print("PREDICTION: Choice Probabilities")
print("=" * 70)
print()
test_states = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 89])
proba = model.predict_proba(test_states)
print(f"{'State':>6} {'P(keep)':>12} {'P(replace)':>14}")
print("-" * 35)
for i, s in enumerate(test_states):
    print(f"{s:>6} {proba[i, 0]:>12.4f} {proba[i, 1]:>14.4f}")
print()

# Compare to empirical frequencies
print("=" * 70)
print("MODEL FIT: Comparison to Empirical")
print("=" * 70)
print()
print(f"{'State':>6} {'Emp P(repl)':>14} {'Model P(repl)':>16}")
print("-" * 40)
for i, s in enumerate(test_states):
    subset = df[df['mileage_bin'] == s]
    model_p = proba[i, 1]
    if len(subset) > 0:
        emp_p = subset['replaced'].mean()
        print(f"{s:>6} {emp_p:>14.4f} {model_p:>16.4f}")
    else:
        print(f"{s:>6} {'N/A':>14} {model_p:>16.4f}")
print()

print("=" * 70)
print(f"Converged: {'Yes' if model.converged_ else 'No'}")
print(f"Log-likelihood: {model.log_likelihood_:.2f}")
print("=" * 70)
