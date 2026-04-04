"""Behavioral cloning versus structural estimation on FrozenLake.

Demonstrates that behavioral cloning (BC) matches observed choice frequencies
but does not recover structural parameters, while NFXP recovers the true
utility parameters from the same data. BC serves as the lower bound baseline
for every benchmark in the library.

Usage:
    python examples/frozen-lake/bc_frozen_lake.py
"""

import numpy as np

from econirl.environments.frozen_lake import FrozenLakeEnvironment
from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility

# ---------------------------------------------------------------------------
# Environment and data
# ---------------------------------------------------------------------------

env = FrozenLakeEnvironment(discount_factor=0.99)
panel = env.generate_panel(n_individuals=500, n_periods=100, seed=42)

utility = LinearUtility(
    feature_matrix=env.feature_matrix,
    parameter_names=env.parameter_names,
)
transitions = env.transition_matrices
problem = env.problem_spec

print("=" * 60)
print("FrozenLake: Behavioral Cloning vs NFXP")
print("=" * 60)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Observations: {panel.num_observations}")
print(f"True parameters: {env.true_parameters}")
print()

# ---------------------------------------------------------------------------
# Behavioral Cloning
# ---------------------------------------------------------------------------

print("--- Behavioral Cloning ---")
bc_est = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
bc_result = bc_est.estimate(panel, utility, problem, transitions)
print(f"Log-likelihood: {float(bc_result.log_likelihood):.2f}")
print(f"BC does not recover structural parameters (it only counts frequencies).")
print()

# ---------------------------------------------------------------------------
# NFXP (structural estimator)
# ---------------------------------------------------------------------------

print("--- NFXP ---")
nfxp_est = NFXPEstimator(se_method="robust", verbose=False)
nfxp_result = nfxp_est.estimate(panel, utility, problem, transitions)
print(nfxp_result.summary())

# ---------------------------------------------------------------------------
# Policy comparison
# ---------------------------------------------------------------------------

bc_policy = np.asarray(bc_result.policy)
nfxp_policy = np.asarray(nfxp_result.policy)

print()
print("=" * 60)
print("Policy Comparison at Selected States")
print("=" * 60)
print(f"{'State':<8} {'BC P(Left)':>10} {'NFXP P(Left)':>12} "
      f"{'BC P(Down)':>10} {'NFXP P(Down)':>12}")
print("-" * 56)
sample_states = [0, 1, 2, 3, 6, 9, 10, 13, 14]
for s in sample_states:
    row, col = s // 4, s % 4
    label = f"{s} ({row},{col})"
    print(f"{label:<8} {bc_policy[s, 0]:>10.3f} {nfxp_policy[s, 0]:>12.3f} "
          f"{bc_policy[s, 1]:>10.3f} {nfxp_policy[s, 1]:>12.3f}")

# Policy distance
policy_diff = np.abs(bc_policy - nfxp_policy).mean()
print()
print(f"Mean absolute policy difference (BC vs NFXP): {policy_diff:.4f}")

# Parameter recovery (NFXP only)
true_dict = env.true_parameters
print()
print("NFXP Parameter Recovery")
print(f"{'Parameter':<16} {'True':>8} {'NFXP':>8} {'SE':>8}")
print("-" * 44)
for i, name in enumerate(env.parameter_names):
    true_val = true_dict[name]
    est_val = float(nfxp_result.parameters[i])
    se_val = float(nfxp_result.standard_errors[i])
    print(f"{name:<16} {true_val:>8.4f} {est_val:>8.4f} {se_val:>8.4f}")

print()
print("BC matches observed frequencies but has no parameters to recover.")
print("NFXP recovers the structural parameters that generated the data.")
