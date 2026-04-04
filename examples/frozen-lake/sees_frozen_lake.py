"""SEES sieve estimation on the FrozenLake environment.

Demonstrates the Sieve Estimator (SEES) on a 4x4 FrozenLake grid. SEES
approximates the value function with sieve basis functions (Fourier or
polynomial) and jointly optimizes structural parameters and basis coefficients
via penalized MLE. This avoids both the inner fixed-point loop of NFXP and
the neural network training of TD-CCP.

Usage:
    python examples/frozen-lake/sees_frozen_lake.py
"""

import numpy as np

from econirl.environments.frozen_lake import FrozenLakeEnvironment
from econirl.estimation.sees import SEESConfig, SEESEstimator
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
print("FrozenLake SEES Estimation")
print("=" * 60)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Observations: {panel.num_observations}")
print(f"True parameters: {env.true_parameters}")
print()

# ---------------------------------------------------------------------------
# SEES estimation with Fourier basis
# ---------------------------------------------------------------------------

print("--- SEES (Fourier basis, K=8) ---")
config_fourier = SEESConfig(
    basis_type="fourier",
    basis_dim=8,
    penalty_lambda=0.01,
    max_iter=500,
    compute_se=True,
    se_method="asymptotic",
    verbose=False,
)
estimator_fourier = SEESEstimator(config=config_fourier)
result_fourier = estimator_fourier.estimate(panel, utility, problem, transitions)
print(result_fourier.summary())

# ---------------------------------------------------------------------------
# SEES estimation with polynomial basis
# ---------------------------------------------------------------------------

print()
print("--- SEES (Polynomial basis, K=6) ---")
config_poly = SEESConfig(
    basis_type="polynomial",
    basis_dim=6,
    penalty_lambda=0.01,
    max_iter=500,
    compute_se=True,
    se_method="asymptotic",
    verbose=False,
)
estimator_poly = SEESEstimator(config=config_poly)
result_poly = estimator_poly.estimate(panel, utility, problem, transitions)
print(result_poly.summary())

# ---------------------------------------------------------------------------
# Parameter recovery comparison
# ---------------------------------------------------------------------------

true_dict = env.true_parameters
print()
print("=" * 60)
print("Parameter Recovery Comparison")
print("=" * 60)
print(f"{'Parameter':<16} {'True':>8} {'Fourier':>8} {'Poly':>8}")
print("-" * 44)
for i, name in enumerate(env.parameter_names):
    true_val = true_dict[name]
    fourier_val = float(result_fourier.parameters[i])
    poly_val = float(result_poly.parameters[i])
    print(f"{name:<16} {true_val:>8.4f} {fourier_val:>8.4f} {poly_val:>8.4f}")

# Sieve coefficients
for label, result in [("Fourier", result_fourier), ("Polynomial", result_poly)]:
    if result.metadata and "alpha" in result.metadata:
        alpha = np.asarray(result.metadata["alpha"])
        print()
        print(f"{label} basis coefficients (alpha):")
        for j, val in enumerate(alpha):
            print(f"  alpha[{j}] = {float(val):.6f}")

# Log-likelihoods and convergence
print()
print(f"Log-likelihood:  Fourier={float(result_fourier.log_likelihood):.2f}  "
      f"Poly={float(result_poly.log_likelihood):.2f}")
print(f"Converged:       Fourier={result_fourier.converged}  "
      f"Poly={result_poly.converged}")
