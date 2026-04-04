"""TD-CCP parameter recovery on the FrozenLake environment.

Demonstrates the Temporal Difference CCP estimator on a 4x4 FrozenLake grid.
TD-CCP trains per-feature neural EV component networks via semi-gradient TD
learning, then uses the learned components in a partial MLE for structural
parameters. This avoids the inner fixed-point loop required by NFXP.

Usage:
    python examples/frozen-lake/tdccp_frozen_lake.py
"""

import numpy as np

from econirl.environments.frozen_lake import FrozenLakeEnvironment
from econirl.estimation.td_ccp import TDCCPConfig, TDCCPEstimator
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
print("FrozenLake TD-CCP Estimation")
print("=" * 60)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Observations: {panel.num_observations}")
print(f"True parameters: {env.true_parameters}")
print()

# ---------------------------------------------------------------------------
# TD-CCP estimation
# ---------------------------------------------------------------------------

config = TDCCPConfig(
    hidden_dim=32,
    num_hidden_layers=2,
    avi_iterations=15,
    epochs_per_avi=20,
    learning_rate=1e-3,
    batch_size=4096,
    n_policy_iterations=3,
    compute_se=True,
    verbose=True,
)
estimator = TDCCPEstimator(config=config, se_method="asymptotic")
result = estimator.estimate(panel, utility, problem, transitions)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

print()
print("--- TD-CCP Results ---")
print(result.summary())

# Parameter recovery table
true_dict = env.true_parameters
print()
print(f"{'Parameter':<16} {'True':>8} {'TD-CCP':>8}")
print("-" * 34)
for i, name in enumerate(env.parameter_names):
    true_val = true_dict[name]
    est_val = float(result.parameters[i])
    print(f"{name:<16} {true_val:>8.4f} {est_val:>8.4f}")

# EV feature decomposition (the killer feature of TD-CCP)
if result.metadata and "ev_features" in result.metadata:
    ev = np.asarray(result.metadata["ev_features"])
    print()
    print("EV Feature Components (sample states)")
    print(f"{'State':<8}", end="")
    for name in env.parameter_names:
        print(f" {name:>14}", end="")
    print()
    for s in [0, 4, 8, 14, 15]:
        print(f"{s:<8}", end="")
        for k in range(ev.shape[1]):
            print(f" {ev[s, k]:>14.4f}", end="")
        print()

# Convergence info
print()
print(f"Converged: {result.converged}")
print(f"Log-likelihood: {float(result.log_likelihood):.2f}")
