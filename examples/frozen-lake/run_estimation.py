"""FrozenLake parameter recovery with three estimators and post-estimation diagnostics.

Demonstrates NFXP, CCP, and MCE-IRL on the classic 4x4 FrozenLake environment.
Known ground truth enables direct parameter recovery evaluation.

Usage:
    python examples/frozen-lake/run_estimation.py
"""

import numpy as np

from econirl.environments.frozen_lake import FrozenLakeEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

env = FrozenLakeEnvironment(discount_factor=0.99)
panel = env.generate_panel(n_individuals=500, n_periods=100, seed=42)

utility = LinearUtility(
    feature_matrix=env.feature_matrix,
    parameter_names=env.parameter_names,
)
transitions = env.transition_matrices
problem = env.problem_spec

print("=" * 70)
print("FrozenLake Parameter Recovery")
print("=" * 70)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Individuals: {panel.num_individuals}, Observations: {panel.num_observations}")
print(f"True parameters: {env.true_parameters}")
print()

# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------

# NFXP
print("--- NFXP ---")
nfxp_est = NFXPEstimator(se_method="robust", verbose=False)
nfxp_result = nfxp_est.estimate(panel, utility, problem, transitions)
print(nfxp_result.summary())
print()

# CCP
print("--- CCP ---")
ccp_est = CCPEstimator(num_policy_iterations=20, se_method="robust", verbose=False)
ccp_result = ccp_est.estimate(panel, utility, problem, transitions)
print(ccp_result.summary())
print()

# MCE-IRL
print("--- MCE-IRL ---")
mce_config = MCEIRLConfig(
    learning_rate=0.1,
    outer_max_iter=500,
    outer_tol=1e-6,
    inner_max_iter=200,
)
mce_est = MCEIRLEstimator(config=mce_config)
mce_result = mce_est.estimate(panel, utility, problem, transitions)
print(mce_result.summary())
print()

# ---------------------------------------------------------------------------
# Parameter recovery table
# ---------------------------------------------------------------------------

true_params = env.get_true_parameter_vector()
true_dict = env.true_parameters

print("=" * 70)
print("Parameter Recovery Table")
print("=" * 70)
print(f"{'Parameter':<16} {'True':>8} {'NFXP':>8} {'CCP':>8} {'MCE-IRL':>8}")
print("-" * 56)
for i, name in enumerate(env.parameter_names):
    true_val = true_dict[name]
    nfxp_val = float(nfxp_result.parameters[i])
    ccp_val = float(ccp_result.parameters[i])
    mce_val = float(mce_result.parameters[i])
    print(f"{name:<16} {true_val:>8.4f} {nfxp_val:>8.4f} {ccp_val:>8.4f} {mce_val:>8.4f}")
print()

# Standard errors
print("Standard Errors")
print("-" * 56)
print(f"{'Parameter':<16} {'NFXP':>8} {'CCP':>8} {'MCE-IRL':>8}")
print("-" * 56)
for i, name in enumerate(env.parameter_names):
    nfxp_se = float(nfxp_result.standard_errors[i])
    ccp_se = float(ccp_result.standard_errors[i])
    mce_se = float(mce_result.standard_errors[i])
    print(f"{name:<16} {nfxp_se:>8.4f} {ccp_se:>8.4f} {mce_se:>8.4f}")
print()

# Log-likelihoods
print(f"Log-likelihood:  NFXP={float(nfxp_result.log_likelihood):.1f}  "
      f"CCP={float(ccp_result.log_likelihood):.1f}  "
      f"MCE-IRL={float(mce_result.log_likelihood):.1f}")
print()

# ---------------------------------------------------------------------------
# Post-estimation diagnostics
# ---------------------------------------------------------------------------

print("=" * 70)
print("Post-Estimation Diagnostics")
print("=" * 70)

# Comparison table
from econirl.inference import etable
print("\n--- etable() ---")
print(etable(nfxp_result, ccp_result, mce_result))

# Extract obs arrays for low-level functions
import jax.numpy as jnp
obs_states = jnp.array(panel.get_all_states())
obs_actions = jnp.array(panel.get_all_actions())

# Vuong test: NFXP vs MCE-IRL
from econirl.inference import vuong_test
print("\n--- Vuong Test (NFXP vs MCE-IRL) ---")
vt = vuong_test(nfxp_result.policy, mce_result.policy, obs_states, obs_actions)
print(f"Z-statistic: {vt['statistic']:.3f}")
print(f"P-value: {vt['p_value']:.4f}")
print(f"Direction: {vt['direction']}")

# Brier score
from econirl.inference import brier_score
print("\n--- Brier Scores ---")
for name, result in [("NFXP", nfxp_result), ("CCP", ccp_result), ("MCE-IRL", mce_result)]:
    bs = brier_score(result.policy, obs_states, obs_actions)
    print(f"{name}: {bs['brier_score']:.4f}")

# KL divergence
from econirl.inference import kl_divergence
print("\n--- KL Divergence ---")
sufficient = panel.sufficient_stats(env.num_states, env.num_actions)
data_ccps = jnp.array(sufficient.empirical_ccps)
for name, result in [("NFXP", nfxp_result), ("CCP", ccp_result), ("MCE-IRL", mce_result)]:
    kl = kl_divergence(data_ccps, result.policy)
    print(f"{name}: {kl['kl_divergence']:.4f}")

# Reward identifiability
from econirl.inference import check_reward_identifiability
print("\n--- Reward Identifiability ---")
ident = check_reward_identifiability(np.asarray(transitions))
print(f"Status: {ident['status']}")
for k, v in ident.items():
    if k != "status":
        print(f"  {k}: {v}")
