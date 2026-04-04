"""IQ-Learn estimation on simulated Rust bus engine data.

Demonstrates Inverse soft-Q Learning (Garg et al. 2021) for recovering reward
parameters from expert demonstrations. IQ-Learn learns a single soft Q-function
that implicitly defines both the optimal policy and the reward, avoiding the
adversarial training loop required by GAIL and AIRL. The recovered reward is
compared against NFXP structural estimates.

Usage:
    python examples/rust-bus-engine/iq_learn_rust_bus.py
"""

import numpy as np
import jax.numpy as jnp

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.iq_learn import IQLearnEstimator, IQLearnConfig
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel

# ── Setup ──

env = RustBusEnvironment(
    operating_cost=0.001,
    replacement_cost=3.0,
    num_mileage_bins=90,
    discount_factor=0.9999,
)

utility = LinearUtility.from_environment(env)
# IQ-Learn requires ActionDependentReward, which has the same feature matrix
# but is recognized by the IQ-Learn type dispatch for linear Q parameterization.
iq_utility = ActionDependentReward(
    feature_matrix=env.feature_matrix,
    parameter_names=env.parameter_names,
)
problem = env.problem_spec
transitions = env.transition_matrices
true_params = env.get_true_parameter_vector()

panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)

print("=" * 60)
print("IQ-Learn vs NFXP on Simulated Rust Bus Data")
print("=" * 60)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Discount factor: {problem.discount_factor}")
print(f"Observations: {panel.num_observations:,}")
print(f"True parameters: operating_cost=0.001, replacement_cost=3.0")
print()

# ── NFXP estimation (structural benchmark) ──

print("--- Running NFXP ---")
nfxp = NFXPEstimator(
    optimizer="BHHH",
    inner_solver="policy",
    inner_tol=1e-12,
    inner_max_iter=200,
    compute_hessian=True,
    verbose=True,
)
nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
print()

# ── IQ-Learn estimation ──

print("--- Running IQ-Learn (linear Q, chi-squared divergence) ---")
iq_config = IQLearnConfig(
    q_type="linear",
    divergence="chi2",
    optimizer="L-BFGS-B",
    max_iter=500,
    verbose=True,
)
iq_estimator = IQLearnEstimator(config=iq_config)
iq_result = iq_estimator.estimate(panel, iq_utility, problem, transitions)
print()

# ── Compare parameters ──

print("=" * 60)
print("Parameter Comparison")
print("=" * 60)
param_names = utility.parameter_names
nfxp_params = jnp.asarray(nfxp_result.parameters)

print(f"{'Parameter':<18} {'True':>10} {'NFXP':>10} {'IQ-Learn':>10}")
print("-" * 50)
for i, name in enumerate(param_names):
    true_val = float(true_params[i])
    nfxp_val = float(nfxp_params[i])
    iq_val = float(iq_result.parameters[i])
    print(f"{name:<18} {true_val:>10.5f} {nfxp_val:>10.5f} {iq_val:>10.5f}")
print()

# IRL rewards are identified up to constants and scale (Kim et al. 2021),
# so cosine similarity is the appropriate metric.
iq_params = jnp.asarray(iq_result.parameters[:len(true_params)])
cos_sim = float(
    jnp.dot(iq_params, true_params)
    / (jnp.linalg.norm(iq_params) * jnp.linalg.norm(true_params))
)
print(f"Cosine similarity (IQ-Learn vs true): {cos_sim:.4f}")
print()

# ── Policy comparison ──

print("=" * 60)
print("Replacement Probability by Mileage State")
print("=" * 60)
nfxp_policy = jnp.asarray(nfxp_result.policy)
iq_policy = jnp.asarray(iq_result.policy)

states_to_show = [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]
print(f"{'State':<10} {'NFXP P(replace)':>16} {'IQ-Learn P(replace)':>20}")
print("-" * 48)
for s in states_to_show:
    nfxp_p = float(nfxp_policy[s, 1])
    iq_p = float(iq_policy[s, 1])
    print(f"{s:<10} {nfxp_p:>16.4f} {iq_p:>20.4f}")
print()

max_diff = float(jnp.abs(nfxp_policy - iq_policy).max())
print(f"Max policy difference (NFXP vs IQ-Learn): {max_diff:.6f}")
print()

# ── Log-likelihoods ──

print("=" * 60)
print("Fit Statistics")
print("=" * 60)
print(f"NFXP log-likelihood:     {float(nfxp_result.log_likelihood):.2f}")
print(f"IQ-Learn log-likelihood: {float(iq_result.log_likelihood):.2f}")
print(f"NFXP converged:          {nfxp_result.converged}")
print(f"IQ-Learn converged:      {iq_result.converged}")
