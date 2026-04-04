"""f-IRL reward recovery on simulated Rust bus engine data.

Demonstrates the f-IRL estimator (Ni et al. 2022) which recovers a tabular
reward function by matching the state-action marginal distribution of the
learned policy to the expert's empirical marginal. Unlike MaxEnt IRL, f-IRL
does not require a feature representation and instead operates directly on
state-action occupancies using f-divergence minimization.

Two f-divergences are compared (KL and chi-squared) alongside an NFXP
structural estimate as baseline. The Rust bus environment provides a fast
tabular MDP with known ground truth for parameter recovery evaluation.

Usage:
    python examples/frozen-lake/f_irl_frozen_lake.py
"""

import numpy as np
import jax.numpy as jnp

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.f_irl import FIRLEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel

# ── Setup ──

env = RustBusEnvironment(
    operating_cost=0.001,
    replacement_cost=3.0,
    num_mileage_bins=90,
    discount_factor=0.9999,
)

utility = LinearUtility.from_environment(env)
transitions = env.transition_matrices
problem = env.problem_spec
true_params = env.get_true_parameter_vector()

panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)

print("=" * 60)
print("f-IRL Reward Recovery on Rust Bus Data")
print("=" * 60)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Discount factor: {problem.discount_factor}")
print(f"Individuals: {panel.num_individuals}")
print(f"Observations: {panel.num_observations:,}")
print(f"True parameters: {env.true_parameters}")
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
print(f"NFXP converged: {nfxp_result.converged}")
print(f"NFXP log-likelihood: {float(nfxp_result.log_likelihood):.2f}")
print()

# ── f-IRL estimation (KL divergence) ──

print("--- Running f-IRL (KL divergence) ---")
firl_kl = FIRLEstimator(
    f_divergence="kl",
    lr=0.5,
    max_iter=200,
    inner_tol=1e-6,
    inner_max_iter=2000,
    horizon=50,
    verbose=True,
)
firl_kl_result = firl_kl.estimate(panel, utility, problem, transitions)
print(f"f-IRL (KL) log-likelihood: {float(firl_kl_result.log_likelihood):.2f}")
print()

# ── f-IRL estimation (chi-squared divergence) ──

print("--- Running f-IRL (chi-squared divergence) ---")
firl_chi2 = FIRLEstimator(
    f_divergence="chi2",
    lr=0.5,
    max_iter=200,
    inner_tol=1e-6,
    inner_max_iter=2000,
    horizon=50,
    verbose=True,
)
firl_chi2_result = firl_chi2.estimate(panel, utility, problem, transitions)
print(f"f-IRL (chi2) log-likelihood: {float(firl_chi2_result.log_likelihood):.2f}")
print()

# ── Compare policies ──

print("=" * 60)
print("Replacement Probability by Mileage State")
print("=" * 60)

nfxp_policy = jnp.asarray(nfxp_result.policy)
firl_kl_policy = jnp.asarray(firl_kl_result.policy)
firl_chi2_policy = jnp.asarray(firl_chi2_result.policy)

states_to_show = [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]
print(f"{'State':<8} {'NFXP':>12} {'f-IRL KL':>12} {'f-IRL chi2':>12}")
print("-" * 46)
for s in states_to_show:
    nfxp_p = float(nfxp_policy[s, 1])
    kl_p = float(firl_kl_policy[s, 1])
    chi2_p = float(firl_chi2_policy[s, 1])
    print(f"{s:<8} {nfxp_p:>12.4f} {kl_p:>12.4f} {chi2_p:>12.4f}")
print()

# ── Policy agreement ──

max_diff_kl = float(jnp.abs(nfxp_policy - firl_kl_policy).max())
max_diff_chi2 = float(jnp.abs(nfxp_policy - firl_chi2_policy).max())
print(f"Max policy difference vs NFXP:")
print(f"  f-IRL KL:   {max_diff_kl:.4f}")
print(f"  f-IRL chi2: {max_diff_chi2:.4f}")
print()

# ── Recovered reward comparison ──

print("=" * 60)
print("Recovered Reward R(s, a=keep) at Selected States")
print("=" * 60)

kl_reward = jnp.asarray(firl_kl_result.metadata["reward_matrix"])
chi2_reward = jnp.asarray(firl_chi2_result.metadata["reward_matrix"])

# Compute NFXP structural reward for comparison
nfxp_reward = jnp.einsum(
    "sak,k->sa",
    jnp.asarray(utility.feature_matrix),
    jnp.asarray(nfxp_result.parameters),
)

print(f"{'State':<8} {'NFXP':>12} {'f-IRL KL':>12} {'f-IRL chi2':>12}")
print("-" * 46)
a = 0  # keep action
for s in states_to_show:
    print(
        f"{s:<8} {float(nfxp_reward[s, a]):>12.4f} "
        f"{float(kl_reward[s, a]):>12.4f} {float(chi2_reward[s, a]):>12.4f}"
    )
print()

# ── Fit statistics ──

print("=" * 60)
print("Fit Statistics")
print("=" * 60)
print(f"{'Estimator':<18} {'Log-Likelihood':>15} {'Converged':>10}")
print("-" * 45)
print(f"{'NFXP':<18} {float(nfxp_result.log_likelihood):>15.2f} {str(nfxp_result.converged):>10}")
print(f"{'f-IRL (KL)':<18} {float(firl_kl_result.log_likelihood):>15.2f} {str(firl_kl_result.converged):>10}")
print(f"{'f-IRL (chi2)':<18} {float(firl_chi2_result.log_likelihood):>15.2f} {str(firl_chi2_result.converged):>10}")
