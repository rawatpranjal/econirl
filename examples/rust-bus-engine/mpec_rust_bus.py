"""MPEC estimation on simulated Rust bus engine data.

Demonstrates the MPEC (Mathematical Programming with Equilibrium Constraints)
estimator from Su and Judd (2012) alongside NFXP for comparison. MPEC avoids
nested fixed-point solving by treating the value function as explicit decision
variables optimized jointly with the structural parameters. Both estimators
should recover similar parameters when the model is correctly specified.

Usage:
    python examples/rust-bus-engine/mpec_rust_bus.py
"""

import numpy as np
import jax.numpy as jnp

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mpec import MPECEstimator, MPECConfig
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
problem = env.problem_spec
transitions = env.transition_matrices
true_params = env.get_true_parameter_vector()

panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)

print("=" * 60)
print("MPEC vs NFXP on Simulated Rust Bus Data")
print("=" * 60)
print(f"States: {env.num_states}, Actions: {env.num_actions}")
print(f"Discount factor: {problem.discount_factor}")
print(f"Observations: {panel.num_observations:,}")
print(f"True parameters: operating_cost=0.001, replacement_cost=3.0")
print()

# ── NFXP estimation ──

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

# ── MPEC estimation ──

print("--- Running MPEC ---")
mpec_config = MPECConfig(
    rho_initial=1.0,
    rho_max=1e6,
    rho_growth=10.0,
    outer_max_iter=50,
    inner_max_iter=500,
    constraint_tol=1e-8,
)
mpec = MPECEstimator(config=mpec_config, verbose=True)
mpec_result = mpec.estimate(panel, utility, problem, transitions)
print()

# ── Compare results ──

print("=" * 60)
print("Parameter Comparison")
print("=" * 60)
param_names = utility.parameter_names
print(f"{'Parameter':<18} {'True':>10} {'NFXP':>10} {'MPEC':>10}")
print("-" * 50)
for i, name in enumerate(param_names):
    true_val = float(true_params[i])
    nfxp_val = float(nfxp_result.parameters[i])
    mpec_val = float(mpec_result.parameters[i])
    print(f"{name:<18} {true_val:>10.5f} {nfxp_val:>10.5f} {mpec_val:>10.5f}")
print()

# ── Policy comparison ──

nfxp_policy = jnp.asarray(nfxp_result.policy)
mpec_policy = jnp.asarray(mpec_result.policy)
max_diff = float(jnp.abs(nfxp_policy - mpec_policy).max())
print(f"Max policy difference (NFXP vs MPEC): {max_diff:.6f}")
print()

# ── Summary tables ──

print("=" * 60)
print("NFXP Summary")
print("=" * 60)
print(nfxp_result.summary())
print()

print("=" * 60)
print("MPEC Summary")
print("=" * 60)
print(mpec_result.summary())
