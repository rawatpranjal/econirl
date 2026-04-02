"""Generate outputs for the counterfactual documentation page.

Fits NFXP, CCP, NNES, and TDCCP on the Rust bus dataset, then runs
all four counterfactual types on each and prints the results.
"""
import numpy as np
import jax.numpy as jnp
import torch
from econirl import NFXP, CCP, NNES, TDCCP
from econirl.datasets import load_rust_bus
from econirl.simulation import (
    CounterfactualType,
    state_extrapolation,
    counterfactual_transitions,
    counterfactual_policy,
    discount_factor_change,
    welfare_decomposition,
    counterfactual,
    elasticity_analysis,
)


def to_jax(x):
    """Convert torch tensor or numpy array to JAX array."""
    if isinstance(x, torch.Tensor):
        return jnp.array(x.detach().numpy())
    return jnp.array(x)

df = load_rust_bus()

# Fit four estimators
print("=" * 70)
print("FITTING ESTIMATORS")
print("=" * 70)

models = {}
for name, cls, kwargs in [
    ("NFXP", NFXP, {}),
    ("CCP", CCP, {"num_policy_iterations": 1}),
    ("NNES", NNES, {"bellman": "npl", "hidden_dim": 32, "n_outer_iterations": 3}),
    ("TDCCP", TDCCP, {"hidden_dim": 64, "avi_iterations": 20}),
]:
    print(f"\nFitting {name}...")
    model = cls(n_states=90, discount=0.9999, **kwargs)
    model.fit(df, state="mileage_bin", action="replaced", id="bus_id")
    models[name] = model
    print(f"  theta_c = {model.params_['theta_c']:.6f}, RC = {model.params_['RC']:.4f}")
    print(f"  Converged: {model.converged_}")

# Use NFXP as the primary model for detailed examples
model = models["NFXP"]
result = model._result
problem = model._problem
utility = model._utility_fn
transition_tensor = to_jax(model._build_transition_tensor(model.transitions_))

print("\n" + "=" * 70)
print("TYPE 1: STATE-VALUE EXTRAPOLATION")
print("=" * 70)

for name, m in models.items():
    r = m._result
    p = m._problem
    t = to_jax(m._build_transition_tensor(m.transitions_))
    mapping = {s: max(0, s - 10) for s in range(p.num_states)}
    cf = state_extrapolation(r, mapping, p, t)
    print(f"\n{name}:")
    print(f"  Average welfare change: {cf.welfare_change:.4f}")
    print(f"  Max policy shift: {float(jnp.abs(cf.policy_change).max()):.4f}")
    print(f"  P(replace|s=50) baseline: {float(r.policy[50, 1]):.4f}")
    print(f"  P(replace|s=50) counterfactual: {float(cf.counterfactual_policy[50, 1]):.4f}")

print("\n" + "=" * 70)
print("TYPE 2: ENVIRONMENT CHANGE (slower depreciation)")
print("=" * 70)

for name, m in models.items():
    r = m._result
    p = m._problem
    u = m._utility_fn
    t_base = to_jax(m._build_transition_tensor(m.transitions_))
    # Slower depreciation: shrink each row's mass toward lower increments
    # Shift 30% of transition probability one bin lower
    new_t = jnp.array(t_base, copy=True)
    for a in range(p.num_actions):
        for s in range(p.num_states):
            row = new_t[a, s, :]
            shifted = jnp.zeros_like(row)
            shifted = shifted.at[0].set(row[0])
            for sp in range(1, p.num_states):
                shifted = shifted.at[sp - 1].add(row[sp] * 0.3)
                shifted = shifted.at[sp].add(row[sp] * 0.7)
            new_t = new_t.at[a, s, :].set(shifted / shifted.sum())
    cf = counterfactual_transitions(r, new_t, u, p, t_base)
    print(f"\n{name}:")
    print(f"  Welfare change: {cf.welfare_change:.4f}")
    print(f"  P(replace|s=50) baseline: {float(r.policy[50, 1]):.4f}")
    print(f"  P(replace|s=50) counterfactual: {float(cf.counterfactual_policy[50, 1]):.4f}")

print("\n" + "=" * 70)
print("TYPE 3a: REWARD PARAMETER CHANGE (double RC)")
print("=" * 70)

for name, m in models.items():
    r = m._result
    p = m._problem
    u = m._utility_fn
    t = to_jax(m._build_transition_tensor(m.transitions_))
    new_params = jnp.array(r.parameters)
    # Double replacement cost (index 1)
    new_params = new_params.at[1].set(new_params[1] * 2.0)
    cf = counterfactual_policy(r, new_params, u, p, t)
    print(f"\n{name}:")
    print(f"  Baseline RC: {float(r.parameters[1]):.4f}")
    print(f"  Counterfactual RC: {float(new_params[1]):.4f}")
    print(f"  Welfare change: {cf.welfare_change:.4f}")
    print(f"  P(replace|s=50) baseline: {float(r.policy[50, 1]):.4f}")
    print(f"  P(replace|s=50) counterfactual: {float(cf.counterfactual_policy[50, 1]):.4f}")

print("\n" + "=" * 70)
print("TYPE 3b: DISCOUNT FACTOR CHANGE (beta 0.9999 -> 0.99)")
print("=" * 70)

for name, m in models.items():
    r = m._result
    p = m._problem
    u = m._utility_fn
    t = to_jax(m._build_transition_tensor(m.transitions_))
    cf = discount_factor_change(r, 0.99, u, p, t)
    print(f"\n{name}:")
    print(f"  Welfare change: {cf.welfare_change:.4f}")
    print(f"  P(replace|s=50) baseline: {float(r.policy[50, 1]):.4f}")
    print(f"  P(replace|s=50) counterfactual: {float(cf.counterfactual_policy[50, 1]):.4f}")

print("\n" + "=" * 70)
print("TYPE 4: WELFARE DECOMPOSITION (double RC + slower depreciation)")
print("=" * 70)

for name, m in models.items():
    r = m._result
    p = m._problem
    u = m._utility_fn
    t_base = to_jax(m._build_transition_tensor(m.transitions_))
    new_params = jnp.array(r.parameters)
    new_params = new_params.at[1].set(new_params[1] * 2.0)
    # Reuse the slower depreciation transitions from Type 2
    new_t = jnp.array(t_base, copy=True)
    for a in range(p.num_actions):
        for s in range(p.num_states):
            row = new_t[a, s, :]
            shifted = jnp.zeros_like(row)
            shifted = shifted.at[0].set(row[0])
            for sp in range(1, p.num_states):
                shifted = shifted.at[sp - 1].add(row[sp] * 0.3)
                shifted = shifted.at[sp].add(row[sp] * 0.7)
            new_t = new_t.at[a, s, :].set(shifted / shifted.sum())
    decomp = welfare_decomposition(r, u, p, t_base, new_parameters=new_params, new_transitions=new_t)
    print(f"\n{name}:")
    print(f"  Total welfare change: {decomp['total_welfare_change']:.4f}")
    print(f"  Reward channel: {decomp['reward_channel']:.4f}")
    print(f"  Transition channel: {decomp['transition_channel']:.4f}")
    print(f"  Interaction: {decomp['interaction_effect']:.4f}")

print("\n" + "=" * 70)
print("ELASTICITY ANALYSIS (NFXP)")
print("=" * 70)

elast = elasticity_analysis(
    result, utility, problem, transition_tensor,
    parameter_name="RC",
    pct_changes=[-0.50, -0.25, 0.25, 0.50, 1.00],
)
print(f"Parameter: {elast['parameter']}")
print(f"Baseline value: {elast['baseline_value']:.4f}")
for pct, pol, wel in zip(
    elast["pct_changes"], elast["policy_changes"], elast["welfare_changes"]
):
    print(f"  {pct:+.0%} -> policy change: {pol:.4f}, welfare change: {wel:.4f}")
if "policy_elasticity" in elast:
    print(f"  Policy elasticity: {elast['policy_elasticity']:.4f}")
    print(f"  Welfare elasticity: {elast['welfare_elasticity']:.4f}")

print("\n" + "=" * 70)
print("UNIFIED DISPATCHER DEMO")
print("=" * 70)

cf1 = counterfactual(
    result=result, utility=utility, problem=problem, transitions=transition_tensor,
    state_mapping={s: max(0, s - 10) for s in range(problem.num_states)},
)
print(f"Type 1 dispatched: type={cf1.counterfactual_type.name}, welfare={cf1.welfare_change:.4f}")

new_params = jnp.array(result.parameters)
new_params = new_params.at[1].set(new_params[1] * 2.0)
cf3 = counterfactual(
    result=result, utility=utility, problem=problem, transitions=transition_tensor,
    new_parameters=new_params,
)
print(f"Type 3 dispatched: type={cf3.counterfactual_type.name}, welfare={cf3.welfare_change:.4f}")

cf_beta = counterfactual(
    result=result, utility=utility, problem=problem, transitions=transition_tensor,
    new_discount=0.99,
)
print(f"Type 3 (beta) dispatched: type={cf_beta.counterfactual_type.name}, welfare={cf_beta.welfare_change:.4f}")
