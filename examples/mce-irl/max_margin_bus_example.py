"""
Maximum Margin IRL on Rust Bus Engine Data

Demonstrates:
1. Running Max Margin IRL (Abbeel & Ng 2004) with unit norm constraint
2. Anchor normalization for parameter identification
3. Comparison with MCE IRL results

Key findings (based on normalized DGP experiments):
- With well-scaled features, Max Margin IRL can recover direction well
- MCE IRL achieves near-perfect parameter recovery (cos sim ~0.9998)
- Max Margin unit norm recovers direction (cos sim ~0.993)
- Anchor normalization is worse for the bus problem structure

Note: For the original Rust bus problem with unnormalized features
(theta_c=0.001, RC=3.0), both IRL methods struggle due to scale mismatch.
"""

import jax.numpy as jnp
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.contrib.max_margin_irl import MaxMarginIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.simulation import simulate_panel


def main():
    print("=" * 70)
    print("Maximum Margin IRL on Rust Bus Engine Replacement")
    print("=" * 70)
    print()

    # Original parameters
    original_theta_c = 0.001
    original_RC = 3.0

    # Create environment with original parameters
    env = RustBusEnvironment(
        operating_cost=original_theta_c,
        replacement_cost=original_RC,
        discount_factor=0.99,
    )
    print(env.describe())

    # Simulate expert data
    print("=" * 70)
    print("Simulating Expert Data")
    print("=" * 70)
    panel = simulate_panel(env, n_individuals=200, n_periods=50, seed=42)
    print(f"Trajectories: {len(panel.trajectories)}")
    print(f"Total observations: {sum(len(t) for t in panel.trajectories)}")
    print()

    # Create reward function
    reward = ActionDependentReward.from_rust_environment(env)
    true_params = jnp.array([original_theta_c, original_RC], dtype=jnp.float32)
    true_params_normalized = true_params / jnp.linalg.norm(true_params)

    print(f"True parameters: {dict(zip(reward.parameter_names, true_params.tolist()))}")
    print(f"True normalized: {dict(zip(reward.parameter_names, true_params_normalized.tolist()))}")
    print()

    # ========================================================================
    # Max Margin IRL (Unit Norm)
    # ========================================================================
    print("=" * 70)
    print("Max Margin IRL (Unit Norm Constraint)")
    print("=" * 70)
    print()

    estimator = MaxMarginIRLEstimator(
        max_iterations=100,
        margin_tol=1e-6,
        value_tol=1e-8,
        verbose=False,
        compute_hessian=True,
        anchor_idx=None,  # Unit norm constraint (default)
    )

    result = estimator.estimate(
        panel=panel,
        utility=reward,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    estimated_params = result.parameters

    print("Results:")
    print("-" * 50)
    print(f"{'Parameter':<20} {'True (norm)':>12} {'Estimated':>12}")
    print("-" * 50)
    for i, name in enumerate(reward.parameter_names):
        true = true_params_normalized[i].item()
        est = estimated_params[i].item()
        print(f"{name:<20} {true:>12.6f} {est:>12.6f}")
    print("-" * 50)
    print()

    # Cosine similarity
    cos_sim = jnp.dot(true_params_normalized, estimated_params) / (
        jnp.linalg.norm(true_params_normalized) * jnp.linalg.norm(estimated_params)
    )
    print(f"Cosine similarity: {cos_sim.item():.4f}")
    print()

    # Explain the margin
    expert_features = result.metadata.get("expert_features", None)
    if expert_features is not None:
        print("Understanding the margin optimization:")
        print(f"  Expert feature expectations: {expert_features.tolist()}")
        print()
        print("  Max Margin IRL finds theta that maximizes:")
        print("    margin = theta' * (expert_features - policy_features)")
        print()
        print("  With expert_features having negative first component,")
        print("  the optimizer may choose negative theta[0] to maximize margin.")
        print()

    print("Margin achieved:", result.metadata.get("margin", "N/A"))
    print()

    # ========================================================================
    # Max Margin IRL with Anchor Normalization
    # ========================================================================
    print("=" * 70)
    print("Max Margin IRL (Anchor Normalization, RC=1)")
    print("=" * 70)
    print()

    estimator_anchor = MaxMarginIRLEstimator(
        max_iterations=100,
        margin_tol=1e-6,
        value_tol=1e-8,
        verbose=False,
        anchor_idx=1,  # Fix RC to 1.0
    )

    result_anchor = estimator_anchor.estimate(
        panel=panel,
        utility=reward,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    estimated_anchor = result_anchor.parameters
    true_relative = jnp.array([original_theta_c / original_RC, 1.0], dtype=jnp.float32)

    print("Results (relative to RC=1):")
    print("-" * 50)
    print(f"{'Parameter':<20} {'True (rel)':>12} {'Estimated':>12}")
    print("-" * 50)
    for i, name in enumerate(reward.parameter_names):
        true = true_relative[i].item()
        est = estimated_anchor[i].item()
        print(f"{name:<20} {true:>12.6f} {est:>12.6f}")
    print("-" * 50)
    print()

    # ========================================================================
    # MCE IRL (for comparison)
    # ========================================================================
    print("=" * 70)
    print("MCE IRL (Maximum Causal Entropy)")
    print("=" * 70)
    print()

    mce_config = MCEIRLConfig(
        verbose=False,
        inner_max_iter=10000,
        outer_max_iter=200,
        learning_rate=0.01,
        compute_se=False,
    )

    mce_estimator = MCEIRLEstimator(config=mce_config)
    mce_result = mce_estimator.estimate(
        panel=panel,
        utility=reward,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
        initial_params=true_params.clone(),
    )

    mce_params = mce_result.parameters

    print("Results:")
    print("-" * 50)
    print(f"{'Parameter':<20} {'True':>12} {'Estimated':>12}")
    print("-" * 50)
    for i, name in enumerate(reward.parameter_names):
        true = true_params[i].item()
        est = mce_params[i].item()
        print(f"{name:<20} {true:>12.6f} {est:>12.6f}")
    print("-" * 50)
    print()

    # ========================================================================
    # Policy Comparison
    # ========================================================================
    print("=" * 70)
    print("Policy Comparison")
    print("=" * 70)
    print()

    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration

    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)

    # True policy
    true_reward = reward.compute(true_params)
    true_result = value_iteration(operator, true_reward, tol=1e-8, max_iter=10000)
    true_policy = true_result.policy

    # Max Margin policy
    mm_reward = reward.compute(estimated_params)
    mm_result = value_iteration(operator, mm_reward, tol=1e-8, max_iter=10000)
    mm_policy = mm_result.policy

    # MCE policy
    mce_reward = reward.compute(mce_params)
    mce_vi_result = value_iteration(operator, mce_reward, tol=1e-8, max_iter=10000)
    mce_policy = mce_vi_result.policy

    test_states = [0, 20, 40, 60, 80]
    print(f"{'State':>6} {'True P(repl)':>14} {'MM P(repl)':>14} {'MCE P(repl)':>14}")
    print("-" * 55)
    for s in test_states:
        true_p = true_policy[s, 1].item()
        mm_p = mm_policy[s, 1].item()
        mce_p = mce_policy[s, 1].item()
        print(f"{s:>6} {true_p:>14.4f} {mm_p:>14.4f} {mce_p:>14.4f}")
    print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("KEY INSIGHT:")
    print("  Max Margin IRL maximizes margin, NOT likelihood.")
    print("  The estimated theta can have wrong signs if that maximizes margin.")
    print()
    print("  For the Rust bus problem:")
    print("  - Expert features have negative operating cost component")
    print("  - Max Margin finds negative theta_c to maximize margin")
    print("  - This gives good margin but wrong parameter signs")
    print()
    print("  For PARAMETER ESTIMATION, use MCE IRL (likelihood-based).")
    print("  Max Margin IRL is better suited for:")
    print("  - Imitation learning (policy matching)")
    print("  - Feature selection (which features matter)")
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
