"""
Maximum Margin IRL on Rust Bus Engine Data

Demonstrates:
1. Running Max Margin IRL (Abbeel & Ng 2004) on the bus engine replacement problem
2. Comparison with MCE IRL results
3. Parameter recovery from simulated data

Note: Max Margin IRL uses a unit norm constraint (||theta||_2 = 1), which means
it identifies reward weights up to scale and direction. This is fundamentally
different from MCE IRL which maximizes likelihood. For problems with features
of very different scales (like the Rust bus problem), Max Margin IRL may not
recover the exact parameter ratio, but should still produce a policy that
matches expert behavior.
"""

import torch
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.max_margin_irl import MaxMarginIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.simulation import simulate_panel


def main():
    print("=" * 70)
    print("Maximum Margin IRL on Rust Bus Engine Replacement")
    print("=" * 70)
    print()

    # Create the Rust bus environment with known true parameters
    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        discount_factor=0.99,
    )
    print(env.describe())

    # Simulate expert data from the environment
    print("=" * 70)
    print("Simulating Expert Data")
    print("=" * 70)
    panel = simulate_panel(env, n_individuals=200, n_periods=50, seed=42)
    print(f"Trajectories: {len(panel.trajectories)}")
    print(f"Total observations: {sum(len(t) for t in panel.trajectories)}")
    print()

    # Compute action frequencies
    keep_count = 0
    replace_count = 0
    for traj in panel.trajectories:
        for action in traj.actions:
            if action.item() == 0:
                keep_count += 1
            else:
                replace_count += 1
    total = keep_count + replace_count
    print(f"Keep actions: {keep_count} ({100*keep_count/total:.1f}%)")
    print(f"Replace actions: {replace_count} ({100*replace_count/total:.1f}%)")
    print()

    # Create action-dependent reward function
    reward = ActionDependentReward.from_rust_environment(env)
    print(f"Reward function: {reward}")
    print(f"Feature matrix shape: {reward.feature_matrix.shape}")
    print(f"Parameter names: {reward.parameter_names}")
    print()

    # Get true parameters
    true_params = env.get_true_parameter_vector()
    print(f"True parameters: {dict(zip(reward.parameter_names, true_params.tolist()))}")
    print()

    # Run Max Margin IRL
    print("=" * 70)
    print("Running Max Margin IRL (Abbeel & Ng 2004)")
    print("=" * 70)
    print()

    estimator = MaxMarginIRLEstimator(
        max_iterations=100,
        margin_tol=1e-6,
        value_tol=1e-8,
        verbose=True,
        compute_hessian=True,
    )

    result = estimator.estimate(
        panel=panel,
        utility=reward,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    print()
    print(result.summary())
    print()

    # Extract and display results
    print("=" * 70)
    print("Max Margin IRL Results")
    print("=" * 70)
    print()

    estimated_params = result.parameters
    print("Parameter Estimates:")
    print("-" * 50)
    for i, name in enumerate(reward.parameter_names):
        est = estimated_params[i]
        true = true_params[i]
        print(f"  {name}: estimated={est:.6f}, true={true:.6f}")
    print()

    # Note: Max Margin IRL identifies parameters up to scale (unit norm constraint)
    # So we compare the ratio
    if len(estimated_params) >= 2 and true_params[1] != 0:
        estimated_ratio = estimated_params[0] / estimated_params[1]
        true_ratio = true_params[0] / true_params[1]
        print(f"Parameter ratio (theta_c / RC):")
        print(f"  Estimated: {estimated_ratio:.6f}")
        print(f"  True:      {true_ratio:.6f}")
        relative_error = abs(estimated_ratio - true_ratio) / abs(true_ratio) * 100
        print(f"  Relative error: {relative_error:.2f}%")
    print()

    # Report margin
    if "margin" in result.metadata:
        print(f"Achieved margin: {result.metadata['margin']:.6f}")
    if "num_violating_policies" in result.metadata:
        print(f"Number of violating policies: {result.metadata['num_violating_policies']}")
    print()

    # Compare with MCE IRL
    print("=" * 70)
    print("Comparison with MCE IRL (Ziebart 2010)")
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
        initial_params=true_params.clone(),  # Warm start for MCE
    )

    mce_params = mce_result.parameters

    print("Method Comparison:")
    print("-" * 70)
    print(f"{'Parameter':<20} {'True':>12} {'Max Margin':>12} {'MCE IRL':>12}")
    print("-" * 70)
    for i, name in enumerate(reward.parameter_names):
        true = true_params[i].item()
        mm = estimated_params[i].item()
        mce = mce_params[i].item()
        print(f"{name:<20} {true:>12.6f} {mm:>12.6f} {mce:>12.6f}")
    print("-" * 70)
    print()

    # Compare ratios
    if len(estimated_params) >= 2 and true_params[1] != 0:
        mm_ratio = estimated_params[0] / estimated_params[1]
        mce_ratio = mce_params[0] / mce_params[1]
        true_ratio = true_params[0] / true_params[1]
        print("Ratio Recovery (theta_c / RC):")
        print("-" * 50)
        print(f"  True:       {true_ratio:.6f}")
        print(f"  Max Margin: {mm_ratio:.6f}")
        print(f"  MCE IRL:    {mce_ratio:.6f} (error: {abs(mce_ratio - true_ratio)/abs(true_ratio)*100:.2f}%)")
    print()

    # Compare policy behavior (more meaningful for Max Margin IRL)
    print("=" * 70)
    print("Policy Comparison (replace probability at key states)")
    print("=" * 70)
    print()

    # Compute policies from each method
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration

    operator = SoftBellmanOperator(env.problem_spec, env.transition_matrices)

    # Max Margin policy
    mm_reward = reward.compute(estimated_params)
    mm_result = value_iteration(operator, mm_reward, tol=1e-8, max_iter=10000)
    mm_policy = mm_result.policy

    # MCE policy
    mce_reward = reward.compute(mce_params)
    mce_result = value_iteration(operator, mce_reward, tol=1e-8, max_iter=10000)
    mce_policy = mce_result.policy

    # True policy
    true_reward = reward.compute(true_params)
    true_result = value_iteration(operator, true_reward, tol=1e-8, max_iter=10000)
    true_policy = true_result.policy

    # Compare at key states
    test_states = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    print(f"{'State':>6} {'True P(repl)':>14} {'MM P(repl)':>14} {'MCE P(repl)':>14}")
    print("-" * 55)
    for s in test_states:
        true_p = true_policy[s, 1].item()
        mm_p = mm_policy[s, 1].item()
        mce_p = mce_policy[s, 1].item()
        print(f"{s:>6} {true_p:>14.4f} {mm_p:>14.4f} {mce_p:>14.4f}")
    print()

    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print("Max Margin IRL identifies reward direction (unit norm constraint),")
    print("which can be useful for feature selection and imitation learning.")
    print("MCE IRL identifies reward magnitude (likelihood-based), which is")
    print("better for structural parameter estimation.")
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
