#!/usr/bin/env python3
"""Test that the MCE IRL fix works - parameters should now be recovered."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def main():
    print("=" * 70)
    print("TEST: MCE IRL Parameter Recovery (after fix)")
    print("=" * 70)
    print()

    # True parameters
    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0

    print(f"True parameters:")
    print(f"  Operating cost:    {TRUE_OPERATING_COST}")
    print(f"  Replacement cost:  {TRUE_REPLACEMENT_COST}")
    print()

    # Create environment
    env = RustBusEnvironment(
        operating_cost=TRUE_OPERATING_COST,
        replacement_cost=TRUE_REPLACEMENT_COST,
        num_mileage_bins=90,
        discount_factor=0.9999,
        scale_parameter=1.0,
        seed=42,
    )

    # Generate data
    panel = simulate_panel(
        env,
        n_individuals=100,
        n_periods=200,
        seed=12345,
    )

    print(f"Generated {panel.num_observations} observations")
    print()

    # Set up estimation
    reward_fn = ActionDependentReward.from_rust_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    # Create estimator
    config = MCEIRLConfig(
        verbose=True,
        outer_max_iter=200,
        learning_rate=0.1,
        outer_tol=1e-6,
        compute_se=False,  # Skip SE for speed
    )
    estimator = MCEIRLEstimator(config=config)

    # Run estimation
    print("Running MCE IRL estimation...")
    print("-" * 50)
    result = estimator.estimate(
        panel=panel,
        utility=reward_fn,
        problem=problem,
        transitions=transitions,
    )

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()

    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])
    est_params = result.parameters

    print(f"{'Parameter':<20} {'True':<12} {'Estimated':<12} {'Error':<12} {'% Error':<10}")
    print("-" * 70)

    for i, name in enumerate(["Operating Cost", "Replacement Cost"]):
        true_val = true_params[i].item()
        est_val = est_params[i].item()
        error = est_val - true_val
        pct_error = (error / true_val) * 100 if true_val != 0 else float('inf')
        print(f"{name:<20} {true_val:<12.6f} {est_val:<12.6f} {error:<+12.6f} {pct_error:<+10.1f}%")

    print()
    print(f"Log-likelihood: {result.log_likelihood:.2f}")
    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.num_iterations}")

    # Check if recovery is good
    param_error = torch.norm(est_params - true_params).item()
    print()
    if param_error < 0.5:
        print("SUCCESS: Parameters recovered within tolerance!")
    else:
        print(f"FAILURE: Parameter error = {param_error:.4f} (threshold: 0.5)")


if __name__ == "__main__":
    main()
