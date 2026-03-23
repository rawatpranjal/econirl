#!/usr/bin/env python3
"""Test MCE IRL with proper initialization."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def main():
    print("=" * 70)
    print("TEST: MCE IRL with Good Initialization")
    print("=" * 70)
    print()

    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0
    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])

    # Use lower discount factor for faster convergence
    env = RustBusEnvironment(
        operating_cost=TRUE_OPERATING_COST,
        replacement_cost=TRUE_REPLACEMENT_COST,
        num_mileage_bins=90,
        discount_factor=0.99,  # Lower for faster VI
        scale_parameter=1.0,
        seed=42,
    )

    panel = simulate_panel(env, n_individuals=100, n_periods=200, seed=12345)

    print(f"True parameters: [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")
    print(f"Observations: {panel.num_observations}")
    print()

    reward_fn = ActionDependentReward.from_rust_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    config = MCEIRLConfig(
        verbose=True,
        outer_max_iter=300,
        learning_rate=0.05,
        outer_tol=1e-6,
        inner_tol=1e-10,
        inner_max_iter=50000,
        compute_se=False,
    )
    estimator = MCEIRLEstimator(config=config)

    # Test with different initializations
    inits = [
        ("Zeros", torch.tensor([0.0, 0.0])),
        ("Near true", torch.tensor([0.0005, 2.5])),
        ("True params", true_params.clone()),
    ]

    for name, init_params in inits:
        print(f"\n{'='*70}")
        print(f"Testing with initialization: {name}")
        print(f"Init: [{init_params[0].item():.6f}, {init_params[1].item():.4f}]")
        print("=" * 70)

        result = estimator.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions,
            initial_params=init_params,
        )

        est_params = result.parameters
        error = torch.norm(est_params - true_params).item()

        print()
        print(f"Final:  [{est_params[0].item():.6f}, {est_params[1].item():.4f}]")
        print(f"True:   [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")
        print(f"Error:  {error:.6f}")
        print(f"Feature diff: {result.metadata.get('feature_difference', 'N/A')}")

        if error < 0.1:
            print("STATUS: GOOD RECOVERY")
        elif error < 0.5:
            print("STATUS: APPROXIMATE RECOVERY")
        else:
            print("STATUS: POOR RECOVERY")


if __name__ == "__main__":
    main()
