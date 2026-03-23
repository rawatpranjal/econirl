#!/usr/bin/env python3
"""Test MCE IRL with smaller learning rate."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def main():
    print("=" * 70)
    print("TEST: MCE IRL with Smaller Learning Rate")
    print("=" * 70)
    print()

    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0
    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])

    env = RustBusEnvironment(
        operating_cost=TRUE_OPERATING_COST,
        replacement_cost=TRUE_REPLACEMENT_COST,
        num_mileage_bins=90,
        discount_factor=0.99,
        scale_parameter=1.0,
        seed=42,
    )

    panel = simulate_panel(env, n_individuals=200, n_periods=200, seed=12345)

    print(f"True parameters: [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")
    print(f"Observations: {panel.num_observations}")
    print()

    reward_fn = ActionDependentReward.from_rust_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    # Much smaller learning rate
    config = MCEIRLConfig(
        verbose=True,
        outer_max_iter=500,
        learning_rate=0.005,  # Much smaller
        outer_tol=1e-8,
        inner_tol=1e-10,
        inner_max_iter=50000,
        compute_se=False,
    )
    estimator = MCEIRLEstimator(config=config)

    # Start near true
    init_params = torch.tensor([0.0008, 2.8])
    print(f"Init: [{init_params[0].item():.6f}, {init_params[1].item():.4f}]")
    print()

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
    print()

    # Relative error
    rel_error_oc = abs(est_params[0].item() - TRUE_OPERATING_COST) / TRUE_OPERATING_COST * 100
    rel_error_rc = abs(est_params[1].item() - TRUE_REPLACEMENT_COST) / TRUE_REPLACEMENT_COST * 100
    print(f"Relative errors:")
    print(f"  Operating cost:   {rel_error_oc:.1f}%")
    print(f"  Replacement cost: {rel_error_rc:.1f}%")


if __name__ == "__main__":
    main()
