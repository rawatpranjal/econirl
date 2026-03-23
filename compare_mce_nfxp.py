#!/usr/bin/env python3
"""Compare MCE IRL (with fix) to NFXP."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.nfxp import NFXPEstimator


def main():
    print("=" * 70)
    print("COMPARISON: MCE IRL vs NFXP")
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

    problem = env.problem_spec
    transitions = env.transition_matrices

    # NFXP uses LinearUtility
    utility_nfxp = LinearUtility.from_environment(env)

    # MCE IRL uses ActionDependentReward
    reward_mce = ActionDependentReward.from_rust_environment(env)

    # Run NFXP
    print("Running NFXP...")
    print("-" * 50)
    nfxp = NFXPEstimator(
        se_method="asymptotic",
        optimizer="L-BFGS-B",
        inner_tol=1e-10,
        outer_tol=1e-6,
        verbose=True,
    )
    result_nfxp = nfxp.estimate(
        panel=panel,
        utility=utility_nfxp,
        problem=problem,
        transitions=transitions,
    )
    print()

    # Run MCE IRL with good init (from NFXP result)
    print("Running MCE IRL (init from NFXP)...")
    print("-" * 50)
    mce_config = MCEIRLConfig(
        verbose=True,
        outer_max_iter=500,
        learning_rate=0.01,
        outer_tol=1e-8,
        inner_tol=1e-10,
        inner_max_iter=50000,
        compute_se=False,
    )
    mce = MCEIRLEstimator(config=mce_config)
    result_mce = mce.estimate(
        panel=panel,
        utility=reward_mce,
        problem=problem,
        transitions=transitions,
        initial_params=result_nfxp.parameters.clone(),
    )
    print()

    # Compare
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'Method':<15} {'Operating Cost':<18} {'Replacement Cost':<18} {'Error':<10}")
    print("-" * 70)
    print(f"{'True':<15} {TRUE_OPERATING_COST:<18.6f} {TRUE_REPLACEMENT_COST:<18.4f}")

    nfxp_error = torch.norm(result_nfxp.parameters - true_params).item()
    print(f"{'NFXP':<15} {result_nfxp.parameters[0].item():<18.6f} "
          f"{result_nfxp.parameters[1].item():<18.4f} {nfxp_error:<10.6f}")

    mce_error = torch.norm(result_mce.parameters - true_params).item()
    print(f"{'MCE IRL':<15} {result_mce.parameters[0].item():<18.6f} "
          f"{result_mce.parameters[1].item():<18.4f} {mce_error:<10.6f}")

    print()
    print("Log-likelihoods:")
    print(f"  NFXP:    {result_nfxp.log_likelihood:.2f}")
    print(f"  MCE IRL: {result_mce.log_likelihood:.2f}")


if __name__ == "__main__":
    main()
