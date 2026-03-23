#!/usr/bin/env python3
"""Test MCE IRL with different initializations."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator


def run_optimization(params_init, empirical_features, reward_fn, problem, transitions,
                     operator, feature_matrix, panel, lr=0.01, max_iter=200):
    """Run MCE IRL optimization from given init."""
    params = params_init.clone()

    def compute_expected(panel, policy):
        n_features = feature_matrix.shape[2]
        n_actions = feature_matrix.shape[1]
        feature_sum = torch.zeros(n_features)
        total = 0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                for a in range(n_actions):
                    feature_sum += policy[s, a] * feature_matrix[s, a, :]
                total += 1
        return feature_sum / total

    best_obj = float('inf')
    best_params = params.clone()

    for step in range(max_iter):
        reward_matrix = reward_fn.compute(params)

        V = torch.zeros(problem.num_states)
        for i in range(50000):
            result = operator.apply(reward_matrix, V)
            if torch.abs(result.V - V).max() < 1e-10:
                break
            V = result.V

        policy = result.policy
        expected_features = compute_expected(panel, policy)
        gradient = empirical_features - expected_features

        obj = 0.5 * torch.sum(gradient ** 2).item()

        if obj < best_obj:
            best_obj = obj
            best_params = params.clone()

        if torch.norm(gradient) < 1e-6:
            break

        params = params + lr * gradient

    return best_params, best_obj


def main():
    print("=" * 70)
    print("DEBUG: MCE IRL with different initializations")
    print("=" * 70)
    print()

    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0

    env = RustBusEnvironment(
        operating_cost=TRUE_OPERATING_COST,
        replacement_cost=TRUE_REPLACEMENT_COST,
        num_mileage_bins=90,
        discount_factor=0.99,
        scale_parameter=1.0,
        seed=42,
    )

    panel = simulate_panel(env, n_individuals=100, n_periods=200, seed=12345)

    reward_fn = ActionDependentReward.from_rust_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    feature_matrix = reward_fn.feature_matrix
    operator = SoftBellmanOperator(problem, transitions)

    # Compute empirical features
    n_features = feature_matrix.shape[2]
    feature_sum = torch.zeros(n_features)
    total = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            feature_sum += feature_matrix[s, a, :]
            total += 1
    empirical_features = feature_sum / total

    print(f"Empirical features: {empirical_features}")
    print(f"True params: [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")
    print()

    # Test different initializations
    inits = [
        ("Zeros", torch.tensor([0.0, 0.0])),
        ("Small positive", torch.tensor([0.01, 1.0])),
        ("Closer to true", torch.tensor([0.001, 2.0])),
        ("Near true", torch.tensor([0.0005, 2.5])),
        ("True params", torch.tensor([0.001, 3.0])),
        ("Random 1", torch.tensor([0.005, 5.0])),
        ("Random 2", torch.tensor([0.002, 4.0])),
    ]

    print(f"{'Initialization':<20} {'Final params':<30} {'Final obj':<15} {'Error':<15}")
    print("-" * 80)

    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])

    for name, init in inits:
        final_params, final_obj = run_optimization(
            init, empirical_features, reward_fn, problem, transitions,
            operator, feature_matrix, panel, lr=0.01, max_iter=200
        )
        error = torch.norm(final_params - true_params).item()
        print(f"{name:<20} [{final_params[0].item():.6f}, {final_params[1].item():.4f}]"
              f"   {final_obj:<15.8f} {error:<15.6f}")

    print()
    print("=" * 70)
    print("With larger learning rate and more iters from good init:")
    print("=" * 70)
    print()

    final_params, final_obj = run_optimization(
        torch.tensor([0.001, 2.0]), empirical_features, reward_fn, problem, transitions,
        operator, feature_matrix, panel, lr=0.1, max_iter=500
    )
    print(f"Final: [{final_params[0].item():.6f}, {final_params[1].item():.4f}], obj={final_obj:.8f}")
    print(f"True:  [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")
    error = torch.norm(final_params - true_params).item()
    print(f"Error: {error:.6f}")


if __name__ == "__main__":
    main()
