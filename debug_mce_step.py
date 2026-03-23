#!/usr/bin/env python3
"""Step-by-step debug of MCE IRL optimization."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator


def main():
    print("=" * 70)
    print("DEBUG: Step-by-step MCE IRL")
    print("=" * 70)
    print()

    # True parameters
    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0

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
    panel = simulate_panel(env, n_individuals=100, n_periods=200, seed=12345)

    reward_fn = ActionDependentReward.from_rust_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    feature_matrix = reward_fn.feature_matrix

    operator = SoftBellmanOperator(problem, transitions)

    # Compute empirical features
    def compute_empirical(panel):
        n_features = feature_matrix.shape[2]
        feature_sum = torch.zeros(n_features)
        total = 0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                a = traj.actions[t].item()
                feature_sum += feature_matrix[s, a, :]
                total += 1
        return feature_sum / total

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

    empirical_features = compute_empirical(panel)
    print(f"Empirical features: {empirical_features}")
    print()

    # Get initial parameters
    params = reward_fn.get_initial_parameters()
    print(f"Initial parameters: {params}")

    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])
    print(f"True parameters: {true_params}")
    print()

    # Run a few steps manually
    lr = 0.1
    print("Manual gradient descent steps:")
    print("-" * 70)

    for step in range(10):
        reward_matrix = reward_fn.compute(params)

        # Soft value iteration
        V = torch.zeros(problem.num_states)
        converged = False
        for i in range(10000):
            result = operator.apply(reward_matrix, V)
            delta = torch.abs(result.V - V).max().item()
            V = result.V
            if delta < 1e-8:
                converged = True
                break

        policy = result.policy

        expected_features = compute_expected(panel, policy)
        gradient = empirical_features - expected_features

        obj = 0.5 * torch.sum(gradient ** 2).item()
        grad_norm = torch.norm(gradient).item()

        print(f"Step {step}: params=[{params[0].item():.6f}, {params[1].item():.4f}], "
              f"obj={obj:.6f}, ||grad||={grad_norm:.6f}, VI converged={converged}")
        print(f"          emp=[{empirical_features[0].item():.4f}, {empirical_features[1].item():.4f}], "
              f"exp=[{expected_features[0].item():.4f}, {expected_features[1].item():.4f}]")

        params = params + lr * gradient

    print()
    print("=" * 70)
    print("Now try starting at TRUE parameters:")
    print("=" * 70)
    print()

    params = true_params.clone()
    for step in range(10):
        reward_matrix = reward_fn.compute(params)

        V = torch.zeros(problem.num_states)
        converged = False
        for i in range(10000):
            result = operator.apply(reward_matrix, V)
            delta = torch.abs(result.V - V).max().item()
            V = result.V
            if delta < 1e-8:
                converged = True
                break

        policy = result.policy

        expected_features = compute_expected(panel, policy)
        gradient = empirical_features - expected_features

        obj = 0.5 * torch.sum(gradient ** 2).item()
        grad_norm = torch.norm(gradient).item()

        print(f"Step {step}: params=[{params[0].item():.6f}, {params[1].item():.4f}], "
              f"obj={obj:.6f}, ||grad||={grad_norm:.6f}, VI converged={converged}")

        params = params + lr * gradient


if __name__ == "__main__":
    main()
