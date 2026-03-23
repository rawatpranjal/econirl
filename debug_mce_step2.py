#!/usr/bin/env python3
"""Step-by-step debug with more VI iterations and better convergence."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator


def main():
    print("=" * 70)
    print("DEBUG: MCE IRL with better VI convergence")
    print("=" * 70)
    print()

    # True parameters
    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0

    # Create environment with lower discount factor for faster convergence
    env = RustBusEnvironment(
        operating_cost=TRUE_OPERATING_COST,
        replacement_cost=TRUE_REPLACEMENT_COST,
        num_mileage_bins=90,
        discount_factor=0.99,  # Lower gamma for faster VI
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

    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])
    print(f"True parameters: {true_params}")
    print()

    # Test with more VI iterations
    VI_MAX = 50000
    VI_TOL = 1e-10

    print("=" * 70)
    print("Starting at TRUE parameters with better VI:")
    print("=" * 70)
    print()

    params = true_params.clone()
    lr = 0.01  # Smaller learning rate

    for step in range(30):
        reward_matrix = reward_fn.compute(params)

        V = torch.zeros(problem.num_states)
        converged = False
        for i in range(VI_MAX):
            result = operator.apply(reward_matrix, V)
            delta = torch.abs(result.V - V).max().item()
            V = result.V
            if delta < VI_TOL:
                converged = True
                vi_iters = i + 1
                break
        else:
            vi_iters = VI_MAX

        policy = result.policy

        expected_features = compute_expected(panel, policy)
        gradient = empirical_features - expected_features

        obj = 0.5 * torch.sum(gradient ** 2).item()
        grad_norm = torch.norm(gradient).item()

        if step % 5 == 0:
            print(f"Step {step:3d}: params=[{params[0].item():.6f}, {params[1].item():.4f}], "
                  f"obj={obj:.8f}, ||grad||={grad_norm:.6f}, VI iters={vi_iters}")

        params = params + lr * gradient

    print()
    print(f"Final params: [{params[0].item():.6f}, {params[1].item():.4f}]")
    print(f"True params:  [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")

    # Now from random init
    print()
    print("=" * 70)
    print("Starting from zeros with better VI:")
    print("=" * 70)
    print()

    params = torch.tensor([0.0, 0.0])
    lr = 0.01

    for step in range(100):
        reward_matrix = reward_fn.compute(params)

        V = torch.zeros(problem.num_states)
        converged = False
        for i in range(VI_MAX):
            result = operator.apply(reward_matrix, V)
            delta = torch.abs(result.V - V).max().item()
            V = result.V
            if delta < VI_TOL:
                converged = True
                vi_iters = i + 1
                break
        else:
            vi_iters = VI_MAX

        policy = result.policy

        expected_features = compute_expected(panel, policy)
        gradient = empirical_features - expected_features

        obj = 0.5 * torch.sum(gradient ** 2).item()
        grad_norm = torch.norm(gradient).item()

        if step % 10 == 0:
            print(f"Step {step:3d}: params=[{params[0].item():.6f}, {params[1].item():.4f}], "
                  f"obj={obj:.8f}, ||grad||={grad_norm:.6f}, VI iters={vi_iters}")

        params = params + lr * gradient

    print()
    print(f"Final params: [{params[0].item():.6f}, {params[1].item():.4f}]")
    print(f"True params:  [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")


if __name__ == "__main__":
    main()
