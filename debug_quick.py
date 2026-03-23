#!/usr/bin/env python3
"""Quick test of MCE IRL from good initialization."""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator


def main():
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

    def compute_expected(panel, policy):
        feature_sum = torch.zeros(n_features)
        total = 0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                for a in range(2):
                    feature_sum += policy[s, a] * feature_matrix[s, a, :]
                total += 1
        return feature_sum / total

    # Start near true params
    params = torch.tensor([0.0005, 2.5])
    lr = 0.05
    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])

    print("Optimizing from [0.0005, 2.5]:")
    print("-" * 60)

    for step in range(50):
        reward_matrix = reward_fn.compute(params)

        V = torch.zeros(problem.num_states)
        for i in range(10000):
            result = operator.apply(reward_matrix, V)
            if torch.abs(result.V - V).max() < 1e-10:
                break
            V = result.V

        policy = result.policy
        expected = compute_expected(panel, policy)
        gradient = empirical_features - expected
        obj = 0.5 * torch.sum(gradient ** 2).item()
        grad_norm = torch.norm(gradient).item()

        if step % 10 == 0:
            error = torch.norm(params - true_params).item()
            print(f"Step {step:2d}: params=[{params[0].item():.6f}, {params[1].item():.4f}], "
                  f"obj={obj:.8f}, error={error:.6f}")

        params = params + lr * gradient

    print()
    print(f"Final:  [{params[0].item():.6f}, {params[1].item():.4f}]")
    print(f"True:   [{TRUE_OPERATING_COST}, {TRUE_REPLACEMENT_COST}]")
    error = torch.norm(params - true_params).item()
    print(f"Error:  {error:.6f}")

    if error < 0.1:
        print("\nSUCCESS: Parameters recovered!")
    else:
        print("\nNOT CONVERGED: Need more iterations or better init")


if __name__ == "__main__":
    main()
