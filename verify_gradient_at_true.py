#!/usr/bin/env python3
"""
Verify that gradient at true parameters is near zero with the fix.
This is the key test - if gradient is small at truth, the estimator is consistent.
"""

import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.estimation.mce_irl import MCEIRLEstimator


def compute_gradient_at_params(params, panel, reward_fn, problem, transitions):
    """Compute the MCE IRL gradient at given parameters."""
    feature_matrix = reward_fn.feature_matrix
    n_features = feature_matrix.shape[2]
    n_actions = feature_matrix.shape[1]

    operator = SoftBellmanOperator(problem, transitions)

    # Compute optimal policy
    reward_matrix = reward_fn.compute(params)
    V = torch.zeros(problem.num_states)
    for _ in range(50000):
        result = operator.apply(reward_matrix, V)
        if torch.abs(result.V - V).max() < 1e-10:
            break
        V = result.V
    policy = result.policy

    # Empirical features
    emp_sum = torch.zeros(n_features)
    total = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            emp_sum += feature_matrix[s, a, :]
            total += 1
    empirical = emp_sum / total

    # Expected features via empirical states (THE FIX)
    exp_sum = torch.zeros(n_features)
    total = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            for a in range(n_actions):
                exp_sum += policy[s, a] * feature_matrix[s, a, :]
            total += 1
    expected_fix = exp_sum / total

    # Expected features via stationary (OLD METHOD)
    gamma = problem.discount_factor
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    initial_dist = torch.zeros(problem.num_states)
    for traj in panel.trajectories:
        if len(traj) > 0:
            initial_dist[traj.states[0].item()] += 1
    initial_dist = initial_dist / initial_dist.sum()

    D = initial_dist.clone()
    for _ in range(1000):
        D_new = initial_dist + gamma * (P_pi.T @ D)
        if torch.abs(D_new - D).max() < 1e-10:
            break
        D = D_new
    D = D / D.sum()

    expected_old = torch.einsum("s,sa,sak->k", D, policy, feature_matrix)

    gradient_fix = empirical - expected_fix
    gradient_old = empirical - expected_old

    return {
        'empirical': empirical,
        'expected_fix': expected_fix,
        'expected_old': expected_old,
        'gradient_fix': gradient_fix,
        'gradient_old': gradient_old,
        'grad_norm_fix': torch.norm(gradient_fix).item(),
        'grad_norm_old': torch.norm(gradient_old).item(),
    }


def main():
    print("=" * 70)
    print("VERIFICATION: Gradient at True Parameters")
    print("=" * 70)
    print()

    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0
    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])

    # Test with different sample sizes
    for n_individuals in [50, 100, 200, 500]:
        env = RustBusEnvironment(
            operating_cost=TRUE_OPERATING_COST,
            replacement_cost=TRUE_REPLACEMENT_COST,
            num_mileage_bins=90,
            discount_factor=0.99,
            scale_parameter=1.0,
            seed=42,
        )

        panel = simulate_panel(env, n_individuals=n_individuals, n_periods=200, seed=12345)

        reward_fn = ActionDependentReward.from_rust_environment(env)
        problem = env.problem_spec
        transitions = env.transition_matrices

        result = compute_gradient_at_params(
            true_params, panel, reward_fn, problem, transitions
        )

        print(f"N = {n_individuals * 200:,} observations:")
        print(f"  ||grad|| with FIX:  {result['grad_norm_fix']:.6f}")
        print(f"  ||grad|| with OLD:  {result['grad_norm_old']:.6f}")
        print(f"  Improvement ratio:  {result['grad_norm_old'] / result['grad_norm_fix']:.1f}x")
        print()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("If ||grad|| with FIX is consistently smaller than with OLD,")
    print("the fix is working. The remaining gradient is sampling noise.")
    print()
    print("For consistent estimation, ||grad|| should shrink as N grows.")


if __name__ == "__main__":
    main()
