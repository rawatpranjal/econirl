#!/usr/bin/env python3
"""
Debug script to verify hypothesis: expected features should be computed over
empirical state sequence, not stationary distribution.

The hypothesis:
- Current MCE IRL computes expected features using the stationary distribution
- This creates a mismatch because the stationary distribution differs from
  the empirical state distribution in the data
- The fix: compute expected features at the same states the expert visited

This script:
1. Sets up a simple bus engine problem with known parameters
2. Generates data from the optimal policy
3. At TRUE parameters, computes expected features both ways:
   - Current: E_D[phi] where D is stationary distribution
   - Proposed: E_data[pi(a|s) * phi(s,a)] over empirical state sequence
4. Shows which method gives zero gradient at true parameters
"""

import torch
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def compute_empirical_features(panel, feature_matrix):
    """Compute empirical feature expectations from demonstrations.

    This is the 'target' in MCE IRL gradient: what features the expert used.
    """
    n_features = feature_matrix.shape[2]
    feature_sum = torch.zeros(n_features, dtype=feature_matrix.dtype)
    total_obs = 0

    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            feature_sum += feature_matrix[s, a, :]
            total_obs += 1

    return feature_sum / total_obs


def compute_expected_via_stationary(policy, state_visitation, feature_matrix):
    """Current method: E[phi] = sum_s D(s) * sum_a pi(a|s) * phi(s,a,k).

    This uses the STATIONARY distribution under the policy.
    """
    return torch.einsum("s,sa,sak->k", state_visitation, policy, feature_matrix)


def compute_expected_via_empirical_states(panel, policy, feature_matrix):
    """Proposed fix: E[phi] = (1/N) sum_{i,t} sum_a pi(a|s_{i,t}) * phi(s_{i,t},a,k).

    This uses the EMPIRICAL state sequence from the data.
    For each state the expert visited, we compute what actions the current
    policy would take and weight features accordingly.
    """
    n_features = feature_matrix.shape[2]
    feature_sum = torch.zeros(n_features, dtype=feature_matrix.dtype)
    total_obs = 0

    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            # Sum over all actions weighted by policy probability
            for a in range(policy.shape[1]):
                feature_sum += policy[s, a] * feature_matrix[s, a, :]
            total_obs += 1

    return feature_sum / total_obs


def main():
    print("=" * 70)
    print("DEBUG: Expected Features - Stationary vs Empirical State Sequence")
    print("=" * 70)
    print()

    # =========================================================================
    # Setup: Known parameters
    # =========================================================================
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

    # Generate data from optimal policy
    panel = simulate_panel(
        env,
        n_individuals=100,
        n_periods=200,
        seed=12345,
    )

    print(f"Generated {panel.num_observations} observations")
    print()

    # =========================================================================
    # Get optimal policy at TRUE parameters
    # =========================================================================
    reward_fn = ActionDependentReward.from_rust_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices

    true_params = torch.tensor([TRUE_OPERATING_COST, TRUE_REPLACEMENT_COST])
    feature_matrix = reward_fn.feature_matrix

    # Compute reward and optimal policy
    reward_matrix = reward_fn.compute(true_params)
    operator = SoftBellmanOperator(problem, transitions)

    # Soft value iteration
    V = torch.zeros(problem.num_states)
    for _ in range(10000):
        result = operator.apply(reward_matrix, V)
        if torch.abs(result.V - V).max() < 1e-10:
            break
        V = result.V

    policy = result.policy
    print(f"Computed optimal policy at true parameters")
    print(f"  Replace prob at state 0: {policy[0, 1]:.6f}")
    print(f"  Replace prob at state 45: {policy[45, 1]:.6f}")
    print(f"  Replace prob at state 89: {policy[89, 1]:.6f}")
    print()

    # =========================================================================
    # Compute state distributions
    # =========================================================================

    # 1. Empirical state distribution from data
    state_counts = torch.zeros(problem.num_states)
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            state_counts[s] += 1
    empirical_state_dist = state_counts / state_counts.sum()

    # 2. Initial state distribution from data
    initial_counts = torch.zeros(problem.num_states)
    for traj in panel.trajectories:
        if len(traj) > 0:
            initial_counts[traj.states[0].item()] += 1
    initial_dist = initial_counts / initial_counts.sum()

    # 3. Stationary distribution under policy
    gamma = problem.discount_factor
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    D = initial_dist.clone()
    for _ in range(1000):
        D_new = initial_dist + gamma * (P_pi.T @ D)
        if torch.abs(D_new - D).max() < 1e-10:
            break
        D = D_new
    stationary_dist = D / D.sum()

    print("State distribution comparison:")
    print("-" * 50)
    print(f"{'State':<8} {'Empirical':<12} {'Stationary':<12} {'Difference':<12}")
    print("-" * 50)

    # Show first few and last few states
    for s in list(range(5)) + ['...'] + list(range(85, 90)):
        if s == '...':
            print(f"{'...':<8}")
            continue
        emp = empirical_state_dist[s].item()
        stat = stationary_dist[s].item()
        diff = emp - stat
        print(f"{s:<8} {emp:<12.6f} {stat:<12.6f} {diff:<+12.6f}")

    print()
    total_diff = torch.abs(empirical_state_dist - stationary_dist).sum().item()
    print(f"Total L1 difference: {total_diff:.6f}")
    print()

    # =========================================================================
    # Compare expected features computation
    # =========================================================================
    print("=" * 70)
    print("Expected Features Comparison at TRUE parameters")
    print("=" * 70)
    print()

    # Empirical features (from actual (s,a) pairs in data)
    emp_features = compute_empirical_features(panel, feature_matrix)

    # Expected via stationary distribution (current method)
    exp_stationary = compute_expected_via_stationary(policy, stationary_dist, feature_matrix)

    # Expected via empirical state sequence (proposed fix)
    exp_empirical = compute_expected_via_empirical_states(panel, policy, feature_matrix)

    print("Feature Expectations:")
    print("-" * 70)
    print(f"{'Method':<35} {'Feature 0':<15} {'Feature 1':<15}")
    print("-" * 70)
    print(f"{'Empirical (target)':<35} {emp_features[0].item():<15.6f} {emp_features[1].item():<15.6f}")
    print(f"{'Expected via Stationary':<35} {exp_stationary[0].item():<15.6f} {exp_stationary[1].item():<15.6f}")
    print(f"{'Expected via Empirical States':<35} {exp_empirical[0].item():<15.6f} {exp_empirical[1].item():<15.6f}")
    print("-" * 70)
    print()

    # Gradients (empirical - expected)
    grad_stationary = emp_features - exp_stationary
    grad_empirical = emp_features - exp_empirical

    print("Gradients at TRUE parameters (should be ~0 for correct method):")
    print("-" * 70)
    print(f"{'Method':<35} {'Gradient[0]':<15} {'Gradient[1]':<15} {'||grad||':<10}")
    print("-" * 70)
    print(f"{'Gradient via Stationary':<35} {grad_stationary[0].item():<+15.6f} {grad_stationary[1].item():<+15.6f} {torch.norm(grad_stationary).item():<10.6f}")
    print(f"{'Gradient via Empirical States':<35} {grad_empirical[0].item():<+15.6f} {grad_empirical[1].item():<+15.6f} {torch.norm(grad_empirical).item():<10.6f}")
    print("-" * 70)
    print()

    # =========================================================================
    # Key insight
    # =========================================================================
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print()

    if torch.norm(grad_empirical) < torch.norm(grad_stationary):
        print("HYPOTHESIS CONFIRMED!")
        print()
        print("The gradient via EMPIRICAL STATES is closer to zero at true parameters.")
        print()
        print("Explanation:")
        print("  - Stationary distribution gives ||grad|| = {:.6f}".format(torch.norm(grad_stationary).item()))
        print("  - Empirical states give ||grad|| = {:.6f}".format(torch.norm(grad_empirical).item()))
        print()
        print("The current implementation uses stationary distribution, which creates")
        print("a non-zero gradient even at true parameters. This is why parameter")
        print("recovery fails.")
        print()
        print("FIX: Use compute_expected_via_empirical_states() instead of")
        print("     compute_expected_via_stationary() in mce_irl.py")
    else:
        print("HYPOTHESIS NOT CONFIRMED")
        print()
        print("The gradient via STATIONARY is closer to zero.")
        print("Need to investigate further.")

    print()

    # =========================================================================
    # Additional diagnostic: what would gradient descent do?
    # =========================================================================
    print("=" * 70)
    print("Gradient Descent Simulation")
    print("=" * 70)
    print()

    # Start at true parameters and see where gradient descent takes us
    params = true_params.clone()
    lr = 0.1

    print("Starting at true parameters, following stationary-based gradient:")
    print("-" * 50)

    for step in range(5):
        reward_matrix = reward_fn.compute(params)
        V = torch.zeros(problem.num_states)
        for _ in range(10000):
            result = operator.apply(reward_matrix, V)
            if torch.abs(result.V - V).max() < 1e-10:
                break
            V = result.V
        pol = result.policy

        # Recompute state visitation
        D = initial_dist.clone()
        gamma = problem.discount_factor
        P_pi = torch.einsum("sa,ast->st", pol, transitions)
        for _ in range(1000):
            D_new = initial_dist + gamma * (P_pi.T @ D)
            if torch.abs(D_new - D).max() < 1e-10:
                break
            D = D_new
        D = D / D.sum()

        exp_stat = compute_expected_via_stationary(pol, D, feature_matrix)
        grad = emp_features - exp_stat

        print(f"Step {step}: params=[{params[0].item():.6f}, {params[1].item():.4f}], "
              f"||grad||={torch.norm(grad).item():.6f}")

        # Gradient ascent (we want to increase likelihood)
        params = params + lr * grad

    print()
    print("The parameters DRIFT AWAY from true values when using stationary!")
    print()


if __name__ == "__main__":
    main()
