#!/usr/bin/env python3
"""Grid search analysis to understand the MCE IRL objective landscape."""

import numpy as np
import torch
import matplotlib.pyplot as plt

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.preferences.action_reward import ActionDependentReward


def compute_objectives(params, panel, reward_fn, problem, transitions, initial_dist):
    """Compute both feature matching objective and log-likelihood."""
    operator = SoftBellmanOperator(problem, transitions)
    reward_matrix = reward_fn.compute(params)

    V = torch.zeros(problem.num_states, dtype=params.dtype)
    for _ in range(10000):
        result = operator.apply(reward_matrix, V)
        V_new = result.V
        if torch.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    policy = result.policy

    gamma = problem.discount_factor
    P_pi = torch.einsum("sa,ast->st", policy, transitions)
    D = initial_dist.clone()
    rho0 = initial_dist.clone()
    for _ in range(5000):
        D_new = rho0 + gamma * (P_pi.T @ D)
        if torch.abs(D_new - D).max() < 1e-10:
            break
        D = D_new
    D = D / D.sum()

    fm = reward_fn.feature_matrix
    total_obs = sum(len(t) for t in panel.trajectories)
    emp_sum = torch.zeros(fm.shape[2])
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s, a = traj.states[t].item(), traj.actions[t].item()
            emp_sum += fm[s, a, :]
    emp_features = emp_sum / total_obs

    exp_features = torch.einsum("s,sa,sak->k", D, policy, fm)
    feature_diff = (emp_features - exp_features).norm().item()

    log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
    ll = sum(
        log_probs[traj.states[t].item(), traj.actions[t].item()].item()
        for traj in panel.trajectories
        for t in range(len(traj))
    )

    return feature_diff, ll


def main():
    print("=" * 70)
    print(" GRID SEARCH: Understanding MCE IRL Objective Landscape")
    print("=" * 70)

    TRUE_THETA_C = 0.001
    TRUE_RC = 3.0
    DISCOUNT = 0.99

    env = RustBusEnvironment(
        operating_cost=TRUE_THETA_C,
        replacement_cost=TRUE_RC,
        num_mileage_bins=90,
        discount_factor=DISCOUNT,
    )

    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=42)
    reward = ActionDependentReward.from_rust_environment(env)

    n_states = env.num_states
    counts = torch.zeros(n_states)
    for traj in panel.trajectories:
        if len(traj) > 0:
            counts[traj.states[0].item()] += 1
    initial_dist = counts / counts.sum()

    # Grid search
    theta_c_values = np.linspace(0.0001, 0.003, 15)
    rc_values = np.linspace(1.0, 5.0, 15)

    feature_diffs = np.zeros((len(rc_values), len(theta_c_values)))
    log_likelihoods = np.zeros((len(rc_values), len(theta_c_values)))

    print("\nRunning grid search...")
    for i, rc in enumerate(rc_values):
        for j, theta_c in enumerate(theta_c_values):
            params = torch.tensor([theta_c, rc], dtype=torch.float32)
            fd, ll = compute_objectives(
                params, panel, reward, env.problem_spec,
                env.transition_matrices, initial_dist
            )
            feature_diffs[i, j] = fd
            log_likelihoods[i, j] = ll
        print(f"  RC = {rc:.2f} done")

    # Find optima
    fm_min_idx = np.unravel_index(feature_diffs.argmin(), feature_diffs.shape)
    ll_max_idx = np.unravel_index(log_likelihoods.argmax(), log_likelihoods.shape)

    fm_opt_theta_c = theta_c_values[fm_min_idx[1]]
    fm_opt_rc = rc_values[fm_min_idx[0]]
    ll_opt_theta_c = theta_c_values[ll_max_idx[1]]
    ll_opt_rc = rc_values[ll_max_idx[0]]

    print(f"\n" + "=" * 70)
    print(" RESULTS")
    print("=" * 70)

    print(f"\nTrue parameters:          theta_c = {TRUE_THETA_C:.6f}, RC = {TRUE_RC:.2f}")
    print(f"Feature matching optimum: theta_c = {fm_opt_theta_c:.6f}, RC = {fm_opt_rc:.2f}")
    print(f"Log-likelihood optimum:   theta_c = {ll_opt_theta_c:.6f}, RC = {ll_opt_rc:.2f}")

    print(f"\nAt true parameters:")
    true_idx = (
        np.abs(rc_values - TRUE_RC).argmin(),
        np.abs(theta_c_values - TRUE_THETA_C).argmin()
    )
    print(f"  Feature diff: {feature_diffs[true_idx]:.6f}")
    print(f"  Log-lik:      {log_likelihoods[true_idx]:.2f}")

    print(f"\nAt feature matching optimum:")
    print(f"  Feature diff: {feature_diffs[fm_min_idx]:.6f}")
    print(f"  Log-lik:      {log_likelihoods[fm_min_idx]:.2f}")

    print(f"\nAt log-likelihood optimum:")
    print(f"  Feature diff: {feature_diffs[ll_max_idx]:.6f}")
    print(f"  Log-lik:      {log_likelihoods[ll_max_idx]:.2f}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Feature matching objective (we want to minimize)
    ax1 = axes[0]
    im1 = ax1.contourf(theta_c_values, rc_values, feature_diffs, levels=20, cmap='viridis_r')
    ax1.scatter([TRUE_THETA_C], [TRUE_RC], c='red', marker='*', s=200, label='True', zorder=5)
    ax1.scatter([fm_opt_theta_c], [fm_opt_rc], c='blue', marker='x', s=200, label='FM Optimum', zorder=5)
    ax1.set_xlabel('theta_c (operating cost)')
    ax1.set_ylabel('RC (replacement cost)')
    ax1.set_title('Feature Matching Objective ||emp - exp|| (minimize)')
    ax1.legend()
    plt.colorbar(im1, ax=ax1)

    # Log-likelihood (we want to maximize)
    ax2 = axes[1]
    im2 = ax2.contourf(theta_c_values, rc_values, log_likelihoods, levels=20, cmap='viridis')
    ax2.scatter([TRUE_THETA_C], [TRUE_RC], c='red', marker='*', s=200, label='True', zorder=5)
    ax2.scatter([ll_opt_theta_c], [ll_opt_rc], c='blue', marker='x', s=200, label='MLE Optimum', zorder=5)
    ax2.set_xlabel('theta_c (operating cost)')
    ax2.set_ylabel('RC (replacement cost)')
    ax2.set_title('Log-Likelihood (maximize)')
    ax2.legend()
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    plt.savefig('mce_irl_landscape.png', dpi=150)
    print("\nSaved visualization to mce_irl_landscape.png")

    # Additional analysis: 1D slices
    print("\n" + "=" * 70)
    print(" 1D SLICES AT TRUE RC = 3.0")
    print("=" * 70)

    rc_idx = np.abs(rc_values - TRUE_RC).argmin()
    print(f"\n{'theta_c':>10} {'Feature Diff':>15} {'Log-Lik':>15}")
    print("-" * 45)
    for j, theta_c in enumerate(theta_c_values):
        print(f"{theta_c:>10.6f} {feature_diffs[rc_idx, j]:>15.6f} {log_likelihoods[rc_idx, j]:>15.2f}")

    print(f"\nMinimum feature diff at RC=3.0: theta_c = {theta_c_values[feature_diffs[rc_idx, :].argmin()]:.6f}")
    print(f"Maximum log-lik at RC=3.0:      theta_c = {theta_c_values[log_likelihoods[rc_idx, :].argmax()]:.6f}")


if __name__ == "__main__":
    main()
