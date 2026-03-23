#!/usr/bin/env python3
"""
Final comparison: NFXP vs MCE IRL on normalized DGP.
Uses smaller problem size for reasonable runtime.
"""

import torch
import numpy as np
from tqdm import tqdm

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def run_experiment(true_angle_deg, n_states=20, gamma=0.9, n_traj=100, n_periods=50, seed=42):
    """Run single experiment and return results."""

    true_angle = np.radians(true_angle_deg)
    true_params = torch.tensor([np.cos(true_angle), np.sin(true_angle)], dtype=torch.float32)

    # Normalized features [-1, 0] for operating cost
    feature_matrix = torch.zeros((n_states, 2, 2), dtype=torch.float32)
    for s in range(n_states):
        feature_matrix[s, 0, 0] = -s / (n_states - 1)
        feature_matrix[s, 1, 1] = -1.0

    # Transitions
    transitions = torch.zeros((2, n_states, n_states), dtype=torch.float32)
    for s in range(n_states):
        next_s = min(s + 1, n_states - 1)
        transitions[0, s, next_s] = 1.0
    transitions[1, :, 0] = 1.0

    # Compute policy
    reward_matrix = torch.einsum("sak,k->sa", feature_matrix, true_params)
    V = torch.zeros(n_states, dtype=torch.float32)
    for _ in range(2000):
        Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)
        if torch.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
    policy = torch.softmax(Q, dim=1)

    # Simulate data
    np.random.seed(seed)
    trajectories = []
    for _ in range(n_traj):
        states, actions, next_states = [], [], []
        state = 0
        for _ in range(n_periods):
            states.append(state)
            action = np.random.choice(2, p=policy[state].numpy())
            actions.append(action)
            next_state = np.random.choice(n_states, p=transitions[action, state].numpy())
            next_states.append(next_state)
            state = next_state
        trajectories.append(Trajectory(
            states=torch.tensor(states),
            actions=torch.tensor(actions),
            next_states=torch.tensor(next_states),
        ))
    panel = Panel(trajectories=trajectories)

    problem = DDCProblem(num_states=n_states, num_actions=2, discount_factor=gamma)
    results = {'true_angle': true_angle_deg, 'true_params': true_params}

    # NFXP
    utility_fn = LinearUtility(feature_matrix=feature_matrix, parameter_names=["op", "rc"])
    nfxp = NFXPEstimator(se_method="asymptotic", verbose=False, inner_tol=1e-8, outer_tol=1e-6)
    try:
        result_nfxp = nfxp.estimate(panel=panel, utility=utility_fn, problem=problem, transitions=transitions)
        nfxp_params = result_nfxp.parameters
        nfxp_angle = np.degrees(np.arctan2(nfxp_params[1].item(), nfxp_params[0].item()))
        nfxp_norm = nfxp_params / torch.norm(nfxp_params)
        results['nfxp_angle'] = nfxp_angle
        results['nfxp_cos'] = torch.dot(nfxp_norm, true_params).item()
        results['nfxp_err'] = abs(nfxp_angle - true_angle_deg)
    except Exception as e:
        results['nfxp_err'] = float('nan')

    # MCE IRL
    reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op", "rc"])
    config = MCEIRLConfig(
        verbose=False, outer_max_iter=100, learning_rate=0.2,
        outer_tol=1e-6, inner_tol=1e-8, inner_max_iter=2000, compute_se=False
    )
    mce = MCEIRLEstimator(config=config)
    init_params = torch.ones(2, dtype=torch.float32) / np.sqrt(2)
    try:
        result_mce = mce.estimate(
            panel=panel, utility=reward_fn, problem=problem,
            transitions=transitions, initial_params=init_params
        )
        mce_params = result_mce.parameters
        mce_angle = np.degrees(np.arctan2(mce_params[1].item(), mce_params[0].item()))
        mce_norm = mce_params / torch.norm(mce_params)
        results['mce_angle'] = mce_angle
        results['mce_cos'] = torch.dot(mce_norm, true_params).item()
        results['mce_err'] = abs(mce_angle - true_angle_deg)
    except Exception as e:
        results['mce_err'] = float('nan')

    return results


def main():
    print("=" * 70)
    print("FINAL COMPARISON: NFXP vs MCE IRL")
    print("Normalized Features, Unit-Norm Parameters")
    print("=" * 70)
    print()

    angles = [20, 30, 45, 60, 70]
    all_results = []

    print("Running experiments...")
    for angle in tqdm(angles, desc="Experiments"):
        result = run_experiment(angle)
        all_results.append(result)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"{'True Angle':<12} {'NFXP Angle':<12} {'NFXP Err':<12} {'MCE Angle':<12} {'MCE Err':<12}")
    print("-" * 60)

    for r in all_results:
        nfxp_angle = r.get('nfxp_angle', float('nan'))
        mce_angle = r.get('mce_angle', float('nan'))
        print(f"{r['true_angle']:<12.0f} {nfxp_angle:<12.2f} {r.get('nfxp_err', float('nan')):<12.4f} "
              f"{mce_angle:<12.2f} {r.get('mce_err', float('nan')):<12.4f}")

    # Averages
    nfxp_errs = [r.get('nfxp_err', float('nan')) for r in all_results if not np.isnan(r.get('nfxp_err', float('nan')))]
    mce_errs = [r.get('mce_err', float('nan')) for r in all_results if not np.isnan(r.get('mce_err', float('nan')))]

    print("-" * 60)
    print(f"{'AVERAGE':<12} {'—':<12} {np.mean(nfxp_errs):<12.4f} {'—':<12} {np.mean(mce_errs):<12.4f}")

    print("\n" + "=" * 70)
    print("COSINE SIMILARITIES (1.0 = perfect)")
    print("=" * 70)
    print()
    print(f"{'True Angle':<12} {'NFXP':<12} {'MCE IRL':<12}")
    print("-" * 36)
    for r in all_results:
        print(f"{r['true_angle']:<12.0f} {r.get('nfxp_cos', float('nan')):<12.6f} {r.get('mce_cos', float('nan')):<12.6f}")

    nfxp_cos = [r.get('nfxp_cos', float('nan')) for r in all_results if not np.isnan(r.get('nfxp_cos', float('nan')))]
    mce_cos = [r.get('mce_cos', float('nan')) for r in all_results if not np.isnan(r.get('mce_cos', float('nan')))]
    print("-" * 36)
    print(f"{'AVERAGE':<12} {np.mean(nfxp_cos):<12.6f} {np.mean(mce_cos):<12.6f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(f"\nNFXP average angle error:    {np.mean(nfxp_errs):.4f}°")
    print(f"MCE IRL average angle error: {np.mean(mce_errs):.4f}°")
    print(f"\nNFXP average cosine sim:     {np.mean(nfxp_cos):.6f}")
    print(f"MCE IRL average cosine sim:  {np.mean(mce_cos):.6f}")

    if np.mean(mce_errs) < 5.0:
        print("\n✓ MCE IRL achieves good parameter recovery (< 5° error)")
    if np.mean(mce_cos) > 0.99:
        print("✓ MCE IRL achieves excellent direction recovery (cos > 0.99)")


if __name__ == "__main__":
    main()
