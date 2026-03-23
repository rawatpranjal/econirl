#!/usr/bin/env python3
"""Quick single experiment test."""

import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def main():
    print("Running single experiment...")

    n_states = 50
    gamma = 0.95
    true_angle_deg = 45
    true_angle = np.radians(true_angle_deg)
    true_params = torch.tensor([np.cos(true_angle), np.sin(true_angle)], dtype=torch.float32)

    print(f"True params: [{true_params[0]:.4f}, {true_params[1]:.4f}] (angle: {true_angle_deg}°)")

    # Create normalized features
    feature_matrix = torch.zeros((n_states, 2, 2), dtype=torch.float32)
    for s in range(n_states):
        normalized_mileage = -s / (n_states - 1)
        feature_matrix[s, 0, 0] = normalized_mileage
        feature_matrix[s, 0, 1] = 0.0
        feature_matrix[s, 1, 0] = 0.0
        feature_matrix[s, 1, 1] = -1.0

    # Transitions (s, a, s')
    transitions = torch.zeros((n_states, 2, n_states), dtype=torch.float32)
    for s in range(n_states):
        for delta, prob in [(0, 0.35), (1, 0.35), (2, 0.30)]:
            next_s = min(s + delta, n_states - 1)
            transitions[s, 0, next_s] += prob
    transitions[:, 1, 0] = 1.0

    # For estimators (a, s, s')
    transitions_est = transitions.permute(1, 0, 2)

    # Compute policy
    reward_matrix = torch.einsum("sak,k->sa", feature_matrix, true_params)
    V = torch.zeros(n_states, dtype=torch.float32)
    for _ in range(50000):
        Q = reward_matrix + gamma * torch.einsum("sat,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)
        if torch.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    Q = reward_matrix + gamma * torch.einsum("sat,t->sa", transitions, V)
    policy = torch.softmax(Q, dim=1)

    print(f"Policy replace prob: s=0: {policy[0,1]:.3f}, s=25: {policy[25,1]:.3f}, s=49: {policy[49,1]:.3f}")

    # Simulate
    np.random.seed(42)
    trajectories = []
    for _ in range(200):
        states, actions, next_states = [], [], []
        state = 0
        for _ in range(100):
            states.append(state)
            action = np.random.choice(2, p=policy[state].numpy())
            actions.append(action)
            next_state = np.random.choice(n_states, p=transitions[state, action].numpy())
            next_states.append(next_state)
            state = next_state
        trajectories.append(Trajectory(
            states=torch.tensor(states),
            actions=torch.tensor(actions),
            next_states=torch.tensor(next_states),
        ))
    panel = Panel(trajectories=trajectories)
    print(f"Simulated {len(trajectories) * 100} observations")

    # Problem
    problem = DDCProblem(num_states=n_states, num_actions=2, discount_factor=gamma)

    # NFXP
    print("\n--- NFXP ---")
    utility_fn = LinearUtility(feature_matrix=feature_matrix, parameter_names=["op", "rc"])
    nfxp = NFXPEstimator(se_method="asymptotic", verbose=False, inner_tol=1e-10, outer_tol=1e-8)
    result_nfxp = nfxp.estimate(panel=panel, utility=utility_fn, problem=problem, transitions=transitions_est)
    nfxp_angle = np.degrees(np.arctan2(result_nfxp.parameters[1].item(), result_nfxp.parameters[0].item()))
    print(f"NFXP: [{result_nfxp.parameters[0]:.4f}, {result_nfxp.parameters[1]:.4f}] angle={nfxp_angle:.2f}° err={abs(nfxp_angle-true_angle_deg):.4f}°")

    # MCE IRL
    print("\n--- MCE IRL ---")
    reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op", "rc"])
    config = MCEIRLConfig(verbose=False, outer_max_iter=300, learning_rate=0.1, outer_tol=1e-8, inner_tol=1e-10, compute_se=False)
    mce = MCEIRLEstimator(config=config)
    init_params = torch.ones(2) / np.sqrt(2)
    result_mce = mce.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=transitions_est, initial_params=init_params)
    mce_angle = np.degrees(np.arctan2(result_mce.parameters[1].item(), result_mce.parameters[0].item()))
    print(f"MCE:  [{result_mce.parameters[0]:.4f}, {result_mce.parameters[1]:.4f}] angle={mce_angle:.2f}° err={abs(mce_angle-true_angle_deg):.4f}°")

    # Compare
    print("\n--- Comparison ---")
    print(f"True angle: {true_angle_deg}°")
    print(f"NFXP error: {abs(nfxp_angle - true_angle_deg):.4f}°")
    print(f"MCE error:  {abs(mce_angle - true_angle_deg):.4f}°")


if __name__ == "__main__":
    main()
