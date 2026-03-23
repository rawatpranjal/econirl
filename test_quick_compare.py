#!/usr/bin/env python3
"""Quick comparison of NFXP vs MCE IRL - minimal setup."""

import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def main():
    print("Quick comparison test...")

    # Small problem
    n_states = 30
    gamma = 0.9  # Lower gamma for faster convergence
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

    # Transitions (a, s, s')
    transitions = torch.zeros((2, n_states, n_states), dtype=torch.float32)
    for s in range(n_states):
        for delta, prob in [(0, 0.35), (1, 0.35), (2, 0.30)]:
            next_s = min(s + delta, n_states - 1)
            transitions[0, s, next_s] += prob
    transitions[1, :, 0] = 1.0

    # Compute policy
    reward_matrix = torch.einsum("sak,k->sa", feature_matrix, true_params)
    V = torch.zeros(n_states, dtype=torch.float32)
    for _ in range(5000):
        Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)
        if torch.abs(V_new - V).max() < 1e-8:
            break
        V = V_new
    Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
    policy = torch.softmax(Q, dim=1)

    print(f"Policy: s=0: {policy[0,1]:.3f}, s={n_states//2}: {policy[n_states//2,1]:.3f}")

    # Simulate smaller dataset
    np.random.seed(42)
    trajectories = []
    for _ in range(100):  # Fewer trajectories
        states, actions, next_states = [], [], []
        state = 0
        for _ in range(50):  # Fewer periods
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
    print(f"Simulated {len(trajectories) * 50} observations")

    # Problem
    problem = DDCProblem(num_states=n_states, num_actions=2, discount_factor=gamma)

    # NFXP
    print("\n--- NFXP ---")
    utility_fn = LinearUtility(feature_matrix=feature_matrix, parameter_names=["op", "rc"])
    nfxp = NFXPEstimator(se_method="asymptotic", verbose=False, inner_tol=1e-8, outer_tol=1e-6)
    result_nfxp = nfxp.estimate(panel=panel, utility=utility_fn, problem=problem, transitions=transitions)
    nfxp_angle = np.degrees(np.arctan2(result_nfxp.parameters[1].item(), result_nfxp.parameters[0].item()))
    print(f"NFXP: [{result_nfxp.parameters[0]:.4f}, {result_nfxp.parameters[1]:.4f}]")
    print(f"NFXP angle: {nfxp_angle:.2f}° (error: {abs(nfxp_angle-true_angle_deg):.4f}°)")

    # MCE IRL
    print("\n--- MCE IRL ---")
    reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op", "rc"])
    config = MCEIRLConfig(
        verbose=False,
        outer_max_iter=100,  # Fewer iterations
        learning_rate=0.1,
        outer_tol=1e-6,
        inner_tol=1e-8,
        inner_max_iter=5000,
        compute_se=False,
    )
    mce = MCEIRLEstimator(config=config)
    init_params = torch.ones(2, dtype=torch.float32) / np.sqrt(2)
    result_mce = mce.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=transitions, initial_params=init_params)
    mce_angle = np.degrees(np.arctan2(result_mce.parameters[1].item(), result_mce.parameters[0].item()))
    print(f"MCE:  [{result_mce.parameters[0]:.4f}, {result_mce.parameters[1]:.4f}]")
    print(f"MCE angle: {mce_angle:.2f}° (error: {abs(mce_angle-true_angle_deg):.4f}°)")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"True angle:  {true_angle_deg}°")
    print(f"NFXP error:  {abs(nfxp_angle - true_angle_deg):.4f}°")
    print(f"MCE error:   {abs(mce_angle - true_angle_deg):.4f}°")

    # Cosine similarity
    nfxp_norm = result_nfxp.parameters / torch.norm(result_nfxp.parameters)
    mce_norm = result_mce.parameters / torch.norm(result_mce.parameters)
    print(f"\nNFXP cosine: {torch.dot(nfxp_norm, true_params).item():.6f}")
    print(f"MCE cosine:  {torch.dot(mce_norm, true_params).item():.6f}")


if __name__ == "__main__":
    main()
