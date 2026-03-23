#!/usr/bin/env python3
"""Minimal test - just MCE IRL to verify the fix works."""

import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig


def main():
    print("Minimal MCE IRL test...")

    # Very small problem
    n_states = 10
    gamma = 0.9
    true_angle_deg = 45
    true_angle = np.radians(true_angle_deg)
    true_params = torch.tensor([np.cos(true_angle), np.sin(true_angle)], dtype=torch.float32)

    print(f"True params: [{true_params[0]:.4f}, {true_params[1]:.4f}] (angle: {true_angle_deg}°)")

    # Normalized features
    feature_matrix = torch.zeros((n_states, 2, 2), dtype=torch.float32)
    for s in range(n_states):
        feature_matrix[s, 0, 0] = -s / (n_states - 1)  # Operating cost
        feature_matrix[s, 1, 1] = -1.0  # Replacement cost

    # Transitions
    transitions = torch.zeros((2, n_states, n_states), dtype=torch.float32)
    for s in range(n_states):
        next_s = min(s + 1, n_states - 1)
        transitions[0, s, next_s] = 1.0  # Keep: always +1
    transitions[1, :, 0] = 1.0  # Replace: reset

    # Compute policy
    reward_matrix = torch.einsum("sak,k->sa", feature_matrix, true_params)
    V = torch.zeros(n_states, dtype=torch.float32)
    for _ in range(1000):
        Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)
        if torch.abs(V_new - V).max() < 1e-8:
            break
        V = V_new
    Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
    policy = torch.softmax(Q, dim=1)

    print(f"Policy: P(replace|s=0)={policy[0,1]:.3f}, P(replace|s=9)={policy[9,1]:.3f}")

    # Simulate
    np.random.seed(42)
    trajectories = []
    for _ in range(50):
        states, actions, next_states = [], [], []
        state = 0
        for _ in range(30):
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
    print(f"Observations: {50 * 30}")

    # MCE IRL
    print("\nRunning MCE IRL...")
    problem = DDCProblem(num_states=n_states, num_actions=2, discount_factor=gamma)
    reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op", "rc"])

    config = MCEIRLConfig(
        verbose=True,
        outer_max_iter=50,
        learning_rate=0.2,
        outer_tol=1e-6,
        inner_tol=1e-8,
        inner_max_iter=1000,
        compute_se=False,
    )
    mce = MCEIRLEstimator(config=config)

    init_params = torch.ones(2, dtype=torch.float32) / np.sqrt(2)
    result = mce.estimate(
        panel=panel,
        utility=reward_fn,
        problem=problem,
        transitions=transitions,
        initial_params=init_params,
        true_params=true_params,
    )

    mce_angle = np.degrees(np.arctan2(result.parameters[1].item(), result.parameters[0].item()))

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"True:  [{true_params[0]:.4f}, {true_params[1]:.4f}] (angle: {true_angle_deg}°)")
    print(f"Est:   [{result.parameters[0]:.4f}, {result.parameters[1]:.4f}] (angle: {mce_angle:.2f}°)")
    print(f"Error: {abs(mce_angle - true_angle_deg):.4f}°")

    mce_norm = result.parameters / torch.norm(result.parameters)
    cos_sim = torch.dot(mce_norm, true_params).item()
    print(f"Cosine similarity: {cos_sim:.6f}")

    if abs(mce_angle - true_angle_deg) < 5:
        print("\nSUCCESS: Recovered within 5 degrees!")
    else:
        print(f"\nNEEDS WORK: Angle error = {abs(mce_angle - true_angle_deg):.2f}°")


if __name__ == "__main__":
    main()
