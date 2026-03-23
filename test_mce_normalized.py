#!/usr/bin/env python3
"""
Test MCE IRL with normalized parameters (unit circle).

The idea: Use parameters that lie on the unit sphere so that
RMSE is meaningful and comparable across experiments.

For Rust bus with 2 params: (theta_c, RC)
Normalized: theta = (theta_c, RC) / ||theta||

We create a custom feature matrix that accounts for the scaling.
"""

import torch
import numpy as np
from tqdm import tqdm

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.core.bellman import SoftBellmanOperator
from econirl.preferences.action_reward import ActionDependentReward


def create_rust_features_normalized(n_states: int = 90) -> torch.Tensor:
    """Create feature matrix for Rust bus with normalized features.

    Features are scaled so that natural parameters are on the unit circle.
    - Feature 0 (operating cost): -s/45 for keep, 0 for replace
    - Feature 1 (replacement cost): 0 for keep, -1 for replace

    With these features:
    - theta = [cos(angle), sin(angle)] gives interpretable behavior
    - angle near 0: low operating cost concern, high replacement cost
    - angle near pi/2: high operating cost concern, low replacement cost
    """
    features = torch.zeros((n_states, 2, 2))
    for s in range(n_states):
        # Keep action: operating cost increases with mileage
        features[s, 0, 0] = -s / 45.0  # Normalize by mid-point
        features[s, 0, 1] = 0.0

        # Replace action: fixed replacement cost
        features[s, 1, 0] = 0.0
        features[s, 1, 1] = -1.0

    return features


def create_transitions(n_states: int = 90) -> torch.Tensor:
    """Create Rust-style transition matrices."""
    transitions = torch.zeros((2, n_states, n_states))

    # Keep action: mileage increases by 0, 1, or 2 with prob 0.35, 0.35, 0.3
    for s in range(n_states):
        for delta, prob in [(0, 0.35), (1, 0.35), (2, 0.30)]:
            next_s = min(s + delta, n_states - 1)
            transitions[0, s, next_s] += prob

    # Replace action: reset to state 0
    transitions[1, :, 0] = 1.0

    return transitions


def soft_value_iteration(reward_matrix, transitions, gamma=0.99, max_iter=50000, tol=1e-10):
    """Soft value iteration."""
    n_states, n_actions = reward_matrix.shape
    V = torch.zeros(n_states)

    for _ in range(max_iter):
        # Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) * V(s')
        Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)

        if torch.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new

    Q = reward_matrix + gamma * torch.einsum("ast,t->sa", transitions, V)
    policy = torch.softmax(Q, dim=1)
    return V, policy


def simulate_data(policy, transitions, n_trajectories=100, n_periods=200, seed=42):
    """Simulate trajectories from policy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    trajectories = []
    n_states, n_actions = policy.shape

    for _ in range(n_trajectories):
        states = []
        actions = []
        next_states = []

        state = 0  # Start at state 0
        for _ in range(n_periods):
            states.append(state)

            # Sample action from policy
            action = np.random.choice(n_actions, p=policy[state].numpy())
            actions.append(action)

            # Transition
            next_state = np.random.choice(n_states, p=transitions[action, state].numpy())
            next_states.append(next_state)
            state = next_state

        trajectories.append(Trajectory(
            states=torch.tensor(states),
            actions=torch.tensor(actions),
            next_states=torch.tensor(next_states),
        ))

    return Panel(trajectories=trajectories)


def mce_irl_normalized(panel, feature_matrix, transitions, gamma=0.99,
                       init_params=None, true_params=None,
                       lr=0.1, max_iter=300, tol=1e-8):
    """MCE IRL with normalized parameters (constrained to unit circle)."""

    n_states, n_actions, n_features = feature_matrix.shape

    # Compute empirical features
    emp_sum = torch.zeros(n_features)
    total = 0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            emp_sum += feature_matrix[s, a, :]
            total += 1
    empirical = emp_sum / total

    # Initialize on unit circle
    if init_params is None:
        angle = np.pi / 4  # 45 degrees
        params = torch.tensor([np.cos(angle), np.sin(angle)], dtype=torch.float32)
    else:
        params = (init_params / torch.norm(init_params)).float()

    best_obj = float('inf')
    best_params = params.clone()

    pbar = tqdm(range(max_iter), desc="MCE IRL (normalized)")

    for i in pbar:
        # Compute reward and policy
        reward_matrix = torch.einsum("sak,k->sa", feature_matrix, params)
        V, policy = soft_value_iteration(reward_matrix, transitions, gamma)

        # Expected features via empirical states
        exp_sum = torch.zeros(n_features)
        total = 0
        for traj in panel.trajectories:
            for t in range(len(traj)):
                s = traj.states[t].item()
                for a in range(n_actions):
                    exp_sum += policy[s, a] * feature_matrix[s, a, :]
                total += 1
        expected = exp_sum / total

        # Gradient
        gradient = empirical - expected
        obj = 0.5 * torch.sum(gradient ** 2).item()
        grad_norm = torch.norm(gradient).item()

        if obj < best_obj:
            best_obj = obj
            best_params = params.clone()

        # Progress
        postfix = {"obj": f"{obj:.6f}", "||grad||": f"{grad_norm:.4f}"}
        if true_params is not None:
            # Compute RMSE on unit circle (angular distance)
            true_norm = true_params / torch.norm(true_params)
            rmse = torch.sqrt(torch.mean((params - true_norm) ** 2)).item()
            cos_sim = torch.dot(params, true_norm).item()
            postfix["RMSE"] = f"{rmse:.4f}"
            postfix["cos_sim"] = f"{cos_sim:.4f}"
        pbar.set_postfix(postfix)

        if grad_norm < tol:
            break

        # Gradient step then project back to unit circle
        params = params + lr * gradient
        params = params / torch.norm(params)

    pbar.close()
    return best_params, best_obj


def main():
    print("=" * 70)
    print("TEST: MCE IRL with Normalized Parameters (Unit Circle)")
    print("=" * 70)
    print()

    n_states = 90
    gamma = 0.99

    # True parameters on unit circle
    # angle = pi/6 means: more weight on replacement cost than operating cost
    true_angle = np.pi / 6  # 30 degrees
    TRUE_PARAMS = torch.tensor([np.cos(true_angle), np.sin(true_angle)], dtype=torch.float32)
    print(f"True params (unit circle): [{TRUE_PARAMS[0]:.4f}, {TRUE_PARAMS[1]:.4f}]")
    print(f"True angle: {np.degrees(true_angle):.1f} degrees")
    print()

    # Create features and transitions
    feature_matrix = create_rust_features_normalized(n_states)
    transitions = create_transitions(n_states)

    # Generate data from true policy
    reward_matrix = torch.einsum("sak,k->sa", feature_matrix, TRUE_PARAMS)
    V, policy = soft_value_iteration(reward_matrix, transitions, gamma)

    print(f"True policy replace prob at state 0: {policy[0, 1]:.4f}")
    print(f"True policy replace prob at state 45: {policy[45, 1]:.4f}")
    print(f"True policy replace prob at state 89: {policy[89, 1]:.4f}")
    print()

    panel = simulate_data(policy, transitions, n_trajectories=200, n_periods=200, seed=12345)
    print(f"Observations: {sum(len(t) for t in panel.trajectories)}")
    print()

    # Initialize at different angle
    init_angle = np.pi / 3  # 60 degrees (off by 30 degrees)
    init_params = torch.tensor([np.cos(init_angle), np.sin(init_angle)], dtype=torch.float32)
    print(f"Init params: [{init_params[0]:.4f}, {init_params[1]:.4f}]")
    print(f"Init angle: {np.degrees(init_angle):.1f} degrees")
    print()

    # Run MCE IRL
    result_params, final_obj = mce_irl_normalized(
        panel, feature_matrix, transitions, gamma,
        init_params=init_params,
        true_params=TRUE_PARAMS,
        lr=0.5,
        max_iter=200,
    )

    # Results
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    true_norm = TRUE_PARAMS / torch.norm(TRUE_PARAMS)
    result_norm = result_params / torch.norm(result_params)

    result_angle = np.arctan2(result_params[1].item(), result_params[0].item())
    true_angle_rad = np.arctan2(TRUE_PARAMS[1].item(), TRUE_PARAMS[0].item())

    print(f"True:   [{true_norm[0]:.4f}, {true_norm[1]:.4f}] (angle: {np.degrees(true_angle_rad):.1f}°)")
    print(f"Est:    [{result_norm[0]:.4f}, {result_norm[1]:.4f}] (angle: {np.degrees(result_angle):.1f}°)")
    print()

    rmse = torch.sqrt(torch.mean((result_params - true_norm) ** 2)).item()
    cos_sim = torch.dot(result_params, true_norm).item()
    angle_error = np.degrees(abs(result_angle - true_angle_rad))

    print(f"RMSE: {rmse:.6f}")
    print(f"Cosine similarity: {cos_sim:.6f}")
    print(f"Angle error: {angle_error:.2f} degrees")
    print(f"Feature diff: {np.sqrt(2 * final_obj):.6f}")

    if angle_error < 5:
        print("\nSUCCESS: Parameters recovered within 5 degrees!")
    elif angle_error < 15:
        print("\nAPPROXIMATE: Parameters recovered within 15 degrees")
    else:
        print("\nNEEDS WORK: Angle error > 15 degrees")


if __name__ == "__main__":
    main()
