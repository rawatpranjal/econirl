#!/usr/bin/env python3
"""Quick comparison of estimators with normalized features."""

import sys
import torch
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.max_margin_irl import MaxMarginIRLEstimator

print("Starting quick comparison...")
print("=" * 60)

# Config
N_STATES = 20
N_ACTIONS = 2
GAMMA = 0.95
N_TRAJ = 50
N_PERIODS = 30

# True params on unit circle (45 degrees = equal weights)
true_angle = 45
true_rad = np.radians(true_angle)
true_params = torch.tensor([np.cos(true_rad), np.sin(true_rad)], dtype=torch.float32)
print(f"True params: {true_params.tolist()}")
print(f"True angle: {true_angle}°")

# Create normalized features
feature_matrix = torch.zeros((N_STATES, N_ACTIONS, 2), dtype=torch.float32)
for s in range(N_STATES):
    # Operating cost: normalized mileage [0, -1]
    feature_matrix[s, 0, 0] = -s / (N_STATES - 1)
    # Replace: fixed cost
    feature_matrix[s, 1, 1] = -1.0

print(f"Feature range: [{feature_matrix.min():.2f}, {feature_matrix.max():.2f}]")

# Transitions: P(s'|s,a)
transitions = torch.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=torch.float32)
for s in range(N_STATES):
    # Keep: mileage +0, +1, or +2
    for d, p in [(0, 0.35), (1, 0.35), (2, 0.30)]:
        ns = min(s + d, N_STATES - 1)
        transitions[s, 0, ns] += p
# Replace: reset
transitions[:, 1, 0] = 1.0

# Compute true policy
reward_matrix = torch.einsum("sak,k->sa", feature_matrix, true_params)
V = torch.zeros(N_STATES)
for _ in range(5000):
    Q = reward_matrix + GAMMA * torch.einsum("sat,t->sa", transitions, V)
    V_new = torch.logsumexp(Q, dim=1)
    if (V_new - V).abs().max() < 1e-10:
        break
    V = V_new
Q = reward_matrix + GAMMA * torch.einsum("sat,t->sa", transitions, V)
policy = torch.softmax(Q, dim=1)

print(f"True policy P(replace): state 0={policy[0,1]:.3f}, state {N_STATES-1}={policy[-1,1]:.3f}")

# Simulate data
np.random.seed(42)
torch.manual_seed(42)
trajectories = []
for _ in range(N_TRAJ):
    states, actions, next_states = [], [], []
    s = 0
    for _ in range(N_PERIODS):
        states.append(s)
        a = np.random.choice(N_ACTIONS, p=policy[s].numpy())
        actions.append(a)
        ns = np.random.choice(N_STATES, p=transitions[s, a].numpy())
        next_states.append(ns)
        s = ns
    trajectories.append(Trajectory(
        states=torch.tensor(states),
        actions=torch.tensor(actions),
        next_states=torch.tensor(next_states)
    ))
panel = Panel(trajectories=trajectories)
print(f"Simulated {len(trajectories) * N_PERIODS} observations")

# Create estimator inputs
problem = DDCProblem(num_states=N_STATES, num_actions=N_ACTIONS, discount_factor=GAMMA)
reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op_cost", "repl_cost"])
utility_fn = LinearUtility(feature_matrix=feature_matrix, parameter_names=["op_cost", "repl_cost"])

# Transpose transitions for estimators: (n_actions, n_states, n_states)
trans_est = transitions.permute(1, 0, 2)

print("\n" + "=" * 60)
print("NFXP Estimation")
print("=" * 60)

try:
    nfxp = NFXPEstimator(
        se_method="asymptotic",
        optimizer="L-BFGS-B",
        inner_tol=1e-6,
        outer_tol=1e-4,
        inner_max_iter=500,
        outer_max_iter=100,
        compute_hessian=False,  # Skip Hessian for speed
        verbose=False
    )
    result_nfxp = nfxp.estimate(panel=panel, utility=utility_fn, problem=problem, transitions=trans_est)
    nfxp_params = result_nfxp.parameters
    nfxp_angle = np.degrees(np.arctan2(nfxp_params[1].item(), nfxp_params[0].item()))
    nfxp_norm = nfxp_params / torch.norm(nfxp_params)
    nfxp_cos = torch.dot(nfxp_norm, true_params).item()
    print(f"NFXP: [{nfxp_params[0]:.4f}, {nfxp_params[1]:.4f}]")
    print(f"Angle: {nfxp_angle:.2f}° (error: {abs(nfxp_angle - true_angle):.2f}°)")
    print(f"Cosine similarity: {nfxp_cos:.4f}")
except Exception as e:
    print(f"NFXP failed: {e}")

print("\n" + "=" * 60)
print("MCE IRL Estimation")
print("=" * 60)

try:
    mce = MCEIRLEstimator(config=MCEIRLConfig(
        verbose=False, outer_max_iter=100, learning_rate=0.5,
        outer_tol=1e-5, inner_tol=1e-6, inner_max_iter=500, compute_se=False
    ))
    result_mce = mce.estimate(panel=panel, utility=reward_fn, problem=problem,
                               transitions=trans_est, initial_params=torch.ones(2)/np.sqrt(2))
    mce_params = result_mce.parameters
    mce_angle = np.degrees(np.arctan2(mce_params[1].item(), mce_params[0].item()))
    mce_norm = mce_params / torch.norm(mce_params)
    mce_cos = torch.dot(mce_norm, true_params).item()
    print(f"MCE: [{mce_params[0]:.4f}, {mce_params[1]:.4f}]")
    print(f"Angle: {mce_angle:.2f}° (error: {abs(mce_angle - true_angle):.2f}°)")
    print(f"Cosine similarity: {mce_cos:.4f}")
except Exception as e:
    print(f"MCE IRL failed: {e}")

print("\n" + "=" * 60)
print("Max Margin IRL Estimation (Unit Norm)")
print("=" * 60)

try:
    mm = MaxMarginIRLEstimator(max_iterations=50, margin_tol=1e-5, verbose=False, anchor_idx=None)
    result_mm = mm.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=trans_est)
    mm_params = result_mm.parameters
    mm_angle = np.degrees(np.arctan2(mm_params[1].item(), mm_params[0].item()))
    mm_cos = torch.dot(mm_params, true_params).item()
    print(f"MaxMargin: [{mm_params[0]:.4f}, {mm_params[1]:.4f}]")
    print(f"Angle: {mm_angle:.2f}° (error: {abs(mm_angle - true_angle):.2f}°)")
    print(f"Cosine similarity: {mm_cos:.4f}")
except Exception as e:
    print(f"Max Margin failed: {e}")

print("\n" + "=" * 60)
print("Max Margin IRL (Anchor θ₁=1)")
print("=" * 60)

try:
    mm_a = MaxMarginIRLEstimator(max_iterations=50, margin_tol=1e-5, verbose=False, anchor_idx=1)
    result_mm_a = mm_a.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=trans_est)
    mm_a_params = result_mm_a.parameters
    mm_a_norm = mm_a_params / torch.norm(mm_a_params)
    mm_a_angle = np.degrees(np.arctan2(mm_a_norm[1].item(), mm_a_norm[0].item()))
    mm_a_cos = torch.dot(mm_a_norm, true_params).item()
    print(f"MM-Anchor raw: [{mm_a_params[0]:.4f}, {mm_a_params[1]:.4f}]")
    print(f"MM-Anchor norm: [{mm_a_norm[0]:.4f}, {mm_a_norm[1]:.4f}]")
    print(f"Angle: {mm_a_angle:.2f}° (error: {abs(mm_a_angle - true_angle):.2f}°)")
    print(f"Cosine similarity: {mm_a_cos:.4f}")
except Exception as e:
    print(f"Max Margin Anchor failed: {e}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"True params: [{true_params[0]:.4f}, {true_params[1]:.4f}], angle={true_angle}°")
print("Done!")
