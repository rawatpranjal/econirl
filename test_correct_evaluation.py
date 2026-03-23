#!/usr/bin/env python3
"""
Correct evaluation of Max Margin IRL based on Abbeel & Ng 2004.

The paper says (Section 3):
  "our algorithm does not necessarily recover the underlying reward function correctly.
   The performance guarantees only depend on (approximately) matching the
   feature expectations, not on recovering the true underlying reward function."

So we should evaluate:
1. Feature expectation matching: ||μ(π̃) - μ_E||₂
2. Policy performance: V^π̃ vs V^πE
NOT parameter recovery: ||θ̃ - θ*||
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.preferences.action_reward import ActionDependentReward
from econirl.estimation.max_margin_irl import MaxMarginIRLEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig

print("=" * 70)
print("Correct Evaluation: Feature Matching (per Abbeel & Ng 2004)")
print("=" * 70)

# Setup
N_STATES, N_ACTIONS, GAMMA = 20, 2, 0.95
N_TRAJ, N_PERIODS = 50, 30
true_angle = 45
true_params = torch.tensor([np.cos(np.radians(true_angle)),
                            np.sin(np.radians(true_angle))], dtype=torch.float32)

# Normalized features
feature_matrix = torch.zeros((N_STATES, N_ACTIONS, 2), dtype=torch.float32)
for s in range(N_STATES):
    feature_matrix[s, 0, 0] = -s / (N_STATES - 1)
    feature_matrix[s, 1, 1] = -1.0

# Transitions
transitions = torch.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=torch.float32)
for s in range(N_STATES):
    for d, p in [(0, 0.35), (1, 0.35), (2, 0.30)]:
        transitions[s, 0, min(s+d, N_STATES-1)] += p
transitions[:, 1, 0] = 1.0

def compute_policy(params):
    """Compute soft optimal policy."""
    R = torch.einsum("sak,k->sa", feature_matrix, params)
    V = torch.zeros(N_STATES)
    for _ in range(5000):
        Q = R + GAMMA * torch.einsum("sat,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)
        if (V_new - V).abs().max() < 1e-10:
            break
        V = V_new
    Q = R + GAMMA * torch.einsum("sat,t->sa", transitions, V)
    return torch.softmax(Q, dim=1), V

def compute_feature_expectations(policy, initial_dist=None):
    """Compute μ(π) = E[Σ γ^t φ(s_t, a_t) | π].

    Uses stationary distribution approach.
    """
    if initial_dist is None:
        initial_dist = torch.zeros(N_STATES)
        initial_dist[0] = 1.0  # Start at state 0

    # Policy-induced transition: P_π[s'|s] = Σ_a π(a|s) P(s'|s,a)
    P_pi = torch.einsum("sa,sat->st", policy, transitions)

    # Discounted state visitation: d = (1-γ)(I - γP_π)^{-1} d_0
    I = torch.eye(N_STATES)
    try:
        d_pi = (1 - GAMMA) * torch.linalg.solve(I - GAMMA * P_pi, initial_dist)
        d_pi = d_pi / d_pi.sum()  # Normalize
    except:
        # Fallback: iterative
        d_pi = initial_dist.clone()
        for _ in range(1000):
            d_new = initial_dist * (1 - GAMMA) + GAMMA * (P_pi.T @ d_pi)
            if (d_new - d_pi).abs().max() < 1e-10:
                break
            d_pi = d_new
        d_pi = d_pi / d_pi.sum()

    # μ(π) = Σ_s d_π(s) Σ_a π(a|s) φ(s,a)
    # = Σ_s d_π(s) * [Σ_a π(a|s) φ(s,a,k)]
    policy_weighted_features = torch.einsum("sa,sak->sk", policy, feature_matrix)
    mu = torch.einsum("s,sk->k", d_pi, policy_weighted_features)

    return mu

def compute_value(policy, params):
    """Compute expected value under true reward."""
    R = torch.einsum("sak,k->sa", feature_matrix, params)
    V = torch.zeros(N_STATES)
    for _ in range(5000):
        # V^π(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
        Q = R + GAMMA * torch.einsum("sat,t->sa", transitions, V)
        V_new = torch.einsum("sa,sa->s", policy, Q)
        if (V_new - V).abs().max() < 1e-10:
            break
        V = V_new
    return V[0].item()  # Value at initial state

# Compute true policy and expert feature expectations
true_policy, true_V = compute_policy(true_params)
mu_E = compute_feature_expectations(true_policy)
V_expert = compute_value(true_policy, true_params)

print(f"\nTrue params: {true_params.tolist()}")
print(f"Expert feature expectations μ_E: {mu_E.tolist()}")
print(f"Expert value V^πE: {V_expert:.4f}")

# Simulate data
np.random.seed(42)
torch.manual_seed(42)
trajectories = []
for _ in range(N_TRAJ):
    states, actions, next_states = [], [], []
    s = 0
    for _ in range(N_PERIODS):
        states.append(s)
        a = np.random.choice(N_ACTIONS, p=true_policy[s].numpy())
        actions.append(a)
        ns = np.random.choice(N_STATES, p=transitions[s, a].numpy())
        next_states.append(ns)
        s = ns
    trajectories.append(Trajectory(
        states=torch.tensor(states), actions=torch.tensor(actions),
        next_states=torch.tensor(next_states)
    ))
panel = Panel(trajectories=trajectories)

problem = DDCProblem(num_states=N_STATES, num_actions=N_ACTIONS, discount_factor=GAMMA)
reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op_cost", "repl_cost"])
trans_est = transitions.permute(1, 0, 2)

print(f"\nSimulated {len(trajectories) * N_PERIODS} observations")

# ============================================================================
# Max Margin IRL
# ============================================================================
print("\n" + "=" * 70)
print("MAX MARGIN IRL")
print("=" * 70)

mm = MaxMarginIRLEstimator(max_iterations=50, margin_tol=1e-6, verbose=False)
result_mm = mm.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=trans_est)
mm_params = result_mm.parameters

# Compute policy from MM params
mm_policy, _ = compute_policy(mm_params)
mu_mm = compute_feature_expectations(mm_policy)
V_mm = compute_value(mm_policy, true_params)  # Value under TRUE reward

print(f"MM params: {mm_params.tolist()}")
print(f"MM feature expectations μ(π̃): {mu_mm.tolist()}")
print(f"MM value V^π̃ (under true R*): {V_mm:.4f}")

# THE CORRECT METRICS (per paper)
feature_dist = torch.norm(mu_mm - mu_E).item()
value_gap = V_expert - V_mm

print(f"\n*** CORRECT EVALUATION (Abbeel & Ng 2004) ***")
print(f"Feature expectation distance ||μ(π̃) - μ_E||₂: {feature_dist:.6f}")
print(f"Value gap (V^πE - V^π̃): {value_gap:.6f}")

# For comparison, also show parameter distance (what we were measuring before)
param_cos = torch.dot(mm_params / torch.norm(mm_params), true_params).item()
print(f"\n(For comparison) Parameter cosine similarity: {param_cos:.4f}")

# ============================================================================
# MCE IRL
# ============================================================================
print("\n" + "=" * 70)
print("MCE IRL")
print("=" * 70)

mce = MCEIRLEstimator(config=MCEIRLConfig(
    verbose=False, outer_max_iter=100, learning_rate=0.5,
    outer_tol=1e-5, inner_tol=1e-6, inner_max_iter=500, compute_se=False
))
result_mce = mce.estimate(panel=panel, utility=reward_fn, problem=problem,
                          transitions=trans_est, initial_params=torch.ones(2)/np.sqrt(2))
mce_params = result_mce.parameters

mce_policy, _ = compute_policy(mce_params)
mu_mce = compute_feature_expectations(mce_policy)
V_mce = compute_value(mce_policy, true_params)

print(f"MCE params: {mce_params.tolist()}")
print(f"MCE feature expectations μ(π̃): {mu_mce.tolist()}")
print(f"MCE value V^π̃ (under true R*): {V_mce:.4f}")

feature_dist_mce = torch.norm(mu_mce - mu_E).item()
value_gap_mce = V_expert - V_mce

print(f"\n*** CORRECT EVALUATION ***")
print(f"Feature expectation distance ||μ(π̃) - μ_E||₂: {feature_dist_mce:.6f}")
print(f"Value gap (V^πE - V^π̃): {value_gap_mce:.6f}")

param_cos_mce = torch.dot(mce_params / torch.norm(mce_params), true_params).item()
print(f"\n(For comparison) Parameter cosine similarity: {param_cos_mce:.4f}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Correct Evaluation")
print("=" * 70)

print(f"\n{'Method':<15} {'||μ-μ_E||₂':<15} {'Value Gap':<15} {'Param Cos':<15}")
print("-" * 60)
print(f"{'Max Margin':<15} {feature_dist:<15.6f} {value_gap:<15.6f} {param_cos:<15.4f}")
print(f"{'MCE IRL':<15} {feature_dist_mce:<15.6f} {value_gap_mce:<15.6f} {param_cos_mce:<15.4f}")
print("-" * 60)

print("""
KEY INSIGHT FROM ABBEEL & NG 2004:
  "our algorithm does not necessarily recover the underlying reward
   function correctly. The performance guarantees only depend on
   (approximately) matching the feature expectations."

So the CORRECT metrics are:
  1. Feature matching: ||μ(π̃) - μ_E||₂  (lower is better)
  2. Value gap: V^πE - V^π̃  (lower is better, 0 = matches expert)

Parameter recovery (cosine similarity) is NOT what Max Margin optimizes for.
""")
