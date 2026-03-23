#!/usr/bin/env python3
"""
Test improvement ideas for Max Margin IRL parameter recovery.

Ideas to test:
1. Sign constraints - enforce known sign of parameters
2. Max Margin as initialization for MCE IRL
3. Automatic feature normalization
4. Policy matching instead of feature matching
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
print("Testing Improvement Ideas for Max Margin IRL")
print("=" * 70)

# Setup (same as quick_comparison.py)
N_STATES, N_ACTIONS, GAMMA = 20, 2, 0.95
N_TRAJ, N_PERIODS = 50, 30

true_angle = 45
true_params = torch.tensor([np.cos(np.radians(true_angle)),
                            np.sin(np.radians(true_angle))], dtype=torch.float32)

# Normalized features
feature_matrix = torch.zeros((N_STATES, N_ACTIONS, 2), dtype=torch.float32)
for s in range(N_STATES):
    feature_matrix[s, 0, 0] = -s / (N_STATES - 1)  # Operating cost
    feature_matrix[s, 1, 1] = -1.0  # Replacement cost

# Transitions
transitions = torch.zeros((N_STATES, N_ACTIONS, N_STATES), dtype=torch.float32)
for s in range(N_STATES):
    for d, p in [(0, 0.35), (1, 0.35), (2, 0.30)]:
        transitions[s, 0, min(s+d, N_STATES-1)] += p
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

problem = DDCProblem(num_states=N_STATES, num_actions=N_ACTIONS, discount_factor=GAMMA)
reward_fn = ActionDependentReward(feature_matrix=feature_matrix, parameter_names=["op_cost", "repl_cost"])
trans_est = transitions.permute(1, 0, 2)

print(f"\nTrue params: {true_params.tolist()}, angle={true_angle}°")
print(f"Data: {len(trajectories) * N_PERIODS} observations")

def evaluate(params, name):
    """Evaluate parameter recovery."""
    norm_params = params / torch.norm(params)
    angle = np.degrees(np.arctan2(params[1].item(), params[0].item()))
    cos_sim = torch.dot(norm_params, true_params).item()
    print(f"{name}: [{params[0]:.4f}, {params[1]:.4f}], "
          f"angle={angle:.1f}° (err={abs(angle-true_angle):.1f}°), cos={cos_sim:.4f}")
    return cos_sim

# ============================================================================
# Baseline: Standard Max Margin
# ============================================================================
print("\n" + "=" * 70)
print("BASELINE: Standard Max Margin IRL")
print("=" * 70)

mm = MaxMarginIRLEstimator(max_iterations=50, margin_tol=1e-5, verbose=False)
result_mm = mm.estimate(panel=panel, utility=reward_fn, problem=problem, transitions=trans_est)
baseline_cos = evaluate(result_mm.parameters, "MaxMargin")

# ============================================================================
# IDEA 1: Use Max Margin as initialization for MCE IRL
# ============================================================================
print("\n" + "=" * 70)
print("IDEA 1: Max Margin → MCE IRL (warm start)")
print("=" * 70)

mce = MCEIRLEstimator(config=MCEIRLConfig(
    verbose=False, outer_max_iter=100, learning_rate=0.5,
    outer_tol=1e-5, inner_tol=1e-6, inner_max_iter=500, compute_se=False
))

# Use Max Margin result as initialization
mm_init = result_mm.parameters.clone()
result_hybrid = mce.estimate(
    panel=panel, utility=reward_fn, problem=problem,
    transitions=trans_est, initial_params=mm_init
)
idea1_cos = evaluate(result_hybrid.parameters, "MM→MCE")

# Compare with random init MCE
result_mce_rand = mce.estimate(
    panel=panel, utility=reward_fn, problem=problem,
    transitions=trans_est, initial_params=torch.ones(2)/np.sqrt(2)
)
evaluate(result_mce_rand.parameters, "MCE (rand init)")

# ============================================================================
# IDEA 2: Constrain signs (costs should be positive weights on negative features)
# ============================================================================
print("\n" + "=" * 70)
print("IDEA 2: Sign-constrained Max Margin")
print("=" * 70)
print("(Features are negative, so positive weights = costs)")
print("Adding bounds: theta >= 0")

# Modify _solve_qp to add sign constraints
# For now, let's test by post-processing: if sign is wrong, flip it
mm_params = result_mm.parameters.clone()
# Our features are negative (costs), so weights should be positive
# If MM found negative weights, that's the sign flip issue
if mm_params[0] < 0:
    print(f"Sign flip detected! MM found negative theta_0={mm_params[0]:.4f}")
    # Option A: Just flip the sign
    mm_flipped = torch.abs(mm_params)
    mm_flipped = mm_flipped / torch.norm(mm_flipped)
    evaluate(mm_flipped, "MM (abs)")

    # Option B: Project to positive orthant then normalize
    mm_proj = torch.clamp(mm_params, min=0.01)
    mm_proj = mm_proj / torch.norm(mm_proj)
    evaluate(mm_proj, "MM (proj+)")
else:
    print("No sign flip - MM found positive weights")
    evaluate(mm_params, "MM (no flip)")

# ============================================================================
# IDEA 3: Feature normalization (auto-scale to unit variance)
# ============================================================================
print("\n" + "=" * 70)
print("IDEA 3: Auto-normalized features")
print("=" * 70)

# Compute feature statistics from data
all_features = []
for traj in panel.trajectories:
    for t in range(len(traj)):
        s, a = traj.states[t].item(), traj.actions[t].item()
        all_features.append(feature_matrix[s, a].numpy())
all_features = np.array(all_features)
feat_mean = all_features.mean(axis=0)
feat_std = all_features.std(axis=0)
print(f"Feature mean: {feat_mean}")
print(f"Feature std:  {feat_std}")

# Normalize features to zero mean, unit variance
feat_norm = feature_matrix.clone()
for k in range(2):
    if feat_std[k] > 1e-6:
        feat_norm[:, :, k] = (feat_norm[:, :, k] - feat_mean[k]) / feat_std[k]

reward_fn_norm = ActionDependentReward(feature_matrix=feat_norm, parameter_names=["op_cost_norm", "repl_cost_norm"])

print(f"Normalized feature range: [{feat_norm.min():.2f}, {feat_norm.max():.2f}]")

mm_norm = MaxMarginIRLEstimator(max_iterations=50, margin_tol=1e-5, verbose=False)
result_mm_norm = mm_norm.estimate(panel=panel, utility=reward_fn_norm, problem=problem, transitions=trans_est)

# Need to transform params back to original scale
# If normalized: feat_norm = (feat - mean) / std
# Then: theta_orig = theta_norm / std (approximately, for linear rewards)
theta_norm = result_mm_norm.parameters
theta_orig_approx = theta_norm / torch.tensor(feat_std, dtype=torch.float32)
theta_orig_approx = theta_orig_approx / torch.norm(theta_orig_approx)

evaluate(theta_norm, "MM (norm feat)")
evaluate(theta_orig_approx, "MM (rescaled)")

# ============================================================================
# IDEA 4: Ensemble / Multiple initializations
# ============================================================================
print("\n" + "=" * 70)
print("IDEA 4: MCE with multiple initializations")
print("=" * 70)

best_ll = float('-inf')
best_params = None
inits = [
    torch.tensor([1.0, 0.0]),
    torch.tensor([0.0, 1.0]),
    torch.tensor([1.0, 1.0]) / np.sqrt(2),
    torch.tensor([0.5, 0.866]),  # 60 degrees
    torch.tensor([0.866, 0.5]),  # 30 degrees
]

for i, init in enumerate(inits):
    try:
        result = mce.estimate(
            panel=panel, utility=reward_fn, problem=problem,
            transitions=trans_est, initial_params=init.clone()
        )
        ll = result.log_likelihood
        if ll > best_ll:
            best_ll = ll
            best_params = result.parameters
        print(f"  Init {i}: LL={ll:.2f}, params={result.parameters.tolist()}")
    except:
        print(f"  Init {i}: failed")

if best_params is not None:
    evaluate(best_params, "MCE (best of 5)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"\nTrue: {true_params.tolist()}, angle={true_angle}°")
print(f"\nBaseline Max Margin cos sim: {baseline_cos:.4f}")
print(f"Idea 1 (MM→MCE) cos sim:     {idea1_cos:.4f}")
print("\nBest approach: Use Max Margin for quick direction estimate,")
print("then refine with MCE IRL for accurate parameters.")
