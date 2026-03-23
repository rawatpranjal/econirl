#!/usr/bin/env python3
"""Debug script for MCE IRL parameter recovery on Rust bus model.

This script investigates why MCE IRL fails to recover ground truth parameters:
- Ground truth: theta_c=0.001, RC=3.0
- MCE IRL estimates: theta_c=0.0198, RC=0.6126 (completely wrong)

Key diagnostics:
1. Sanity check: Does gradient = 0 at true parameters?
2. Feature expectations: Do empirical vs expected features match?
3. Gradient computation: Is the direction correct?
4. Scale issues: Are features properly scaled?
5. Objective landscape: Is there a unique minimum?
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Import from econirl
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.core.types import DDCProblem


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def compute_gradient_and_features(
    params: torch.Tensor,
    panel,
    reward_fn: ActionDependentReward,
    problem: DDCProblem,
    transitions: torch.Tensor,
    initial_dist: torch.Tensor,
    inner_tol: float = 1e-10,
    inner_max_iter: int = 10000,
    svf_tol: float = 1e-10,
    svf_max_iter: int = 5000,
) -> dict:
    """Compute gradient, empirical features, and expected features.

    Returns a dict with all the diagnostic information.
    """
    operator = SoftBellmanOperator(problem, transitions)

    # Compute reward matrix
    reward_matrix = reward_fn.compute(params)

    # Soft value iteration (backward pass)
    V = torch.zeros(problem.num_states, dtype=params.dtype)
    for i in range(inner_max_iter):
        result = operator.apply(reward_matrix, V)
        V_new = result.V
        delta = torch.abs(V_new - V).max().item()
        V = V_new
        if delta < inner_tol:
            inner_converged = True
            inner_iters = i + 1
            break
    else:
        inner_converged = False
        inner_iters = inner_max_iter

    policy = result.policy

    # State visitation frequencies (forward pass)
    n_states = problem.num_states
    gamma = problem.discount_factor

    # Compute P_pi
    P_pi = torch.einsum("sa,ast->st", policy, transitions)

    # Fixed point iteration for state visitation
    D = initial_dist.clone()
    rho0 = initial_dist.clone()

    for i in range(svf_max_iter):
        D_new = rho0 + gamma * (P_pi.T @ D)
        delta = torch.abs(D_new - D).max().item()
        D = D_new
        if delta < svf_tol:
            svf_converged = True
            svf_iters = i + 1
            break
    else:
        svf_converged = False
        svf_iters = svf_max_iter

    # Normalize to probability
    D_normalized = D / D.sum()

    # Compute empirical features
    feature_matrix = reward_fn.feature_matrix  # (n_states, n_actions, n_features)
    total_obs = sum(len(traj) for traj in panel.trajectories)
    n_features = feature_matrix.shape[2]
    empirical_sum = torch.zeros(n_features, dtype=feature_matrix.dtype)

    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            empirical_sum += feature_matrix[s, a, :]

    empirical_features = empirical_sum / total_obs if total_obs > 0 else empirical_sum

    # Compute expected features under policy
    # E[phi] = sum_s sum_a D(s) * pi(a|s) * phi(s,a,k)
    expected_features = torch.einsum("s,sa,sak->k", D_normalized, policy, feature_matrix)

    # Gradient: empirical - expected (for gradient ascent on likelihood)
    gradient = empirical_features - expected_features

    # Compute log-likelihood
    log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
    ll = 0.0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            state = traj.states[t].item()
            action = traj.actions[t].item()
            ll += log_probs[state, action].item()

    # Compute objective: ||empirical - expected||^2 / 2
    obj = 0.5 * torch.sum((empirical_features - expected_features) ** 2).item()

    return {
        "params": params.clone(),
        "reward_matrix": reward_matrix,
        "value_function": V,
        "policy": policy,
        "state_visitation": D_normalized,
        "empirical_features": empirical_features,
        "expected_features": expected_features,
        "gradient": gradient,
        "gradient_norm": torch.norm(gradient).item(),
        "feature_diff": torch.norm(empirical_features - expected_features).item(),
        "log_likelihood": ll,
        "objective": obj,
        "inner_converged": inner_converged,
        "inner_iters": inner_iters,
        "svf_converged": svf_converged,
        "svf_iters": svf_iters,
    }


def main():
    print_header("MCE IRL DEBUGGING SCRIPT")
    print("Investigating parameter recovery failure on Rust Bus model")

    # Ground truth parameters
    TRUE_THETA_C = 0.001
    TRUE_RC = 3.0
    DISCOUNT = 0.99

    print(f"\nGround Truth Parameters:")
    print(f"  operating_cost (theta_c): {TRUE_THETA_C}")
    print(f"  replacement_cost (RC):    {TRUE_RC}")
    print(f"  discount_factor (beta):   {DISCOUNT}")

    # Create environment
    env = RustBusEnvironment(
        operating_cost=TRUE_THETA_C,
        replacement_cost=TRUE_RC,
        num_mileage_bins=90,
        discount_factor=DISCOUNT,
    )

    # Simulate data
    panel = simulate_panel(env, n_individuals=100, n_periods=50, seed=42)
    n_obs = sum(len(t) for t in panel.trajectories)
    n_replace = sum((t.actions == 1).sum().item() for t in panel.trajectories)

    print(f"\nSimulated Data:")
    print(f"  Individuals:      {len(panel.trajectories)}")
    print(f"  Total obs:        {n_obs:,}")
    print(f"  Replacements:     {n_replace:,}")
    print(f"  Replacement rate: {n_replace/n_obs:.2%}")

    # Create reward function
    reward = ActionDependentReward.from_rust_environment(env)

    # Compute initial distribution from data
    n_states = env.num_states
    counts = torch.zeros(n_states, dtype=torch.float32)
    for traj in panel.trajectories:
        if len(traj) > 0:
            initial_state = traj.states[0].item()
            counts[initial_state] += 1
    initial_dist = counts / counts.sum()

    print(f"\nInitial State Distribution:")
    print(f"  Entropy: {-(initial_dist * torch.log(initial_dist + 1e-10)).sum():.3f}")
    print(f"  States with nonzero mass: {(initial_dist > 0).sum().item()}")
    print(f"  P(state=0): {initial_dist[0]:.4f}")

    # =========================================================================
    # TEST 1: Sanity check - gradient at true parameters
    # =========================================================================
    print_header("TEST 1: Gradient at True Parameters")

    true_params = torch.tensor([TRUE_THETA_C, TRUE_RC])
    result_true = compute_gradient_and_features(
        true_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )

    print(f"\nAt true parameters theta_c={TRUE_THETA_C}, RC={TRUE_RC}:")
    print(f"  Empirical features:  {result_true['empirical_features'].numpy()}")
    print(f"  Expected features:   {result_true['expected_features'].numpy()}")
    print(f"  Gradient:            {result_true['gradient'].numpy()}")
    print(f"  Gradient norm:       {result_true['gradient_norm']:.6f}")
    print(f"  Feature difference:  {result_true['feature_diff']:.6f}")
    print(f"  Log-likelihood:      {result_true['log_likelihood']:.2f}")
    print(f"  Objective (||grad||^2/2): {result_true['objective']:.6f}")

    if result_true['gradient_norm'] < 0.1:
        print("\n  [PASS] Gradient is near zero at true parameters!")
    else:
        print(f"\n  [FAIL] Gradient is NOT zero at true parameters!")
        print(f"         This suggests the optimization target doesn't have")
        print(f"         a stationary point at the true parameters!")

    print(f"\n  Inner loop converged: {result_true['inner_converged']} ({result_true['inner_iters']} iters)")
    print(f"  SVF converged: {result_true['svf_converged']} ({result_true['svf_iters']} iters)")

    # =========================================================================
    # TEST 2: Examine the feature matrix structure
    # =========================================================================
    print_header("TEST 2: Feature Matrix Analysis")

    fm = reward.feature_matrix
    print(f"\nFeature matrix shape: {fm.shape}")
    print(f"  (num_states, num_actions, num_features)")

    print(f"\nFeature ranges:")
    print(f"  Feature 0 (operating cost):")
    print(f"    Keep action:    min={fm[:, 0, 0].min():.1f}, max={fm[:, 0, 0].max():.1f}")
    print(f"    Replace action: min={fm[:, 1, 0].min():.1f}, max={fm[:, 1, 0].max():.1f}")
    print(f"  Feature 1 (replacement cost):")
    print(f"    Keep action:    min={fm[:, 0, 1].min():.1f}, max={fm[:, 0, 1].max():.1f}")
    print(f"    Replace action: min={fm[:, 1, 1].min():.1f}, max={fm[:, 1, 1].max():.1f}")

    print(f"\nSample feature values:")
    for s in [0, 10, 30, 50, 80]:
        print(f"  State {s:2d}: Keep={fm[s, 0].numpy()}, Replace={fm[s, 1].numpy()}")

    # Scale analysis
    keep_features_scale = torch.abs(fm[:, 0, :]).max(dim=0).values
    replace_features_scale = torch.abs(fm[:, 1, :]).max(dim=0).values

    print(f"\nFeature scale (max abs value):")
    print(f"  Keep:    {keep_features_scale.numpy()}")
    print(f"  Replace: {replace_features_scale.numpy()}")
    print(f"  Ratio (operating/replacement): {keep_features_scale[0] / max(replace_features_scale[1], 1e-10):.1f}x")

    # =========================================================================
    # TEST 3: Gradient at different parameter values
    # =========================================================================
    print_header("TEST 3: Gradient Landscape")

    test_params = [
        (0.001, 3.0, "True"),
        (0.0, 3.0, "Zero theta_c"),
        (0.001, 0.0, "Zero RC"),
        (0.0, 0.0, "Zero both"),
        (0.01, 1.0, "Initial guess"),
        (0.0198, 0.6126, "MCE IRL estimate"),
        (0.002, 6.0, "2x parameters"),
        (0.0005, 1.5, "0.5x parameters"),
    ]

    print(f"\n{'Parameters':<20} {'Gradient':<25} {'||grad||':>10} {'LL':>12} {'Obj':>10}")
    print("-" * 80)

    for theta_c, rc, name in test_params:
        params = torch.tensor([theta_c, rc])
        result = compute_gradient_and_features(
            params, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )
        grad_str = f"[{result['gradient'][0]:.4f}, {result['gradient'][1]:.4f}]"
        print(f"({theta_c:.4f}, {rc:.2f}) {name:<10} {grad_str:<25} {result['gradient_norm']:>10.4f} {result['log_likelihood']:>12.2f} {result['objective']:>10.4f}")

    # =========================================================================
    # TEST 4: Verify gradient direction
    # =========================================================================
    print_header("TEST 4: Verify Gradient Direction")

    # Starting from initial guess, check if gradient points toward true params
    init_params = torch.tensor([0.01, 1.0])
    result_init = compute_gradient_and_features(
        init_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )

    direction_to_true = true_params - init_params
    gradient = result_init['gradient']

    # Cosine similarity
    cos_sim = torch.dot(gradient, direction_to_true) / (
        torch.norm(gradient) * torch.norm(direction_to_true)
    )

    print(f"\nFrom initial params ({init_params.numpy()}):")
    print(f"  Direction to true params: {direction_to_true.numpy()}")
    print(f"  Gradient:                 {gradient.numpy()}")
    print(f"  Cosine similarity:        {cos_sim.item():.4f}")

    if cos_sim > 0:
        print(f"\n  [INFO] Gradient points TOWARDS true parameters (cos > 0)")
    else:
        print(f"\n  [WARNING] Gradient points AWAY from true parameters!")

    # =========================================================================
    # TEST 5: Run manual gradient ascent with verbose output
    # =========================================================================
    print_header("TEST 5: Manual Gradient Ascent")

    params = torch.tensor([0.01, 1.0])  # Initial guess
    lr = 0.05
    max_iter = 100

    print(f"\nStarting gradient ascent:")
    print(f"  Initial params: {params.numpy()}")
    print(f"  Learning rate:  {lr}")
    print(f"  Max iterations: {max_iter}")

    print(f"\n{'Iter':>5} {'theta_c':>10} {'RC':>10} {'||grad||':>12} {'LL':>12} {'Obj':>10} {'Converged':>10}")
    print("-" * 80)

    for i in range(max_iter):
        result = compute_gradient_and_features(
            params, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        if i % 10 == 0 or i < 5:
            conv_str = "Y" if result['inner_converged'] and result['svf_converged'] else "N"
            print(f"{i:>5} {params[0].item():>10.6f} {params[1].item():>10.4f} {result['gradient_norm']:>12.6f} {result['log_likelihood']:>12.2f} {result['objective']:>10.6f} {conv_str:>10}")

        # Check convergence
        if result['gradient_norm'] < 1e-6:
            print(f"\nConverged at iteration {i}!")
            break

        # Gradient step
        params = params + lr * result['gradient']

    print(f"\nFinal params: theta_c={params[0].item():.6f}, RC={params[1].item():.4f}")
    print(f"True params:  theta_c={TRUE_THETA_C:.6f}, RC={TRUE_RC:.4f}")

    # =========================================================================
    # TEST 6: Check state visitation distribution
    # =========================================================================
    print_header("TEST 6: State Visitation Analysis")

    # Compute state visitation at true params
    result_true = compute_gradient_and_features(
        true_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )

    D = result_true['state_visitation']
    policy = result_true['policy']

    print(f"\nState visitation under true policy:")
    print(f"  Top 5 most visited states:")
    top_states = torch.argsort(D, descending=True)[:5]
    for s in top_states:
        print(f"    State {s.item():2d}: D={D[s].item():.4f}, P(replace|s)={policy[s, 1].item():.4f}")

    # Empirical state frequencies
    state_counts = torch.zeros(n_states)
    action_counts = torch.zeros((n_states, 2))
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s = traj.states[t].item()
            a = traj.actions[t].item()
            state_counts[s] += 1
            action_counts[s, a] += 1

    empirical_state_freq = state_counts / state_counts.sum()

    print(f"\n  Empirical state frequencies (from data):")
    top_empirical = torch.argsort(empirical_state_freq, descending=True)[:5]
    for s in top_empirical:
        emp_replace = action_counts[s, 1] / max(state_counts[s], 1)
        print(f"    State {s.item():2d}: freq={empirical_state_freq[s].item():.4f}, P(replace|s)={emp_replace.item():.4f}")

    # Compare model vs empirical
    print(f"\n  Model vs Empirical state distribution:")
    print(f"    KL divergence: {(D * torch.log(D / (empirical_state_freq + 1e-10) + 1e-10)).sum().item():.4f}")
    print(f"    L1 distance:   {torch.abs(D - empirical_state_freq).sum().item():.4f}")

    # =========================================================================
    # TEST 7: Check for identification issues
    # =========================================================================
    print_header("TEST 7: Identification Check")

    # The Rust model has a subtle identification issue:
    # Only the RATIO of theta_c/RC is identified, not the absolute levels
    # This is because the model can scale both parameters proportionally

    print("\nChecking if parameters are identified or just the ratio:")

    scale_factors = [0.5, 1.0, 2.0, 4.0]
    print(f"\n{'Scale':>8} {'theta_c':>10} {'RC':>10} {'LL':>12} {'Policy MAE':>12}")
    print("-" * 55)

    # First compute true policy for comparison
    result_true = compute_gradient_and_features(
        true_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )
    true_policy = result_true['policy']

    for scale in scale_factors:
        scaled_params = true_params * scale
        result = compute_gradient_and_features(
            scaled_params, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )
        policy_mae = torch.abs(result['policy'] - true_policy).mean().item()
        print(f"{scale:>8.1f} {scaled_params[0].item():>10.6f} {scaled_params[1].item():>10.4f} {result['log_likelihood']:>12.2f} {policy_mae:>12.6f}")

    print("\nNote: If likelihood changes significantly with scale but policy doesn't,")
    print("      then parameters ARE identified through the likelihood, not just")
    print("      through feature matching.")

    # =========================================================================
    # TEST 8: Numerical gradient check
    # =========================================================================
    print_header("TEST 8: Numerical Gradient Check")

    params = torch.tensor([TRUE_THETA_C, TRUE_RC])
    eps = 1e-5

    # Analytical gradient
    result = compute_gradient_and_features(
        params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )
    analytical_grad = result['gradient']

    # Numerical gradient (using objective = ||empirical - expected||^2 / 2)
    # Note: We want to MINIMIZE this, so gradient descent on objective
    # But MCE IRL does gradient ASCENT on likelihood
    numerical_grad = torch.zeros(2)

    for i in range(2):
        params_plus = params.clone()
        params_plus[i] += eps
        result_plus = compute_gradient_and_features(
            params_plus, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        params_minus = params.clone()
        params_minus[i] -= eps
        result_minus = compute_gradient_and_features(
            params_minus, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        # Gradient of objective (minimize)
        numerical_grad[i] = (result_plus['objective'] - result_minus['objective']) / (2 * eps)

    print(f"\nAt true parameters:")
    print(f"  Analytical gradient (feature matching): {analytical_grad.numpy()}")
    print(f"  Numerical gradient (of objective):      {-numerical_grad.numpy()}")  # Negate for ascent
    print(f"  Difference: {(analytical_grad + numerical_grad).numpy()}")  # Should match with negation

    # Now check with likelihood gradient
    numerical_ll_grad = torch.zeros(2)
    for i in range(2):
        params_plus = params.clone()
        params_plus[i] += eps
        result_plus = compute_gradient_and_features(
            params_plus, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        params_minus = params.clone()
        params_minus[i] -= eps
        result_minus = compute_gradient_and_features(
            params_minus, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        numerical_ll_grad[i] = (result_plus['log_likelihood'] - result_minus['log_likelihood']) / (2 * eps)

    print(f"\n  Numerical LL gradient: {numerical_ll_grad.numpy()}")
    print(f"\nNote: Feature matching gradient != LL gradient in general!")
    print(f"      MCE IRL uses feature matching, NFXP uses LL.")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_header("SUMMARY")

    print("""
KEY FINDINGS:

1. Gradient at true parameters:
   - If non-zero: The feature matching objective doesn't have a stationary
     point at true parameters. This is a FUNDAMENTAL ISSUE with the approach.

2. Feature scale mismatch:
   - Feature 0 (operating cost) ranges from 0 to -89
   - Feature 1 (replacement cost) is fixed at -1
   - This creates poor conditioning and slow convergence

3. Identification:
   - The Rust model's true identifying variation comes from the LIKELIHOOD,
     not just feature matching
   - MCE IRL's feature matching may not fully utilize this variation

4. Potential fixes:
   a) Use likelihood-based optimization (like NFXP) instead of feature matching
   b) Normalize features to similar scales
   c) Use adaptive learning rates per parameter
   d) Consider if the model is even suitable for MCE IRL

5. The key issue may be that MCE IRL's objective (feature matching) is
   fundamentally different from maximum likelihood, and the stationary
   points don't coincide for the Rust model.
""")


if __name__ == "__main__":
    main()
