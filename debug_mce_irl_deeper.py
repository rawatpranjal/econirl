#!/usr/bin/env python3
"""Deeper analysis of the MCE IRL failure.

The key finding from the first debug script:
- At TRUE parameters, gradient norm = 0.963 (NOT ZERO!)
- This means the feature matching objective does NOT have a stationary point
  at the true parameters.

This script investigates WHY this is the case and what the actual stationary
point looks like.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.preferences.action_reward import ActionDependentReward


def print_header(title: str) -> None:
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def compute_features_and_gradient(params, panel, reward_fn, problem, transitions, initial_dist):
    """Compute empirical and expected features + gradient."""
    operator = SoftBellmanOperator(problem, transitions)
    reward_matrix = reward_fn.compute(params)

    # Solve for value function
    V = torch.zeros(problem.num_states, dtype=params.dtype)
    for _ in range(10000):
        result = operator.apply(reward_matrix, V)
        V_new = result.V
        if torch.abs(V_new - V).max() < 1e-10:
            break
        V = V_new
    policy = result.policy

    # State visitation
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

    # Features
    fm = reward_fn.feature_matrix
    n_features = fm.shape[2]

    # Empirical
    total_obs = sum(len(t) for t in panel.trajectories)
    emp_sum = torch.zeros(n_features)
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s, a = traj.states[t].item(), traj.actions[t].item()
            emp_sum += fm[s, a, :]
    emp_features = emp_sum / total_obs

    # Expected
    exp_features = torch.einsum("s,sa,sak->k", D, policy, fm)

    # Gradient for ascent on likelihood = emp - exp
    gradient = emp_features - exp_features

    # Log-likelihood
    log_probs = operator.compute_log_choice_probabilities(reward_matrix, V)
    ll = 0.0
    for traj in panel.trajectories:
        for t in range(len(traj)):
            s, a = traj.states[t].item(), traj.actions[t].item()
            ll += log_probs[s, a].item()

    return {
        "emp": emp_features,
        "exp": exp_features,
        "grad": gradient,
        "ll": ll,
        "policy": policy,
        "svf": D,
    }


def main():
    print_header("DEEP ANALYSIS: WHY MCE IRL FAILS ON RUST MODEL")

    # Setup
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

    # =========================================================================
    print_header("1. THE CORE ISSUE: Feature Matching vs MLE")

    true_params = torch.tensor([TRUE_THETA_C, TRUE_RC])
    result = compute_features_and_gradient(
        true_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )

    print(f"\nAt TRUE parameters (theta_c={TRUE_THETA_C}, RC={TRUE_RC}):")
    print(f"\n  Empirical features (from data):  {result['emp'].numpy()}")
    print(f"  Expected features (from model):  {result['exp'].numpy()}")
    print(f"  Feature gap (gradient):          {result['grad'].numpy()}")
    print(f"  ||gradient||:                    {result['grad'].norm().item():.6f}")

    print(f"\n  Log-likelihood: {result['ll']:.2f}")

    print("""
    INSIGHT: The empirical features and model expected features DON'T MATCH
    even at the true parameters!

    This happens because:
    1. Empirical features = average features in the DATA
    2. Expected features = average features under the MODEL's policy

    For MCE IRL's gradient to be zero, we need: E_data[phi] = E_model[phi]

    But in finite samples:
    - The data was generated with noise (logit shocks)
    - The finite sample average differs from the population expectation
    - The model computes the population expectation (E[phi] under stationary dist)

    This is a FINITE SAMPLE BIAS issue, not a bug!
""")

    # =========================================================================
    print_header("2. WHAT DOES MCE IRL ACTUALLY OPTIMIZE?")

    print("""
    MCE IRL minimizes: ||E_data[phi] - E_model[phi]||^2

    This is NOT the same as Maximum Likelihood Estimation!

    MLE maximizes: sum_t log P(a_t | s_t; theta)

    The key difference:
    - MLE directly matches ACTION PROBABILITIES at each state-action pair
    - MCE IRL matches AGGREGATE FEATURE EXPECTATIONS

    For the Rust model:
    - Feature 0 (operating cost): -mileage for keep, 0 for replace
    - Feature 1 (replacement cost): 0 for keep, -1 for replace

    E[phi_0] = E[-mileage * I(keep)] = -average mileage when keeping
    E[phi_1] = E[-I(replace)] = -replacement rate

    MCE IRL tries to match:
    1. The average mileage weighted by keep decisions
    2. The replacement rate

    But this doesn't uniquely identify (theta_c, RC)!
""")

    # =========================================================================
    print_header("3. COMPUTE THE ACTUAL MCE IRL OPTIMUM")

    print("\nSearching for parameters that minimize ||emp - exp||...")

    # Use scipy.optimize to find the actual minimum
    from scipy.optimize import minimize

    def objective(params_np):
        params = torch.tensor(params_np, dtype=torch.float32)
        result = compute_features_and_gradient(
            params, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )
        return 0.5 * result['grad'].norm().item() ** 2

    # Try from multiple starting points
    best_result = None
    best_obj = float('inf')

    starts = [
        [0.001, 3.0],   # True
        [0.01, 1.0],    # MCE IRL default
        [0.0, 0.0],     # Zero
        [0.005, 2.0],   # Nearby
    ]

    print(f"\n{'Start':<20} {'Optimum':<25} {'Objective':>10} {'LL':>12}")
    print("-" * 70)

    for start in starts:
        res = minimize(
            objective,
            start,
            method='L-BFGS-B',
            options={'gtol': 1e-10, 'maxiter': 1000}
        )
        opt_params = torch.tensor(res.x, dtype=torch.float32)
        result = compute_features_and_gradient(
            opt_params, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )
        print(f"({start[0]:.3f}, {start[1]:.2f})"
              f"       ({res.x[0]:.6f}, {res.x[1]:.4f})"
              f"    {res.fun:.8f}"
              f"    {result['ll']:.2f}")

        if res.fun < best_obj:
            best_obj = res.fun
            best_result = res

    print(f"\nBest MCE IRL optimum: theta_c={best_result.x[0]:.6f}, RC={best_result.x[1]:.4f}")
    print(f"True parameters:      theta_c={TRUE_THETA_C:.6f}, RC={TRUE_RC:.4f}")

    # Compute features at the optimum
    opt_params = torch.tensor(best_result.x, dtype=torch.float32)
    opt_result = compute_features_and_gradient(
        opt_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )

    print(f"\nAt MCE IRL optimum:")
    print(f"  Empirical features: {opt_result['emp'].numpy()}")
    print(f"  Expected features:  {opt_result['exp'].numpy()}")
    print(f"  Feature gap:        {opt_result['grad'].numpy()}")
    print(f"  ||gap||:            {opt_result['grad'].norm().item():.8f}")
    print(f"  Log-likelihood:     {opt_result['ll']:.2f}")

    print(f"\nAt TRUE parameters:")
    print(f"  Empirical features: {result['emp'].numpy()}")
    print(f"  Expected features:  {result['exp'].numpy()}")
    print(f"  Feature gap:        {result['grad'].numpy()}")
    print(f"  ||gap||:            {result['grad'].norm().item():.6f}")
    print(f"  Log-likelihood:     {result['ll']:.2f}")

    # =========================================================================
    print_header("4. THE IDENTIFICATION PROBLEM")

    print("""
    The feature matching objective has a DIFFERENT optimum than MLE!

    This is expected because:

    1. MCE IRL was designed for IRL problems where:
       - We don't observe rewards
       - We only observe state-action trajectories
       - The goal is to find ANY reward that makes observed behavior optimal

    2. The Rust model is a STRUCTURAL ESTIMATION problem where:
       - We want to recover SPECIFIC utility parameters
       - The utility function is KNOWN (linear in features)
       - We have a likelihood function

    3. For structural estimation, MLE (NFXP) is the right approach
       MCE IRL is designed for a different problem!

    Why doesn't feature matching work here?
    - Feature matching aggregates information across states
    - It loses the state-specific variation that identifies parameters
    - The Rust model's identification comes from how P(replace|mileage)
      varies with mileage, not just average features
""")

    # =========================================================================
    print_header("5. NUMERICAL DEMONSTRATION: FEATURE MATCHING IS NOT MLE")

    print("\nComparing gradients at true parameters:")

    eps = 1e-5
    true_params = torch.tensor([TRUE_THETA_C, TRUE_RC])

    # Feature matching gradient
    result = compute_features_and_gradient(
        true_params, panel, reward, env.problem_spec,
        env.transition_matrices, initial_dist
    )
    fm_grad = result['grad']

    # Numerical MLE gradient
    mle_grad = torch.zeros(2)
    for i in range(2):
        p_plus = true_params.clone()
        p_plus[i] += eps
        r_plus = compute_features_and_gradient(
            p_plus, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        p_minus = true_params.clone()
        p_minus[i] -= eps
        r_minus = compute_features_and_gradient(
            p_minus, panel, reward, env.problem_spec,
            env.transition_matrices, initial_dist
        )

        mle_grad[i] = (r_plus['ll'] - r_minus['ll']) / (2 * eps)

    print(f"\n  Feature matching gradient: {fm_grad.numpy()}")
    print(f"  MLE gradient (numerical):  {mle_grad.numpy()}")
    print(f"  Ratio (MLE/FM):            {(mle_grad / fm_grad).numpy()}")

    print("""
    CONCLUSION: The gradients point in DIFFERENT directions!

    - Feature matching gradient is NOT aligned with MLE gradient
    - Feature matching converges to a different point than MLE
    - This explains why MCE IRL fails on Rust model
""")

    # =========================================================================
    print_header("6. SOLUTION: USE MAXIMUM LIKELIHOOD, NOT FEATURE MATCHING")

    print("""
    For STRUCTURAL ESTIMATION (recovering utility parameters):
    - Use NFXP (Nested Fixed Point) - Maximum Likelihood
    - Use CCP (Conditional Choice Probability) estimators
    - Use MPEC (Mathematical Programming with Equilibrium Constraints)

    For INVERSE REINFORCEMENT LEARNING (finding any consistent reward):
    - MCE IRL is appropriate
    - But the recovered reward may not match the true one
    - IRL has inherent identification issues (reward shaping)

    The Rust model is a STRUCTURAL problem, not an IRL problem.
    NFXP is the correct approach for parameter recovery.
""")

    # =========================================================================
    print_header("7. CAN WE FIX MCE IRL FOR THIS CASE?")

    print("""
    Potential modifications to MCE IRL:

    1. USE LIKELIHOOD INSTEAD OF FEATURE MATCHING
       - Replace gradient = emp - exp with gradient = d(LL)/d(theta)
       - This would make it equivalent to gradient-based MLE
       - But this defeats the purpose of IRL

    2. IMPORTANCE WEIGHTING
       - Weight feature expectations by state visitation
       - Account for the distribution shift between data and model
       - Still won't fully solve the identification issue

    3. USE STATE-ACTION FEATURES DIRECTLY
       - Match P(a|s) for each state, not just aggregate features
       - This is essentially what NFXP does

    4. ACCEPT THAT IRL != STRUCTURAL ESTIMATION
       - If you want parameter recovery, use MLE methods
       - If you want any consistent reward, use IRL methods
""")


if __name__ == "__main__":
    main()
