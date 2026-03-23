#!/usr/bin/env python3
"""
Compare NFXP vs MCE IRL vs Max Margin IRL on the same DGP.

Key design choices:
1. Normalize state features to [-1, 1] scale
2. Use unit-norm reward weights for DGP
3. Estimate with all three methods
4. Compare recovery quality
"""

import torch
import numpy as np
from dataclasses import dataclass

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.core.bellman import SoftBellmanOperator
from econirl.preferences.action_reward import ActionDependentReward
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimation.max_margin_irl import MaxMarginIRLEstimator


@dataclass
class DGPConfig:
    """Configuration for the Data Generating Process."""
    n_states: int = 50
    n_actions: int = 2
    gamma: float = 0.95
    n_trajectories: int = 200
    n_periods: int = 100
    seed: int = 42


def create_normalized_problem(config: DGPConfig):
    """
    Create a Rust-style bus replacement problem with NORMALIZED features.

    Features are scaled to [-1, 1]:
    - Feature 0: mileage cost for keep action, normalized to [-1, 0]
    - Feature 1: replacement cost for replace action, fixed at -1
    """
    n_states = config.n_states

    # Feature matrix: (n_states, n_actions, n_features)
    feature_matrix = torch.zeros((n_states, 2, 2), dtype=torch.float32)

    for s in range(n_states):
        # Normalize mileage to [-1, 0] range
        # State 0 -> 0 (no cost), State n_states-1 -> -1 (max cost)
        normalized_mileage = -s / (n_states - 1)

        # Keep action: incur mileage cost
        feature_matrix[s, 0, 0] = normalized_mileage  # Operating cost feature
        feature_matrix[s, 0, 1] = 0.0  # No replacement cost

        # Replace action: no mileage cost (reset), but pay replacement cost
        feature_matrix[s, 1, 0] = 0.0  # No operating cost
        feature_matrix[s, 1, 1] = -1.0  # Fixed replacement cost

    # Transition matrices: (n_states, n_actions, n_states) for DDC
    transitions = torch.zeros((n_states, 2, n_states), dtype=torch.float32)

    # Keep: mileage increases by 0, 1, or 2
    for s in range(n_states):
        for delta, prob in [(0, 0.35), (1, 0.35), (2, 0.30)]:
            next_s = min(s + delta, n_states - 1)
            transitions[s, 0, next_s] += prob

    # Replace: reset to state 0
    transitions[:, 1, 0] = 1.0

    # Convert to estimator format: (n_actions, n_states, n_states)
    transitions_estimator = transitions.permute(1, 0, 2)

    # Problem spec
    problem = DDCProblem(
        num_states=n_states,
        num_actions=2,
        discount_factor=config.gamma,
    )

    # Reward function for MCE IRL and Max Margin
    reward_fn = ActionDependentReward(
        feature_matrix=feature_matrix,
        parameter_names=["operating_cost", "replacement_cost"],
    )

    # Utility function for NFXP (same structure)
    utility_fn = LinearUtility(
        feature_matrix=feature_matrix,
        parameter_names=["operating_cost", "replacement_cost"],
    )

    return problem, transitions, transitions_estimator, feature_matrix, reward_fn, utility_fn


def compute_optimal_policy(feature_matrix, params, transitions, gamma, max_iter=50000, tol=1e-10):
    """Soft value iteration to get optimal policy."""
    reward_matrix = torch.einsum("sak,k->sa", feature_matrix, params)
    n_states = reward_matrix.shape[0]
    V = torch.zeros(n_states, dtype=torch.float32)

    for _ in range(max_iter):
        # Q(s,a) = R(s,a) + gamma * sum_s' P(s'|s,a) V(s')
        Q = reward_matrix + gamma * torch.einsum("sat,t->sa", transitions, V)
        V_new = torch.logsumexp(Q, dim=1)
        if torch.abs(V_new - V).max() < tol:
            V = V_new
            break
        V = V_new

    Q = reward_matrix + gamma * torch.einsum("sat,t->sa", transitions, V)
    policy = torch.softmax(Q, dim=1)
    return V, policy


def simulate_panel(policy, transitions, config: DGPConfig) -> Panel:
    """Simulate panel data from policy."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    n_states, n_actions = policy.shape
    trajectories = []

    for _ in range(config.n_trajectories):
        states, actions, next_states = [], [], []
        state = 0

        for _ in range(config.n_periods):
            states.append(state)
            action = np.random.choice(n_actions, p=policy[state].numpy())
            actions.append(action)
            next_state = np.random.choice(n_states, p=transitions[state, action].numpy())
            next_states.append(next_state)
            state = next_state

        trajectories.append(Trajectory(
            states=torch.tensor(states),
            actions=torch.tensor(actions),
            next_states=torch.tensor(next_states),
        ))

    return Panel(trajectories=trajectories)


def run_experiment(true_angle_deg: float, config: DGPConfig):
    """Run a single experiment with given true parameters."""

    # True parameters on unit circle
    true_angle = np.radians(true_angle_deg)
    true_params = torch.tensor([np.cos(true_angle), np.sin(true_angle)], dtype=torch.float32)

    print(f"\n{'='*80}")
    print(f"EXPERIMENT: True angle = {true_angle_deg}°")
    print(f"True params: [{true_params[0]:.4f}, {true_params[1]:.4f}]")
    print(f"Parameter ratio (θ₀/θ₁): {true_params[0]/true_params[1]:.4f}")
    print(f"{'='*80}")

    # Create problem with normalized features
    problem, transitions, transitions_est, feature_matrix, reward_fn, utility_fn = \
        create_normalized_problem(config)

    print(f"\nNormalized features:")
    print(f"  Feature range: [{feature_matrix.min():.2f}, {feature_matrix.max():.2f}]")
    print(f"  Keep action mileage cost at state 0: {feature_matrix[0, 0, 0]:.3f}")
    print(f"  Keep action mileage cost at state {config.n_states-1}: {feature_matrix[-1, 0, 0]:.3f}")
    print(f"  Replace action cost: {feature_matrix[0, 1, 1]:.3f}")

    # Compute true policy and simulate data
    V_true, policy_true = compute_optimal_policy(feature_matrix, true_params, transitions, config.gamma)

    print(f"\nTrue policy P(replace):")
    for s in [0, config.n_states//4, config.n_states//2, 3*config.n_states//4, config.n_states-1]:
        print(f"  State {s}: {policy_true[s, 1]:.4f}")

    panel = simulate_panel(policy_true, transitions, config)
    n_obs = sum(len(t) for t in panel.trajectories)

    # Count actions
    n_keep = sum((t.actions == 0).sum().item() for t in panel.trajectories)
    n_replace = sum((t.actions == 1).sum().item() for t in panel.trajectories)
    print(f"\nSimulated {n_obs} observations: {n_keep} keep ({100*n_keep/n_obs:.1f}%), {n_replace} replace ({100*n_replace/n_obs:.1f}%)")

    results = {
        'true_angle': true_angle_deg,
        'true_params': true_params,
    }

    # =========================================================================
    # NFXP Estimation
    # =========================================================================
    print(f"\n--- NFXP Estimation ---")

    nfxp = NFXPEstimator(
        se_method="asymptotic",
        optimizer="L-BFGS-B",
        inner_tol=1e-10,
        outer_tol=1e-8,
        verbose=False,
    )

    try:
        result_nfxp = nfxp.estimate(
            panel=panel,
            utility=utility_fn,
            problem=problem,
            transitions=transitions_est,
        )

        nfxp_params = result_nfxp.parameters
        nfxp_angle = np.degrees(np.arctan2(nfxp_params[1].item(), nfxp_params[0].item()))
        nfxp_norm = nfxp_params / torch.norm(nfxp_params)

        print(f"NFXP params: [{nfxp_params[0]:.6f}, {nfxp_params[1]:.6f}]")
        print(f"NFXP angle: {nfxp_angle:.2f}° (error: {abs(nfxp_angle - true_angle_deg):.4f}°)")

        results['nfxp_params'] = nfxp_params
        results['nfxp_angle'] = nfxp_angle
        results['nfxp_cos'] = torch.dot(nfxp_norm, true_params).item()
    except Exception as e:
        print(f"NFXP failed: {e}")
        results['nfxp_params'] = None

    # =========================================================================
    # MCE IRL Estimation
    # =========================================================================
    print(f"\n--- MCE IRL Estimation ---")

    mce_config = MCEIRLConfig(
        verbose=False,
        outer_max_iter=200,
        learning_rate=0.5,  # Faster learning
        outer_tol=1e-6,
        inner_tol=1e-8,
        inner_max_iter=5000,
        compute_se=False,
    )
    mce = MCEIRLEstimator(config=mce_config)

    # Initialize from unit vector
    init_params = torch.ones(2) / np.sqrt(2)

    try:
        result_mce = mce.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions_est,
            initial_params=init_params.clone(),
        )

        mce_params = result_mce.parameters
        mce_angle = np.degrees(np.arctan2(mce_params[1].item(), mce_params[0].item()))
        mce_norm = mce_params / torch.norm(mce_params)

        print(f"MCE params: [{mce_params[0]:.6f}, {mce_params[1]:.6f}]")
        print(f"MCE angle: {mce_angle:.2f}° (error: {abs(mce_angle - true_angle_deg):.4f}°)")

        results['mce_params'] = mce_params
        results['mce_angle'] = mce_angle
        results['mce_cos'] = torch.dot(mce_norm, true_params).item()
    except Exception as e:
        print(f"MCE IRL failed: {e}")
        results['mce_params'] = None

    # =========================================================================
    # Max Margin IRL Estimation (Unit Norm)
    # =========================================================================
    print(f"\n--- Max Margin IRL Estimation (Unit Norm) ---")

    mm_estimator = MaxMarginIRLEstimator(
        max_iterations=100,
        margin_tol=1e-6,
        verbose=False,
        anchor_idx=None,  # Unit norm constraint
    )

    try:
        result_mm = mm_estimator.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions_est,
        )

        mm_params = result_mm.parameters
        mm_angle = np.degrees(np.arctan2(mm_params[1].item(), mm_params[0].item()))
        mm_norm = mm_params / torch.norm(mm_params)

        print(f"MM params: [{mm_params[0]:.6f}, {mm_params[1]:.6f}]")
        print(f"MM angle: {mm_angle:.2f}° (error: {abs(mm_angle - true_angle_deg):.4f}°)")
        print(f"Margin: {result_mm.metadata.get('margin', 'N/A'):.6f}")

        results['mm_params'] = mm_params
        results['mm_angle'] = mm_angle
        results['mm_cos'] = torch.dot(mm_norm, true_params).item()
    except Exception as e:
        print(f"Max Margin IRL failed: {e}")
        results['mm_params'] = None

    # =========================================================================
    # Max Margin IRL Estimation (Anchor Normalization)
    # =========================================================================
    print(f"\n--- Max Margin IRL Estimation (Anchor θ₁=1) ---")

    mm_anchor = MaxMarginIRLEstimator(
        max_iterations=100,
        margin_tol=1e-6,
        verbose=False,
        anchor_idx=1,  # Fix θ₁ = 1
    )

    try:
        result_mm_anchor = mm_anchor.estimate(
            panel=panel,
            utility=reward_fn,
            problem=problem,
            transitions=transitions_est,
        )

        mm_anchor_params = result_mm_anchor.parameters
        mm_anchor_normalized = mm_anchor_params / torch.norm(mm_anchor_params)
        mm_anchor_angle = np.degrees(np.arctan2(mm_anchor_normalized[1].item(), mm_anchor_normalized[0].item()))

        print(f"MM-Anchor raw: [{mm_anchor_params[0]:.6f}, {mm_anchor_params[1]:.6f}]")
        print(f"MM-Anchor normalized: [{mm_anchor_normalized[0]:.6f}, {mm_anchor_normalized[1]:.6f}]")
        print(f"MM-Anchor angle: {mm_anchor_angle:.2f}° (error: {abs(mm_anchor_angle - true_angle_deg):.4f}°)")

        results['mm_anchor_params'] = mm_anchor_params
        results['mm_anchor_normalized'] = mm_anchor_normalized
        results['mm_anchor_angle'] = mm_anchor_angle
        results['mm_anchor_cos'] = torch.dot(mm_anchor_normalized, true_params).item()
    except Exception as e:
        print(f"Max Margin Anchor failed: {e}")
        results['mm_anchor_params'] = None

    return results


def main():
    print("=" * 80)
    print("COMPARISON: NFXP vs MCE IRL vs Max Margin IRL")
    print("Normalized Features ([-1, 1] scale), Unit-Norm Parameters")
    print("=" * 80)

    config = DGPConfig(
        n_states=30,  # Fewer states for faster computation
        gamma=0.95,
        n_trajectories=100,  # Fewer trajectories
        n_periods=50,  # Fewer periods
        seed=42,
    )

    print(f"\nDGP Config:")
    print(f"  States: {config.n_states}")
    print(f"  Discount: {config.gamma}")
    print(f"  Trajectories: {config.n_trajectories}")
    print(f"  Periods: {config.n_periods}")
    print(f"  Total obs: {config.n_trajectories * config.n_periods}")

    # Test at different true angles (different operating/replacement cost trade-offs)
    # 45° means equal weights, lower angles favor replacement, higher favor operating
    angles = [30, 45, 60]  # Reduced for faster testing
    results = []

    for angle in angles:
        result = run_experiment(angle, config)
        results.append(result)

    # =========================================================================
    # Summary Table
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: Recovery Quality Across Experiments")
    print("=" * 80)

    print(f"\n{'Angle':<8} {'NFXP':<12} {'MCE IRL':<12} {'MaxMargin':<12} {'MM-Anchor':<12}")
    print(f"{'(deg)':<8} {'Angle Err':<12} {'Angle Err':<12} {'Angle Err':<12} {'Angle Err':<12}")
    print("-" * 56)

    nfxp_errors = []
    mce_errors = []
    mm_errors = []
    mm_anchor_errors = []

    for r in results:
        true_angle = r['true_angle']

        nfxp_err = abs(r.get('nfxp_angle', float('nan')) - true_angle) if r.get('nfxp_params') is not None else float('nan')
        mce_err = abs(r.get('mce_angle', float('nan')) - true_angle) if r.get('mce_params') is not None else float('nan')
        mm_err = abs(r.get('mm_angle', float('nan')) - true_angle) if r.get('mm_params') is not None else float('nan')
        mm_anchor_err = abs(r.get('mm_anchor_angle', float('nan')) - true_angle) if r.get('mm_anchor_params') is not None else float('nan')

        print(f"{true_angle:<8.0f} {nfxp_err:<12.4f} {mce_err:<12.4f} {mm_err:<12.4f} {mm_anchor_err:<12.4f}")

        if not np.isnan(nfxp_err): nfxp_errors.append(nfxp_err)
        if not np.isnan(mce_err): mce_errors.append(mce_err)
        if not np.isnan(mm_err): mm_errors.append(mm_err)
        if not np.isnan(mm_anchor_err): mm_anchor_errors.append(mm_anchor_err)

    print("-" * 56)
    print(f"{'AVERAGE':<8} {np.mean(nfxp_errors):<12.4f} {np.mean(mce_errors):<12.4f} "
          f"{np.mean(mm_errors):<12.4f} {np.mean(mm_anchor_errors):<12.4f}")

    # Cosine similarities
    print(f"\n{'Angle':<8} {'NFXP':<12} {'MCE IRL':<12} {'MaxMargin':<12} {'MM-Anchor':<12}")
    print(f"{'(deg)':<8} {'Cos Sim':<12} {'Cos Sim':<12} {'Cos Sim':<12} {'Cos Sim':<12}")
    print("-" * 56)

    for r in results:
        true_angle = r['true_angle']
        nfxp_cos = r.get('nfxp_cos', float('nan'))
        mce_cos = r.get('mce_cos', float('nan'))
        mm_cos = r.get('mm_cos', float('nan'))
        mm_anchor_cos = r.get('mm_anchor_cos', float('nan'))

        print(f"{true_angle:<8.0f} {nfxp_cos:<12.4f} {mce_cos:<12.4f} {mm_cos:<12.4f} {mm_anchor_cos:<12.4f}")

    print("-" * 56)
    avg_nfxp_cos = np.mean([r.get('nfxp_cos', float('nan')) for r in results if r.get('nfxp_params') is not None])
    avg_mce_cos = np.mean([r.get('mce_cos', float('nan')) for r in results if r.get('mce_params') is not None])
    avg_mm_cos = np.mean([r.get('mm_cos', float('nan')) for r in results if r.get('mm_params') is not None])
    avg_mm_anchor_cos = np.mean([r.get('mm_anchor_cos', float('nan')) for r in results if r.get('mm_anchor_params') is not None])
    print(f"{'AVERAGE':<8} {avg_nfxp_cos:<12.4f} {avg_mce_cos:<12.4f} {avg_mm_cos:<12.4f} {avg_mm_anchor_cos:<12.4f}")

    # =========================================================================
    # Conclusions
    # =========================================================================
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    print("\n1. ANGLE ERROR (lower is better):")
    print(f"   NFXP:        {np.mean(nfxp_errors):.4f}° average")
    print(f"   MCE IRL:     {np.mean(mce_errors):.4f}° average")
    print(f"   Max Margin:  {np.mean(mm_errors):.4f}° average")
    print(f"   MM-Anchor:   {np.mean(mm_anchor_errors):.4f}° average")

    print("\n2. COSINE SIMILARITY (1.0 is perfect, -1.0 is opposite):")
    print(f"   NFXP:        {avg_nfxp_cos:.4f}")
    print(f"   MCE IRL:     {avg_mce_cos:.4f}")
    print(f"   Max Margin:  {avg_mm_cos:.4f}")
    print(f"   MM-Anchor:   {avg_mm_anchor_cos:.4f}")

    print("\n3. KEY FINDINGS:")
    if np.mean(mce_errors) < 2.0 and avg_mce_cos > 0.99:
        print("   ✓ MCE IRL achieves excellent parameter recovery")
    elif avg_mce_cos > 0.95:
        print("   ✓ MCE IRL achieves good parameter recovery")
    else:
        print(f"   ✗ MCE IRL needs improvement (cos sim = {avg_mce_cos:.4f})")

    if avg_mm_cos < 0:
        print("   ✗ Max Margin IRL finds OPPOSITE direction (sign flip issue)")
    elif avg_mm_cos > 0.95:
        print("   ✓ Max Margin IRL recovers correct direction")
    else:
        print(f"   ~ Max Margin IRL partial recovery (cos sim = {avg_mm_cos:.4f})")

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
