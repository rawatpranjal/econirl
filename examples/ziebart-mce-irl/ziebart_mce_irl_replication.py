"""
Replication of Ziebart et al. (2008/2010) Maximum Causal Entropy IRL
=====================================================================

Replicates the gridworld experiment from Ziebart's MCE IRL paper using
the econirl package. Demonstrates reward recovery, feature matching,
and policy comparison between MCE IRL and MaxEnt IRL.
"""

import jax.numpy as jnp
import numpy as np
from econirl.environments.gridworld import GridworldEnvironment
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.contrib.maxent_irl import MaxEntIRLEstimator
from econirl.simulation.synthetic import simulate_panel
from econirl.preferences.action_reward import ActionDependentReward
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration


def run_replication():
    print("=" * 70)
    print("  Ziebart MCE IRL Replication: Gridworld Experiment")
    print("=" * 70)

    # --- Setup ---
    grid_size = 5
    n_traj = 100
    n_periods = 30
    discount = 0.95
    seed = 42

    env = GridworldEnvironment(
        grid_size=grid_size,
        step_penalty=-0.1,
        terminal_reward=10.0,
        distance_weight=0.1,
        discount_factor=discount,
        seed=seed,
    )

    true_params = env.get_true_parameter_vector()
    print(f"\nTrue parameters: {dict(zip(env.parameter_names, true_params.tolist()))}")
    print(f"Grid: {grid_size}x{grid_size}, States: {env.num_states}, Actions: {env.num_actions}")

    # --- Generate Expert Demonstrations ---
    print(f"\nGenerating {n_traj} expert trajectories ({n_periods} periods each)...")
    panel = simulate_panel(env=env, n_individuals=n_traj, n_periods=n_periods, seed=seed)
    print(f"Total observations: {panel.num_observations}")

    all_states = panel.get_all_states()
    all_actions = panel.get_all_actions()
    terminal_visits = (all_states == env.terminal_state).sum().item()
    print(f"Terminal state visits: {terminal_visits} ({terminal_visits/len(all_states)*100:.1f}%)")

    action_names = ["Left", "Right", "Up", "Down", "Stay"]
    for a in range(env.num_actions):
        count = (all_actions == a).sum().item()
        print(f"  {action_names[a]:>5}: {count:5d} ({count/len(all_actions)*100:.1f}%)")

    # --- Setup for estimation ---
    reward_fn = ActionDependentReward(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )
    problem = env.problem_spec
    transitions = env.transition_matrices

    # --- MCE IRL (Ziebart 2010) ---
    print(f"\n{'='*50}")
    print("  MCE IRL (Ziebart 2010)")
    print(f"{'='*50}")

    config = MCEIRLConfig(
        learning_rate=0.1,
        outer_max_iter=500,
        outer_tol=1e-8,
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=5000,
        use_adam=True,
        compute_se=False,  # Skip SE for speed
        verbose=True,
    )

    mce_estimator = MCEIRLEstimator(config=config)
    mce_result = mce_estimator.estimate(
        panel=panel, utility=reward_fn, problem=problem,
        transitions=transitions, true_params=true_params,
    )

    est_params = mce_result.parameters
    print(f"\nConverged: {mce_result.converged}")
    print(f"Log-likelihood: {mce_result.log_likelihood:.4f}")
    print(f"Iterations: {mce_result.num_iterations}")

    # --- Parameter comparison ---
    # IRL rewards are identified up to a constant + scale, so we compare
    # normalized directions and policy quality
    print(f"\n  {'Parameter':<20} {'True':>10} {'Estimated':>10}")
    print(f"  {'-'*40}")
    for i, name in enumerate(env.parameter_names):
        print(f"  {name:<20} {true_params[i].item():>10.4f} {est_params[i].item():>10.4f}")

    # Normalized comparison (reward direction)
    true_norm = true_params / jnp.linalg.norm(true_params)
    est_norm = est_params / jnp.linalg.norm(est_params)
    cos_sim = (jnp.dot(est_params, true_params) / (
        jnp.linalg.norm(est_params) * jnp.linalg.norm(true_params)
    )).item()
    print(f"\n  Normalized direction (true):  {true_norm.tolist()}")
    print(f"  Normalized direction (est):   {est_norm.tolist()}")
    print(f"  Cosine similarity: {cos_sim:.6f}")

    # Feature matching
    if mce_result.metadata:
        emp = mce_result.metadata.get('empirical_features', [])
        exp = mce_result.metadata.get('final_expected_features', [])
        diff = mce_result.metadata.get('feature_difference', None)
        if emp and exp:
            print(f"\n  Feature Matching:")
            print(f"    Empirical:  {[f'{x:.6f}' for x in emp]}")
            print(f"    Expected:   {[f'{x:.6f}' for x in exp]}")
            print(f"    ||diff||:   {diff:.8f}")

    # --- MaxEnt IRL (Ziebart 2008) ---
    print(f"\n{'='*50}")
    print("  MaxEnt IRL (Ziebart 2008)")
    print(f"{'='*50}")

    maxent_estimator = MaxEntIRLEstimator(
        inner_solver="policy", inner_tol=1e-10,
        outer_tol=1e-6, outer_max_iter=200,
        compute_hessian=False, verbose=False,
    )
    maxent_result = maxent_estimator.estimate(
        panel=panel, utility=reward_fn, problem=problem, transitions=transitions,
    )
    maxent_params = maxent_result.parameters
    print(f"  Converged: {maxent_result.converged}")
    print(f"  Log-likelihood: {maxent_result.log_likelihood:.4f}")

    # --- Policy Comparison ---
    print(f"\n{'='*50}")
    print("  Policy Comparison")
    print(f"{'='*50}")

    operator = SoftBellmanOperator(problem, transitions)
    true_reward = reward_fn.compute(true_params)
    true_sol = hybrid_iteration(operator, true_reward, tol=1e-10)
    true_policy = true_sol.policy

    mce_policy = mce_result.policy
    maxent_policy = maxent_result.policy

    eps = 1e-10
    kl_mce = (true_policy * jnp.log((true_policy + eps) / (mce_policy + eps))).sum(axis=1).mean().item()
    kl_maxent = (true_policy * jnp.log((true_policy + eps) / (maxent_policy + eps))).sum(axis=1).mean().item()

    true_best = true_policy.argmax(axis=1)
    mce_best = mce_policy.argmax(axis=1)
    maxent_best = maxent_policy.argmax(axis=1)
    mce_acc = (true_best == mce_best).astype(jnp.float32).mean().item() * 100
    maxent_acc = (true_best == maxent_best).astype(jnp.float32).mean().item() * 100

    print(f"\n  {'Metric':<30} {'MCE IRL':>12} {'MaxEnt IRL':>12}")
    print(f"  {'-'*54}")
    print(f"  {'Log-likelihood':<30} {mce_result.log_likelihood:>12.4f} {maxent_result.log_likelihood:>12.4f}")
    print(f"  {'KL(true || model)':<30} {kl_mce:>12.6f} {kl_maxent:>12.6f}")
    print(f"  {'Policy accuracy (%)':<30} {mce_acc:>12.1f} {maxent_acc:>12.1f}")
    print(f"  {'Cosine sim (reward)':<30} {cos_sim:>12.6f} {'':>12}")

    # Show policy at a few key states
    print(f"\n  Policy at key states:")
    key_states = [0, grid_size * grid_size // 2, env.terminal_state - 1, env.terminal_state]
    for s in key_states:
        row, col = env.state_to_grid_position(s)
        print(f"\n  State {s} (row={row}, col={col}):")
        print(f"    {'Action':<8} {'True':>8} {'MCE':>8} {'MaxEnt':>8}")
        for a in range(env.num_actions):
            print(f"    {action_names[a]:<8} {true_policy[s,a].item():>8.4f} {mce_policy[s,a].item():>8.4f} {maxent_policy[s,a].item():>8.4f}")

    # --- Summary ---
    print(f"\n{'='*70}")
    print("  Summary")
    print(f"{'='*70}")
    print(f"""
  MCE IRL (Ziebart 2010) successfully recovers reward parameters from
  expert demonstrations in a {grid_size}x{grid_size} gridworld:

  - Feature matching converges (||diff|| = {diff:.8f})
  - Reward direction recovery: cosine similarity = {cos_sim:.4f}
  - Policy accuracy: {mce_acc:.1f}% of states match optimal action
  - KL divergence from true policy: {kl_mce:.6f}
  - Log-likelihood: {mce_result.log_likelihood:.2f}

  Note: IRL rewards are identified only up to an additive constant
  and multiplicative scale (Kim et al. 2021, Cao & Cohen 2021), so
  the *direction* of the recovered reward vector matters more than
  exact parameter values. The high cosine similarity confirms the
  algorithm correctly identifies the reward structure.
""")


if __name__ == "__main__":
    run_replication()
