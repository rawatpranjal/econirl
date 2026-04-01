#!/usr/bin/env python3
"""
Taxi-Gridworld MCE IRL Experiments
===================================

Tests MCE IRL and NFXP on a 10x10 gridworld with known ground truth.
Compares R(s) vs R(s,a) features under in-sample, out-of-sample, and transfer.

Key question: What does MCE IRL give us beyond NFXP?

Experiments:
    1. R(s,a) in-sample:  Both MCE IRL and NFXP should recover true params
    2. R(s,a) transfer:   Train on deterministic grid, test on stochastic grid
    3. R(s) in-sample:    State-only features — MCE IRL uses state visitation
    4. R(s) transfer:     Does state-only reward transfer to new dynamics?
"""

import time
import numpy as np
import jax.numpy as jnp

from econirl.environments.gridworld import GridworldEnvironment
from econirl.core.types import DDCProblem, Panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.preferences.linear import LinearUtility
from econirl.preferences.reward import LinearReward
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel


GRID_SIZE = 10
N_STATES = GRID_SIZE * GRID_SIZE  # 100
N_ACTIONS = 5
TRUE_PARAMS = {"step_penalty": -0.1, "terminal_reward": 10.0, "distance_weight": 0.1}


def make_env(stochastic=False, noise=0.0, seed=42):
    """Create gridworld environment, optionally with stochastic transitions."""
    env = GridworldEnvironment(
        grid_size=GRID_SIZE, discount_factor=0.99,
        step_penalty=TRUE_PARAMS["step_penalty"],
        terminal_reward=TRUE_PARAMS["terminal_reward"],
        distance_weight=TRUE_PARAMS["distance_weight"],
        seed=seed,
    )
    if stochastic and noise > 0:
        # Add noise to transitions: with prob `noise`, random action instead
        trans = jnp.array(env.transition_matrices)
        uniform = jnp.ones((N_ACTIONS, N_STATES, N_STATES)) / N_STATES
        trans = (1 - noise) * trans + noise * uniform
        # Re-normalize rows
        for a in range(N_ACTIONS):
            trans = trans.at[a].set(trans[a] / trans[a].sum(axis=1, keepdims=True))
        return env, trans
    return env, env.transition_matrices


def make_rsa_features(env):
    """R(s,a) features: the default 3 action-dependent features from GridworldEnvironment."""
    return LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )


def make_rs_features(env):
    """R(s) features: state-only, 2 features (step_penalty_indicator, distance)."""
    n_states = env.num_states
    features_np = np.zeros((n_states, 2))
    terminal = env.terminal_state
    for s in range(n_states):
        if s != terminal:
            features_np[s, 0] = 1.0  # step penalty indicator
            dist = abs(s // GRID_SIZE - (GRID_SIZE - 1)) + abs(s % GRID_SIZE - (GRID_SIZE - 1))
            features_np[s, 1] = -dist / (2.0 * GRID_SIZE)
    features = jnp.array(features_np)  # distance feature
    return LinearReward(
        state_features=features,
        parameter_names=["step_penalty", "distance_weight"],
        n_actions=N_ACTIONS,
    )


def compute_pct_optimal(policy, env, transitions):
    """Compute % of optimal value achieved by a policy."""
    problem = env.problem_spec
    operator = SoftBellmanOperator(problem, transitions)
    true_params_vec = jnp.array(list(TRUE_PARAMS.values()))
    true_reward = jnp.einsum("sak,k->sa", env.feature_matrix, true_params_vec)
    # Optimal value
    sol_opt = value_iteration(operator, true_reward, tol=1e-10, max_iter=5000)
    V_opt = sol_opt.V
    # Policy reward: r_pi(s) = Σ_a π(a|s) r(s,a)
    r_pi = (policy * true_reward).sum(axis=1)  # (S,)
    # Policy transitions: P_pi(s'|s) = Σ_a π(a|s) P(s'|s,a)
    P_pi = jnp.einsum("sa,ast->st", policy, transitions)
    I = jnp.eye(N_STATES)
    V_pi = jnp.linalg.solve(I - problem.discount_factor * P_pi, r_pi)
    # % optimal
    denom = V_opt.mean().item()
    if abs(denom) < 1e-8:
        return 100.0
    pct = (V_pi.mean() / V_opt.mean() * 100).item()
    return max(0, min(100, pct))


def run_estimator(name, est_class, panel, utility, problem, transitions, true_params_vec=None):
    """Run an estimator and return result dict."""
    t0 = time.time()
    try:
        est = est_class()
        result = est.estimate(panel=panel, utility=utility, problem=problem, transitions=transitions)
        elapsed = time.time() - t0

        params = result.parameters
        info = {
            "name": name,
            "time": elapsed,
            "converged": result.converged,
            "ll": result.log_likelihood,
            "params": {n: params[i].item() for i, n in enumerate(utility.parameter_names)} if params is not None else {},
        }
        if true_params_vec is not None and params is not None:
            rmse = float(jnp.sqrt(jnp.mean((params - true_params_vec) ** 2)))
            info["param_rmse"] = rmse

        if result.policy is not None:
            info["policy"] = result.policy

        return info
    except Exception as e:
        return {"name": name, "time": time.time() - t0, "error": str(e)}


def print_result(info, label=""):
    """Print a result dict."""
    prefix = f"  [{label}] " if label else "  "
    if "error" in info:
        print(f"{prefix}{info['name']}: FAILED — {info['error']}")
        return
    conv = "Yes" if info.get("converged") else "No"
    print(f"{prefix}{info['name']}: {info['time']:.1f}s, conv={conv}, LL={info.get('ll', 'N/A')}")
    for pname, pval in info.get("params", {}).items():
        print(f"    {pname}: {pval:.4f}")
    if "param_rmse" in info:
        print(f"    param RMSE: {info['param_rmse']:.4f}")


# ============================================================================
# Experiments
# ============================================================================

def experiment_rsa_insample():
    """Experiment 1: R(s,a) features, in-sample."""
    print("\n" + "=" * 60)
    print("Experiment 1: R(s,a) In-Sample")
    print("  Features: step_penalty, terminal_reward, distance_weight")
    print("  Both MCE IRL and NFXP should recover true parameters")
    print("=" * 60)

    env, transitions = make_env()
    panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)
    problem = env.problem_spec
    utility = make_rsa_features(env)
    true_vec = jnp.array(list(TRUE_PARAMS.values()))

    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.estimation.nfxp import NFXPEstimator

    # MCE IRL with L-BFGS-B
    config = MCEIRLConfig(compute_se=False, optimizer="L-BFGS-B",
                          outer_max_iter=500, inner_max_iter=1000, verbose=False)
    mce = type('', (), {"__call__": lambda self: MCEIRLEstimator(config=config)})()
    r_mce = run_estimator("MCE IRL", lambda: MCEIRLEstimator(config=config),
                          panel, utility, problem, transitions, true_vec)
    print_result(r_mce, "R(s,a) in-sample")

    # NFXP
    r_nfxp = run_estimator("NFXP-NK", NFXPEstimator, panel, utility, problem, transitions, true_vec)
    print_result(r_nfxp, "R(s,a) in-sample")

    return r_mce, r_nfxp


def experiment_rsa_transfer():
    """Experiment 2: R(s,a) features, transfer to stochastic dynamics."""
    print("\n" + "=" * 60)
    print("Experiment 2: R(s,a) Transfer")
    print("  Train on deterministic grid, test on stochastic (10% noise)")
    print("  Tests whether recovered reward generalizes to new dynamics")
    print("=" * 60)

    # Train on deterministic
    env_train, trans_train = make_env(stochastic=False)
    panel = simulate_panel(env_train, n_individuals=200, n_periods=100, seed=42)
    problem = env_train.problem_spec
    utility = make_rsa_features(env_train)
    true_vec = jnp.array(list(TRUE_PARAMS.values()))

    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.estimation.nfxp import NFXPEstimator

    config = MCEIRLConfig(compute_se=False, optimizer="L-BFGS-B",
                          outer_max_iter=500, inner_max_iter=1000, verbose=False)

    r_mce = run_estimator("MCE IRL", lambda: MCEIRLEstimator(config=config),
                          panel, utility, problem, trans_train, true_vec)
    print_result(r_mce, "train (deterministic)")

    r_nfxp = run_estimator("NFXP-NK", NFXPEstimator, panel, utility, problem, trans_train, true_vec)
    print_result(r_nfxp, "train (deterministic)")

    # Transfer: compute policy under stochastic dynamics using recovered reward
    _, trans_test = make_env(stochastic=True, noise=0.1)
    operator_test = SoftBellmanOperator(problem, trans_test)

    for name, info in [("MCE IRL", r_mce), ("NFXP-NK", r_nfxp)]:
        if "params" in info and info["params"]:
            params_vec = jnp.array([info["params"][n] for n in utility.parameter_names])
            reward_matrix = utility.compute(params_vec)
            sol = value_iteration(operator_test, reward_matrix, tol=1e-10, max_iter=5000)
            policy_test = sol.policy
            pct = compute_pct_optimal(policy_test, env_train, trans_test)
            print(f"  [{name} transfer] Policy on stochastic grid: {pct:.1f}% optimal")


def experiment_rs_insample():
    """Experiment 3: R(s) state-only features, in-sample."""
    print("\n" + "=" * 60)
    print("Experiment 3: R(s) In-Sample (State-Only Features)")
    print("  Features: step_penalty, distance_weight (NO terminal_reward)")
    print("  MCE IRL uses state visitation; tests R(s) identification")
    print("=" * 60)

    env, transitions = make_env()
    panel = simulate_panel(env, n_individuals=200, n_periods=100, seed=42)
    problem = env.problem_spec
    utility_rs = make_rs_features(env)

    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
    from econirl.estimation.maxent_irl import MaxEntIRLEstimator

    # MCE IRL
    config = MCEIRLConfig(compute_se=False, optimizer="L-BFGS-B",
                          outer_max_iter=500, inner_max_iter=1000, verbose=False)
    r_mce = run_estimator("MCE IRL", lambda: MCEIRLEstimator(config=config),
                          panel, utility_rs, problem, transitions)
    print_result(r_mce, "R(s) in-sample")

    # MaxEnt IRL (also supports state-only features)
    r_maxent = run_estimator("MaxEnt IRL", lambda: MaxEntIRLEstimator(
        optimizer="L-BFGS-B", inner_max_iter=1000, outer_max_iter=500, verbose=False),
        panel, utility_rs, problem, transitions)
    print_result(r_maxent, "R(s) in-sample")

    # Note: NFXP requires action-dependent features for logit choice probs
    print("  [NFXP] Cannot estimate R(s)-only — requires R(s,a) for P(a|s) identification")


def experiment_rs_transfer():
    """Experiment 4: R(s) state-only features, transfer."""
    print("\n" + "=" * 60)
    print("Experiment 4: R(s) Transfer")
    print("  Train R(s) on deterministic, test on stochastic")
    print("=" * 60)

    env_train, trans_train = make_env(stochastic=False)
    panel = simulate_panel(env_train, n_individuals=200, n_periods=100, seed=42)
    problem = env_train.problem_spec
    utility_rs = make_rs_features(env_train)

    from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig

    config = MCEIRLConfig(compute_se=False, optimizer="L-BFGS-B",
                          outer_max_iter=500, inner_max_iter=1000, verbose=False)
    r_mce = run_estimator("MCE IRL", lambda: MCEIRLEstimator(config=config),
                          panel, utility_rs, problem, trans_train)
    print_result(r_mce, "train R(s)")

    # Transfer: apply R(s) to stochastic grid
    _, trans_test = make_env(stochastic=True, noise=0.1)
    if "params" in r_mce and r_mce["params"]:
        params_vec = jnp.array([r_mce["params"][n] for n in utility_rs.parameter_names])
        # R(s) reward: expand to (S, A) for policy computation
        reward_sa = utility_rs.compute(params_vec)  # Should be (S, A)
        operator_test = SoftBellmanOperator(problem, trans_test)
        sol_rs = value_iteration(operator_test, reward_sa, tol=1e-10, max_iter=5000)
        policy_test = sol_rs.policy
        pct = compute_pct_optimal(policy_test, env_train, trans_test)
        print(f"  [MCE IRL R(s) transfer] Policy on stochastic grid: {pct:.1f}% optimal")


if __name__ == "__main__":
    print("=" * 60)
    print("Taxi-Gridworld: MCE IRL vs NFXP")
    print(f"Grid: {GRID_SIZE}x{GRID_SIZE} = {N_STATES} states, {N_ACTIONS} actions")
    print(f"True params: {TRUE_PARAMS}")
    print("=" * 60)

    experiment_rsa_insample()
    experiment_rsa_transfer()
    experiment_rs_insample()
    experiment_rs_transfer()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
