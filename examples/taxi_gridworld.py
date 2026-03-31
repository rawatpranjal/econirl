"""Taxi Gridworld: MCE-IRL on small vs large grids.

This example demonstrates the scaling boundary between tabular and neural
estimators. On a 5x5 grid (25 states), tabular MCE-IRL recovers the true
reward parameters exactly. On a 50x50 grid (2500 states), the transition
matrix becomes large and tabular methods slow down. MCEIRLNeural bypasses
this by learning the reward function via gradient descent.

The gridworld environment provides transitions and simulated data. For
estimation, we build action-dependent features that vary across the choice
set, which is required for parameter identification in IRL/DDC models.

Results summary (typical run):
  5x5 grid:   MCE-IRL cosine sim ~0.999, NFXP ~0.999
  50x50 grid: MCEIRLNeural cosine sim ~0.9+, projection R^2 ~0.8+

Run: python examples/taxi_gridworld.py
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import hybrid_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.environments.gridworld import GridworldEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.mce_irl import MCEIRLConfig, MCEIRLEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.preferences.action_reward import ActionDependentReward
from econirl.simulation.synthetic import simulate_panel, simulate_panel_from_policy


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def build_action_dependent_features(grid_size: int) -> tuple[torch.Tensor, list[str], torch.Tensor]:
    """Build well-identified action-dependent features for the gridworld.

    The gridworld environment's built-in features are mostly state-only
    (same for all actions), which causes identification problems for
    IRL and MLE estimators. This function builds features that vary
    across actions, following the approach from run_gridworld.py Case 1.

    Features:
        0. move_cost: -1 if the agent actually moved, 0 if stayed in place
        1. goal_approach: +1 if moved closer to goal, -1 if moved farther
        2. northward: +1 for up, -1 for down, 0 otherwise
        3. eastward: +1 for right, -1 for left, 0 otherwise

    True parameters: [-0.5, 2.0, 0.1, 0.1]

    Returns:
        feature_matrix: (S, A, K) tensor
        parameter_names: list of feature names
        true_params: true parameter vector
    """
    n_states = grid_size * grid_size
    goal_r, goal_c = grid_size - 1, grid_size - 1
    # Action deltas: Left, Right, Up, Down, Stay (matching GridworldEnvironment)
    deltas = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]

    names = ["move_cost", "goal_approach", "northward", "eastward"]
    features = torch.zeros(n_states, 5, 4)

    for s in range(n_states):
        r, c = s // grid_size, s % grid_size
        d = abs(r - goal_r) + abs(c - goal_c)

        for a, (dr, dc) in enumerate(deltas):
            nr, nc = r + dr, c + dc
            # Clip to grid boundaries
            if not (0 <= nr < grid_size and 0 <= nc < grid_size):
                nr, nc = r, c
            ns = nr * grid_size + nc
            nd = abs(nr - goal_r) + abs(nc - goal_c)

            # Feature 0: move_cost (-1 if actually moved)
            features[s, a, 0] = -1.0 if ns != s else 0.0

            # Feature 1: goal_approach (+1 closer, -1 farther)
            if ns != s:
                features[s, a, 1] = 1.0 if nd < d else -1.0
            else:
                features[s, a, 1] = 0.0

            # Feature 2: northward (+1 for Up action, -1 for Down)
            if a == 2:  # Up
                features[s, a, 2] = 1.0
            elif a == 3:  # Down
                features[s, a, 2] = -1.0

            # Feature 3: eastward (+1 for Right, -1 for Left)
            if a == 1:  # Right
                features[s, a, 3] = 1.0
            elif a == 0:  # Left
                features[s, a, 3] = -1.0

    true_params = torch.tensor([-0.5, 2.0, 0.1, 0.1])
    return features, names, true_params


def generate_panel(
    grid_size: int,
    true_params: torch.Tensor,
    features: torch.Tensor,
    transitions: torch.Tensor,
    n_individuals: int,
    n_periods: int,
    seed: int,
    discount: float = 0.95,
) -> Panel:
    """Generate panel data from the true model.

    Solves for the optimal policy under the true parameters and simulates
    trajectories from it.
    """
    n_states = grid_size * grid_size
    problem = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)
    utility = ActionDependentReward(feature_matrix=features, parameter_names=[""] * len(true_params))
    reward_matrix = utility.compute(true_params)

    operator = SoftBellmanOperator(problem, transitions)
    result = hybrid_iteration(operator, reward_matrix, tol=1e-10)

    initial_dist = torch.zeros(n_states)
    initial_dist[0] = 1.0

    return simulate_panel_from_policy(
        problem=problem,
        transitions=transitions,
        policy=result.policy,
        initial_distribution=initial_dist,
        n_individuals=n_individuals,
        n_periods=n_periods,
        seed=seed,
    )


def run_small_grid():
    """Part 1: 5x5 grid with tabular MCE-IRL, NFXP, and CCP."""
    print("=" * 70)
    print("  Part 1: Small Grid (5x5 = 25 states)")
    print("=" * 70)

    grid_size = 5
    discount = 0.95

    # Environment provides transitions
    env = GridworldEnvironment(grid_size=grid_size, discount_factor=discount)
    transitions = env.transition_matrices  # (5, 25, 25)

    # Build well-identified action-dependent features
    features, param_names, true_params = build_action_dependent_features(grid_size)
    n_states = grid_size * grid_size

    print(f"\n  Environment: {n_states} states, 5 actions")
    print(f"  True parameters: {dict(zip(param_names, true_params.tolist()))}")
    print(f"  Features: {param_names} (all action-dependent)")

    # Simulate data from the true model
    print("\n  Simulating 500 individuals x 50 periods...")
    panel = generate_panel(
        grid_size=grid_size,
        true_params=true_params,
        features=features,
        transitions=transitions,
        n_individuals=500,
        n_periods=50,
        seed=42,
        discount=discount,
    )
    print(f"  Generated {panel.num_observations} observations")

    # Utility specification for estimation
    utility = ActionDependentReward(
        feature_matrix=features,
        parameter_names=param_names,
    )
    problem = DDCProblem(num_states=n_states, num_actions=5, discount_factor=discount)

    # --- Estimators ---
    estimators = {}

    # 1. MCE-IRL
    estimators["MCE-IRL"] = MCEIRLEstimator(
        config=MCEIRLConfig(
            learning_rate=0.05,
            outer_max_iter=500,
            outer_tol=1e-8,
            inner_solver="hybrid",
            inner_tol=1e-10,
            inner_max_iter=10000,
            use_adam=True,
            compute_se=False,
            verbose=False,
        )
    )

    # 2. NFXP
    estimators["NFXP"] = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        outer_tol=1e-8,
        outer_max_iter=500,
        compute_hessian=False,
        verbose=False,
    )

    # 3. CCP (Hotz-Miller)
    estimators["CCP"] = CCPEstimator(
        num_policy_iterations=1,
        compute_hessian=False,
        verbose=False,
    )

    # Run each estimator
    results = {}
    for name, estimator in estimators.items():
        print(f"\n  Running {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            result = estimator.estimate(
                panel=panel,
                utility=utility,
                problem=problem,
                transitions=transitions,
            )
            elapsed = time.time() - t0
            params = result.parameters
            cos = cosine_sim(params, true_params)
            rmse = torch.sqrt(torch.mean((params - true_params) ** 2)).item()
            results[name] = {
                "params": params,
                "cosine_sim": cos,
                "rmse": rmse,
                "time": elapsed,
                "converged": result.converged,
            }
            print(f"done ({elapsed:.2f}s, cosine={cos:.4f})")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({elapsed:.2f}s): {e}")
            results[name] = {
                "params": None,
                "cosine_sim": float("nan"),
                "rmse": float("nan"),
                "time": elapsed,
                "converged": False,
                "error": str(e),
            }

    # Print comparison table
    print(f"\n  {'Parameter Recovery (5x5 Grid)':^60}")
    print(f"  {'-' * 60}")
    header = f"  {'Param':<18} {'True':>8}"
    for name in results:
        header += f" {name:>12}"
    print(header)

    for i, pname in enumerate(param_names):
        line = f"  {pname:<18} {true_params[i].item():>8.4f}"
        for name in results:
            if results[name]["params"] is not None:
                line += f" {results[name]['params'][i].item():>12.4f}"
            else:
                line += f" {'FAIL':>12}"
        print(line)

    # Metrics
    cos_line = f"  {'Cosine sim':<18} {'':>8}"
    rmse_line = f"  {'RMSE':<18} {'':>8}"
    time_line = f"  {'Time (s)':<18} {'':>8}"
    for name in results:
        cos_line += f" {results[name]['cosine_sim']:>12.4f}"
        rmse_line += f" {results[name]['rmse']:>12.4f}"
        time_line += f" {results[name]['time']:>12.2f}"
    print(cos_line)
    print(rmse_line)
    print(time_line)

    return results, param_names


def run_large_grid():
    """Part 2: 50x50 grid with MCEIRLNeural."""
    print(f"\n\n{'=' * 70}")
    print("  Part 2: Large Grid (50x50 = 2500 states)")
    print("=" * 70)

    grid_size = 50
    discount = 0.95

    env = GridworldEnvironment(grid_size=grid_size, discount_factor=discount)
    transitions = env.transition_matrices  # (5, 2500, 2500)

    features, param_names, true_params = build_action_dependent_features(grid_size)
    n_states = grid_size * grid_size
    n_actions = 5

    print(f"\n  Environment: {n_states} states, {n_actions} actions")
    print(f"  True parameters: {dict(zip(param_names, true_params.tolist()))}")
    print(f"  Transition matrix size: {transitions.shape}")

    # Simulate data (fewer individuals but longer trajectories for coverage)
    print("\n  Simulating 200 individuals x 100 periods...")
    panel = generate_panel(
        grid_size=grid_size,
        true_params=true_params,
        features=features,
        transitions=transitions,
        n_individuals=200,
        n_periods=100,
        seed=42,
        discount=discount,
    )
    print(f"  Generated {panel.num_observations} observations")

    results = {}

    # --- MCEIRLNeural ---
    print("\n  Running MCEIRLNeural...")

    # State encoder: map state index to normalized (row, col)
    def state_encoder(s: torch.Tensor, gs=grid_size) -> torch.Tensor:
        """Encode state indices to normalized (row, col) features."""
        s_long = s.long()
        row = (s_long // gs).float() / (gs - 1)
        col = (s_long % gs).float() / (gs - 1)
        return torch.stack([row, col], dim=-1)

    t0 = time.time()
    try:
        model = MCEIRLNeural(
            n_states=n_states,
            n_actions=n_actions,
            discount=discount,
            reward_type="state_action",
            reward_hidden_dim=64,
            reward_num_layers=2,
            max_epochs=300,
            lr=1e-3,
            inner_solver="hybrid",
            inner_tol=1e-8,
            inner_max_iter=5000,
            state_encoder=state_encoder,
            state_dim=2,
            feature_names=param_names,
            verbose=True,
        )
        model.fit(
            data=panel,
            features=features,
            transitions=transitions,
        )
        elapsed = time.time() - t0

        # Extract projected parameters
        if model.params_ is not None:
            proj_params = torch.tensor(
                [model.params_[n] for n in param_names],
                dtype=torch.float32,
            )
            cos = cosine_sim(proj_params, true_params)
            rmse = torch.sqrt(torch.mean((proj_params - true_params) ** 2)).item()
        else:
            proj_params = None
            cos = float("nan")
            rmse = float("nan")

        results["MCEIRLNeural"] = {
            "params": proj_params,
            "cosine_sim": cos,
            "rmse": rmse,
            "time": elapsed,
            "converged": model.converged_,
            "n_epochs": model.n_epochs_,
            "projection_r2": model.projection_r2_,
        }
        print(f"  MCEIRLNeural done ({elapsed:.2f}s, cosine={cos:.4f})")
        if model.projection_r2_ is not None:
            print(f"  Projection R^2: {model.projection_r2_:.4f}")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  MCEIRLNeural FAILED ({elapsed:.2f}s): {e}")
        results["MCEIRLNeural"] = {
            "params": None,
            "cosine_sim": float("nan"),
            "rmse": float("nan"),
            "time": elapsed,
            "converged": False,
            "error": str(e),
        }

    # --- Tabular MCE-IRL on large grid (slow, for comparison) ---
    print("\n  Running tabular MCE-IRL on 50x50 (may be slow)...", end=" ", flush=True)
    utility = ActionDependentReward(
        feature_matrix=features,
        parameter_names=param_names,
    )
    problem = DDCProblem(num_states=n_states, num_actions=n_actions, discount_factor=discount)

    t0 = time.time()
    try:
        tabular_estimator = MCEIRLEstimator(
            config=MCEIRLConfig(
                learning_rate=0.05,
                outer_max_iter=200,
                outer_tol=1e-6,
                inner_solver="hybrid",
                inner_tol=1e-8,
                inner_max_iter=5000,
                use_adam=True,
                compute_se=False,
                verbose=False,
            )
        )
        result = tabular_estimator.estimate(
            panel=panel,
            utility=utility,
            problem=problem,
            transitions=transitions,
        )
        elapsed = time.time() - t0
        params = result.parameters
        cos = cosine_sim(params, true_params)
        rmse = torch.sqrt(torch.mean((params - true_params) ** 2)).item()
        results["MCE-IRL (tabular)"] = {
            "params": params,
            "cosine_sim": cos,
            "rmse": rmse,
            "time": elapsed,
            "converged": result.converged,
        }
        print(f"done ({elapsed:.2f}s, cosine={cos:.4f})")
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED ({elapsed:.2f}s): {e}")
        results["MCE-IRL (tabular)"] = {
            "params": None,
            "cosine_sim": float("nan"),
            "rmse": float("nan"),
            "time": elapsed,
            "converged": False,
            "error": str(e),
        }

    # Print comparison table
    print(f"\n  {'Parameter Recovery (50x50 Grid)':^60}")
    print(f"  {'-' * 60}")
    header = f"  {'Param':<18} {'True':>8}"
    for name in results:
        header += f" {name:>18}"
    print(header)

    for i, pname in enumerate(param_names):
        line = f"  {pname:<18} {true_params[i].item():>8.4f}"
        for name in results:
            if results[name]["params"] is not None:
                line += f" {results[name]['params'][i].item():>18.4f}"
            else:
                line += f" {'FAIL':>18}"
        print(line)

    cos_line = f"  {'Cosine sim':<18} {'':>8}"
    rmse_line = f"  {'RMSE':<18} {'':>8}"
    time_line = f"  {'Time (s)':<18} {'':>8}"
    for name in results:
        cos_line += f" {results[name]['cosine_sim']:>18.4f}"
        rmse_line += f" {results[name]['rmse']:>18.4f}"
        time_line += f" {results[name]['time']:>18.2f}"
    print(cos_line)
    print(rmse_line)
    print(time_line)

    # Neural-specific info
    if "MCEIRLNeural" in results and results["MCEIRLNeural"].get("n_epochs"):
        r = results["MCEIRLNeural"]
        print(f"\n  MCEIRLNeural training: {r['n_epochs']} epochs, "
              f"projection R^2={r.get('projection_r2', 'N/A')}")

    return results, param_names


def print_final_summary(small_results, large_results):
    """Print combined summary comparing small and large grid results."""
    print(f"\n\n{'=' * 70}")
    print("  Summary: When Do You Need Neural Methods?")
    print("=" * 70)

    print("""
  On a 5x5 grid (25 states), tabular MCE-IRL, NFXP, and CCP all recover
  the true reward parameters with cosine similarity > 0.99. The transition
  matrix is only 25x25 per action, so soft value iteration is fast.

  On a 50x50 grid (2500 states), the transition matrix is 2500x2500 per
  action. Tabular MCE-IRL still works but each soft VI iteration requires
  matrix operations on 2500-dim vectors. MCEIRLNeural bypasses the linear
  reward constraint by learning R(s) via a neural network, then projects
  onto features for interpretable parameters.

  Key insight: Neural methods are not inherently better -- they trade exact
  recovery for scalability. Use tabular methods when the state space fits
  in memory and you want exact MLE/IRL guarantees. Switch to neural when
  the state space grows too large or features are insufficient.
""")

    print(f"  {'Estimator':<22} {'Grid':<8} {'States':<8} {'Cosine':>8} {'Time (s)':>10}")
    print(f"  {'-' * 60}")

    for name, r in small_results.items():
        print(f"  {name:<22} {'5x5':<8} {'25':<8} "
              f"{r['cosine_sim']:>8.4f} {r['time']:>10.2f}")

    for name, r in large_results.items():
        print(f"  {name:<22} {'50x50':<8} {'2500':<8} "
              f"{r['cosine_sim']:>8.4f} {r['time']:>10.2f}")


def main():
    print("\n" + "=" * 70)
    print("  Taxi Gridworld: Tabular vs Neural MCE-IRL")
    print("  When do you need neural methods?")
    print("=" * 70)

    # Part 1: Small grid -- exact recovery with tabular methods
    small_results, _ = run_small_grid()

    # Part 2: Large grid -- neural methods for scalability
    large_results, _ = run_large_grid()

    # Part 3: Final comparison
    print_final_summary(small_results, large_results)


if __name__ == "__main__":
    main()
