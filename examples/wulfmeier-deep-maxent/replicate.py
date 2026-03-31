#!/usr/bin/env python3
"""Wulfmeier (2016) Deep MaxEnt IRL replication on Objectworld and Binaryworld.

This script benchmarks MCEIRLNeural (deep IRL with a 2-hidden-layer MLP reward
network) against linear MCE-IRL on two environments from Wulfmeier et al. (2016).
Both environments have reward functions that depend nonlinearly on features, so a
neural reward model should outperform a linear one.

The evaluation metric is Expected Value Difference (EVD), which measures the gap
between the true optimal value and the value of the learned policy evaluated
under the true reward. Lower EVD means the learned policy is closer to optimal.

For each environment, the script sweeps over demonstration counts and random seeds,
fitting both estimators and recording EVD. Results are saved as JSON and printed as
a table.

Usage:
    python replicate.py
    python replicate.py --grid-size 16 --demo-counts 8 16 32 --n-seeds 3

References:
    Wulfmeier, M., Ondruska, P., & Posner, I. (2016). Maximum entropy deep
        inverse reinforcement learning. arXiv:1507.04888.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from econirl.environments.objectworld import ObjectworldEnvironment
from econirl.environments.binaryworld import BinaryworldEnvironment
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration
from econirl.core.types import DDCProblem, Panel
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.estimators.mceirl_neural import MCEIRLNeural
from econirl.preferences.linear import LinearUtility

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

GRID_SIZE = 32
DEMO_COUNTS = [8, 16, 32, 64, 128]
N_SEEDS = 5
MAX_STEPS = 50
NOISE_FRACTION = 0.3
DISCOUNT = 0.9


# -----------------------------------------------------------------------
# Expected Value Difference
# -----------------------------------------------------------------------


def compute_evd(
    true_reward: torch.Tensor,
    transitions: torch.Tensor,
    learned_policy: np.ndarray,
    discount: float,
) -> float:
    """Compute Expected Value Difference between optimal and learned policies.

    EVD = mean_s [V*(s; r_true) - V^pi_learned(s; r_true)]

    where V* is the value under the optimal policy for the true reward and
    V^pi_learned is the value of the learned policy evaluated under the true
    reward. Lower is better.

    Args:
        true_reward: Ground-truth reward vector of shape (n_states,).
        transitions: Transition matrices of shape (n_actions, n_states, n_states).
        learned_policy: Learned policy of shape (n_states, n_actions) as numpy.
        discount: Discount factor.

    Returns:
        Scalar EVD value averaged across states.
    """
    n_states = true_reward.shape[0]
    n_actions = transitions.shape[0]

    # Build reward matrix (S, A) from state-only reward
    reward_matrix = true_reward.unsqueeze(1).expand(n_states, n_actions).clone()

    # Compute V* via soft policy iteration under true reward
    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=discount,
        scale_parameter=1.0,
    )
    operator = SoftBellmanOperator(problem, transitions)
    result = policy_iteration(operator, reward_matrix)
    V_star = result.V.numpy()

    # Compute V^pi_learned under true reward via matrix inversion.
    # R_pi(s) = sum_a pi(a|s) * r(s,a)
    # P_pi(s, s') = sum_a pi(a|s) * P(s'|s,a)
    # V_pi = (I - gamma * P_pi)^{-1} R_pi
    policy_np = np.asarray(learned_policy, dtype=np.float64)
    r_true_np = true_reward.numpy().astype(np.float64)

    # State-only reward: R_pi(s) = r(s) for all policies
    R_pi = r_true_np.copy()

    # P_pi[s, s'] = sum_a pi(a|s) * P(s'|s, a)
    trans_np = transitions.numpy().astype(np.float64)
    P_pi = np.einsum("sa,ast->st", policy_np, trans_np)

    I = np.eye(n_states, dtype=np.float64)
    V_learned = np.linalg.solve(I - discount * P_pi, R_pi)

    evd = float(np.mean(V_star - V_learned))
    return evd


# -----------------------------------------------------------------------
# Panel to DataFrame conversion
# -----------------------------------------------------------------------


def panel_to_dataframe(panel: Panel) -> pd.DataFrame:
    """Convert a Panel of trajectories to a DataFrame for MCEIRLNeural."""
    rows = []
    for traj in panel.trajectories:
        for t in range(len(traj.states)):
            rows.append({
                "agent_id": traj.individual_id,
                "state": traj.states[t].item(),
                "action": traj.actions[t].item(),
            })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------
# MCE-IRL Neural (Deep IRL)
# -----------------------------------------------------------------------


def run_estimator_mce_neural(
    env,
    panel: Panel,
    grid_size: int,
    discount: float,
    max_epochs: int = 200,
    lr: float = 0.01,
) -> np.ndarray:
    """Fit MCEIRLNeural and return the learned policy.

    The neural reward network uses a 2-hidden-layer MLP with 64 units and
    ReLU activations, matching Wulfmeier (2016) Section IV.A.

    Args:
        env: An Objectworld or Binaryworld environment instance.
        panel: Panel of demonstration trajectories.
        grid_size: Side length of the grid.
        discount: Discount factor.
        max_epochs: Maximum training epochs.
        lr: Learning rate.

    Returns:
        Learned policy of shape (n_states, n_actions) as numpy array.
    """
    n_states = env.num_states
    n_actions = env.num_actions

    df = panel_to_dataframe(panel)

    model = MCEIRLNeural(
        n_states=n_states,
        n_actions=n_actions,
        discount=discount,
        reward_type="state",
        reward_hidden_dim=64,
        reward_num_layers=2,
        max_epochs=max_epochs,
        lr=lr,
        inner_solver="hybrid",
        inner_tol=1e-8,
        inner_max_iter=5000,
        state_encoder=env.encode_states,
        state_dim=env.state_dim,
        verbose=False,
    )

    model.fit(
        data=df,
        state="state",
        action="action",
        id="agent_id",
        transitions=env.transition_matrices.numpy(),
    )

    return model.policy_


# -----------------------------------------------------------------------
# Linear MCE-IRL
# -----------------------------------------------------------------------


def run_estimator_mce_linear(
    env,
    panel: Panel,
    discount: float,
) -> np.ndarray:
    """Fit linear MCE-IRL and return the learned policy.

    Uses L-BFGS-B optimization with the hybrid inner solver for soft
    value iteration, following the standard MCE-IRL algorithm from
    Ziebart (2010).

    Args:
        env: An Objectworld or Binaryworld environment instance.
        panel: Panel of demonstration trajectories.
        discount: Discount factor.

    Returns:
        Learned policy of shape (n_states, n_actions) as numpy array.
    """
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    estimator = MCEIRLEstimator(config=MCEIRLConfig(
        optimizer="L-BFGS-B",
        inner_solver="hybrid",
        inner_max_iter=5000,
        inner_tol=1e-8,
        outer_max_iter=500,
        outer_tol=1e-6,
        compute_se=False,
        verbose=False,
    ))

    result = estimator.estimate(
        panel,
        utility,
        env.problem_spec,
        env.transition_matrices,
    )

    policy = result.policy
    if isinstance(policy, torch.Tensor):
        return policy.numpy()
    return np.asarray(policy)


# -----------------------------------------------------------------------
# Benchmark runner
# -----------------------------------------------------------------------


def run_benchmark(
    env_name: str,
    grid_size: int,
    demo_counts: list[int],
    n_seeds: int,
    max_steps: int,
    noise_fraction: float,
    discount: float,
    max_epochs: int = 200,
    lr: float = 0.01,
) -> dict:
    """Run the full benchmark for one environment type.

    For each combination of demonstration count and random seed, generates an
    environment, samples noisy expert demonstrations, fits both estimators,
    and records EVD.

    Args:
        env_name: "objectworld" or "binaryworld".
        grid_size: Side length of the grid.
        demo_counts: List of demonstration counts to test.
        n_seeds: Number of random seeds per demo count.
        max_steps: Trajectory length for demos.
        noise_fraction: Fraction of random actions in demos.
        discount: Discount factor.
        max_epochs: Maximum epochs for MCEIRLNeural.
        lr: Learning rate for MCEIRLNeural.

    Returns:
        Dictionary with results for each demo count.
    """
    results = {}

    for n_demos in demo_counts:
        neural_evds = []
        linear_evds = []

        for seed in range(n_seeds):
            print(f"  {env_name} | N={n_demos}, seed={seed}")

            # Create environment with this seed
            if env_name == "objectworld":
                env = ObjectworldEnvironment(
                    grid_size=grid_size,
                    n_colors=2,
                    n_objects_per_color=3,
                    discount_factor=discount,
                    feature_type="continuous",
                    seed=seed,
                )
            else:
                env = BinaryworldEnvironment(
                    grid_size=grid_size,
                    discount_factor=discount,
                    seed=seed,
                )

            # Generate noisy expert demonstrations
            panel = env.simulate_demonstrations(
                n_demos=n_demos,
                max_steps=max_steps,
                noise_fraction=noise_fraction,
                seed=seed + 1000,
            )

            # Run MCEIRLNeural (Deep IRL)
            t0 = time.time()
            try:
                neural_policy = run_estimator_mce_neural(
                    env, panel, grid_size, discount,
                    max_epochs=max_epochs, lr=lr,
                )
                neural_evd = compute_evd(
                    env.true_reward, env.transition_matrices,
                    neural_policy, discount,
                )
                neural_time = time.time() - t0
                print(f"    Neural EVD = {neural_evd:.4f} ({neural_time:.1f}s)")
            except Exception as e:
                print(f"    Neural FAILED: {e}")
                neural_evd = float("nan")

            # Run linear MCE-IRL
            t0 = time.time()
            try:
                linear_policy = run_estimator_mce_linear(env, panel, discount)
                linear_evd = compute_evd(
                    env.true_reward, env.transition_matrices,
                    linear_policy, discount,
                )
                linear_time = time.time() - t0
                print(f"    Linear EVD = {linear_evd:.4f} ({linear_time:.1f}s)")
            except Exception as e:
                print(f"    Linear FAILED: {e}")
                linear_evd = float("nan")

            neural_evds.append(neural_evd)
            linear_evds.append(linear_evd)

        results[n_demos] = {
            "neural_evds": neural_evds,
            "linear_evds": linear_evds,
            "neural_mean": float(np.nanmean(neural_evds)),
            "neural_std": float(np.nanstd(neural_evds)),
            "linear_mean": float(np.nanmean(linear_evds)),
            "linear_std": float(np.nanstd(linear_evds)),
        }

    return results


# -----------------------------------------------------------------------
# Display and save
# -----------------------------------------------------------------------


def print_results_table(env_name: str, results: dict) -> None:
    """Print a formatted table of EVD results for one environment."""
    print(f"\n{'=' * 65}")
    print(f"  {env_name} Results (EVD, lower is better)")
    print(f"{'=' * 65}")
    print(f"  {'N demos':>8}  {'Neural (mean +/- std)':>24}  {'Linear (mean +/- std)':>24}")
    print(f"  {'-' * 8}  {'-' * 24}  {'-' * 24}")

    for n_demos in sorted(results.keys()):
        r = results[n_demos]
        neural_str = f"{r['neural_mean']:.4f} +/- {r['neural_std']:.4f}"
        linear_str = f"{r['linear_mean']:.4f} +/- {r['linear_std']:.4f}"
        print(f"  {n_demos:>8}  {neural_str:>24}  {linear_str:>24}")

    print()


def save_results(all_results: dict, output_path: Path) -> None:
    """Save all benchmark results to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Wulfmeier (2016) Deep MaxEnt IRL replication"
    )
    parser.add_argument(
        "--grid-size", type=int, default=GRID_SIZE,
        help="Side length of the grid (default: 32)",
    )
    parser.add_argument(
        "--demo-counts", type=int, nargs="+", default=DEMO_COUNTS,
        help="Demonstration counts to test (default: 8 16 32 64 128)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=N_SEEDS,
        help="Number of random seeds per demo count (default: 5)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=MAX_STEPS,
        help="Trajectory length for demonstrations (default: 50)",
    )
    parser.add_argument(
        "--noise-fraction", type=float, default=NOISE_FRACTION,
        help="Fraction of random actions in demonstrations (default: 0.3)",
    )
    parser.add_argument(
        "--discount", type=float, default=DISCOUNT,
        help="Discount factor (default: 0.9)",
    )
    parser.add_argument(
        "--max-epochs", type=int, default=200,
        help="Maximum training epochs for MCEIRLNeural (default: 200)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01,
        help="Learning rate for MCEIRLNeural (default: 0.01)",
    )
    args = parser.parse_args()

    all_results = {
        "config": {
            "grid_size": args.grid_size,
            "demo_counts": args.demo_counts,
            "n_seeds": args.n_seeds,
            "max_steps": args.max_steps,
            "noise_fraction": args.noise_fraction,
            "discount": args.discount,
            "max_epochs": args.max_epochs,
            "lr": args.lr,
        },
    }

    for env_name in ["objectworld", "binaryworld"]:
        print(f"\nRunning {env_name} benchmark...")
        results = run_benchmark(
            env_name=env_name,
            grid_size=args.grid_size,
            demo_counts=args.demo_counts,
            n_seeds=args.n_seeds,
            max_steps=args.max_steps,
            noise_fraction=args.noise_fraction,
            discount=args.discount,
            max_epochs=args.max_epochs,
            lr=args.lr,
        )
        all_results[env_name] = results
        print_results_table(env_name, results)

    output_path = Path(__file__).parent / "results.json"
    save_results(all_results, output_path)


if __name__ == "__main__":
    main()
