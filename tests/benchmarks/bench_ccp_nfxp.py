#!/usr/bin/env python
"""Benchmark comparison: CCP estimators vs NFXP on Rust bus data.

This script compares:
1. Parameter recovery
2. Standard errors
3. Estimation speed
4. NPL convergence

Usage:
    python -m tests.benchmarks.bench_ccp_nfxp
    # or
    python tests/benchmarks/bench_ccp_nfxp.py
"""

import time
import torch
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.ccp import CCPEstimator
from econirl.simulation.synthetic import simulate_panel


def print_separator(char="=", width=80):
    print(char * width)


def print_header(title: str):
    print_separator()
    print(f"{title:^80}")
    print_separator()


def run_benchmark(
    n_individuals: int = 500,
    n_periods: int = 100,
    seed: int = 42,
    npl_iterations: int = 10,
):
    """Run full benchmark comparison."""

    print_header("CCP vs NFXP Benchmark on Rust Bus Data")

    # Setup environment
    print("\n[Setup]")
    env = RustBusEnvironment(
        operating_cost=0.001,
        replacement_cost=3.0,
        num_mileage_bins=90,
        discount_factor=0.9999,
        scale_parameter=1.0,
        seed=seed,
    )
    print(f"Environment: RustBusEnvironment")
    print(f"  operating_cost = {env._operating_cost}")
    print(f"  replacement_cost = {env._replacement_cost}")
    print(f"  num_states = {env.num_states}")
    print(f"  discount_factor = {env.problem_spec.discount_factor}")

    # Get problem components
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)
    true_params = env.get_true_parameter_vector()

    # Simulate data
    print(f"\n[Data Simulation]")
    print(f"Simulating panel with {n_individuals} individuals × {n_periods} periods...")
    panel = simulate_panel(env, n_individuals=n_individuals, n_periods=n_periods, seed=seed)
    n_obs = panel.num_observations
    print(f"Total observations: {n_obs:,}")

    # Define estimators
    # Note: For high β (like 0.9999), we need:
    # - Policy iteration instead of value iteration (faster convergence)
    # - Looser inner tolerance (1e-5 instead of 1e-10)
    # - Good initial values (near true params)
    estimators = {
        "NFXP": NFXPEstimator(
            se_method="asymptotic",
            verbose=False,
            inner_solver="policy",  # Use policy iteration for high β
            inner_tol=1e-5,  # Looser tolerance for high β
            inner_max_iter=500,
            outer_max_iter=200,
            outer_tol=1e-6,
        ),
        "Hotz-Miller": CCPEstimator(
            num_policy_iterations=1,
            se_method="asymptotic",
            verbose=False,
        ),
        f"NPL (K={npl_iterations})": CCPEstimator(
            num_policy_iterations=npl_iterations,
            se_method="asymptotic",
            convergence_tol=1e-6,
            verbose=False,
        ),
    }

    # Good initial values (near true params) help optimization
    # In practice, you might use a preliminary consistent estimator
    initial_params = true_params.clone()

    # Run estimators
    results = {}
    times = {}

    print(f"\n[Estimation]")
    for name, estimator in estimators.items():
        print(f"Running {name}...", end=" ", flush=True)
        start = time.time()
        result = estimator.estimate(
            panel, utility, problem, transitions, initial_params=initial_params
        )
        elapsed = time.time() - start
        results[name] = result
        times[name] = elapsed
        print(f"done ({elapsed:.2f}s)")

    # Print comparison table
    print_header("Results Comparison")

    param_names = utility.parameter_names

    # Parameters
    print("\nParameters:")
    print(f"{'':20} {'True':>12} ", end="")
    for name in estimators.keys():
        print(f"{name:>15}", end=" ")
    print()
    print("-" * (20 + 13 + 16 * len(estimators)))

    for i, pname in enumerate(param_names):
        print(f"{pname:20} {true_params[i].item():12.5f} ", end="")
        for name in estimators.keys():
            est = results[name].parameters[i].item()
            print(f"{est:15.5f}", end=" ")
        print()

    # Standard errors
    print("\nStandard Errors:")
    print(f"{'':20} ", end="")
    for name in estimators.keys():
        print(f"{name:>15}", end=" ")
    print()
    print("-" * (20 + 16 * len(estimators)))

    for i, pname in enumerate(param_names):
        print(f"{pname:20} ", end="")
        for name in estimators.keys():
            se = results[name].standard_errors[i].item()
            if np.isfinite(se):
                print(f"{se:15.6f}", end=" ")
            else:
                print(f"{'N/A':>15}", end=" ")
        print()

    # Estimation metrics
    print("\nEstimation Metrics:")
    print("-" * 80)

    # Time
    print(f"{'Estimation Time':20} ", end="")
    for name in estimators.keys():
        print(f"{times[name]:14.2f}s", end=" ")
    print()

    # Speedup
    nfxp_time = times["NFXP"]
    print(f"{'Speedup vs NFXP':20} ", end="")
    for name in estimators.keys():
        speedup = nfxp_time / times[name]
        print(f"{speedup:14.1f}x", end=" ")
    print()

    # Log-likelihood
    print(f"{'Log-Likelihood':20} ", end="")
    for name in estimators.keys():
        ll = results[name].log_likelihood
        print(f"{ll:15.1f}", end=" ")
    print()

    # Prediction accuracy
    print(f"{'Prediction Accuracy':20} ", end="")
    for name in estimators.keys():
        acc = results[name].goodness_of_fit.prediction_accuracy * 100
        print(f"{acc:14.1f}%", end=" ")
    print()

    # NPL convergence info
    npl_name = f"NPL (K={npl_iterations})"
    if npl_name in results:
        npl_result = results[npl_name]
        print(f"\nNPL Convergence:")
        print(f"  Iterations: {npl_result.num_iterations}")
        print(f"  Converged: {npl_result.metadata.get('npl_converged', 'N/A')}")

    # Parameter recovery analysis
    print_header("Parameter Recovery Analysis")

    print(f"{'Method':20} {'Param':15} {'Bias':>12} {'SE':>12} {'t-stat':>12}")
    print("-" * 80)

    for name in estimators.keys():
        result = results[name]
        for i, pname in enumerate(param_names):
            est = result.parameters[i].item()
            true_val = true_params[i].item()
            se = result.standard_errors[i].item()
            bias = est - true_val

            if np.isfinite(se) and se > 0:
                t_stat = bias / se
                t_str = f"{t_stat:12.2f}"
            else:
                t_str = f"{'N/A':>12}"

            print(f"{name:20} {pname:15} {bias:12.5f} {se:12.6f} {t_str}")

    print_separator()
    print("Benchmark complete.")

    return results, times


def main():
    """Run benchmark with default settings."""
    run_benchmark(
        n_individuals=500,
        n_periods=100,
        seed=42,
        npl_iterations=10,
    )


if __name__ == "__main__":
    main()
