#!/usr/bin/env python3
"""
Rust (1987) Bus Engine Replacement Replication
===============================================

This example replicates the estimation of the bus engine replacement model
from John Rust's seminal 1987 Econometrica paper using the econirl package.

The model:
- Harold Zurcher (superintendent) observes bus mileage each period
- Decides whether to keep running or replace the engine
- Replacement has fixed cost, operating cost increases with mileage
- Goal: Estimate the cost parameters from observed choices

This script demonstrates:
1. Setting up the environment with known parameters
2. Simulating panel data
3. Estimating parameters using NFXP
4. Evaluating parameter recovery
5. Conducting counterfactual analysis
6. Generating publication-ready output

Usage:
    python examples/rust_bus_replication.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

# econirl imports
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.preferences.linear import LinearUtility
from econirl.estimation.nfxp import NFXPEstimator
from econirl.simulation.synthetic import simulate_panel
from econirl.simulation.counterfactual import counterfactual_policy, compute_welfare_effect
from econirl.visualization.policy import (
    plot_choice_probabilities,
    plot_empirical_vs_predicted,
    plot_counterfactual_comparison,
)
from econirl.visualization.value import plot_value_function, create_value_summary_figure


def main():
    print("=" * 70)
    print("Rust (1987) Bus Engine Replacement - econirl Replication")
    print("=" * 70)
    print()

    # =========================================================================
    # Step 1: Create Environment with Known Parameters
    # =========================================================================
    print("Step 1: Creating Rust Bus Environment")
    print("-" * 40)

    # These are the "true" parameters we'll try to recover
    TRUE_OPERATING_COST = 0.001
    TRUE_REPLACEMENT_COST = 3.0

    env = RustBusEnvironment(
        operating_cost=TRUE_OPERATING_COST,
        replacement_cost=TRUE_REPLACEMENT_COST,
        num_mileage_bins=90,
        discount_factor=0.9999,
        scale_parameter=1.0,
        seed=42,
    )

    print(env.describe())

    # =========================================================================
    # Step 2: Simulate Panel Data
    # =========================================================================
    print("\nStep 2: Simulating Panel Data")
    print("-" * 40)

    N_INDIVIDUALS = 500
    N_PERIODS = 100
    SEED = 12345

    panel = simulate_panel(
        env,
        n_individuals=N_INDIVIDUALS,
        n_periods=N_PERIODS,
        seed=SEED,
    )

    print(f"Simulated panel with:")
    print(f"  - {panel.num_individuals} individuals")
    print(f"  - {N_PERIODS} periods each")
    print(f"  - {panel.num_observations:,} total observations")

    # Summary statistics
    actions = panel.get_all_actions()
    replace_rate = (actions == 1).float().mean().item()
    print(f"\nEmpirical replacement rate: {replace_rate:.1%}")

    # =========================================================================
    # Step 3: Set Up Estimation
    # =========================================================================
    print("\nStep 3: Setting Up NFXP Estimation")
    print("-" * 40)

    # Create utility specification
    utility = LinearUtility.from_environment(env)
    print(f"Parameters to estimate: {utility.parameter_names}")

    # Get problem specification and transitions
    problem = env.problem_spec
    transitions = env.transition_matrices

    # Create estimator
    estimator = NFXPEstimator(
        se_method="asymptotic",
        optimizer="L-BFGS-B",
        inner_tol=1e-10,
        outer_tol=1e-6,
        verbose=True,
    )

    # =========================================================================
    # Step 4: Estimate Parameters
    # =========================================================================
    print("\nStep 4: Running NFXP Estimation")
    print("-" * 40)

    result = estimator.estimate(
        panel=panel,
        utility=utility,
        problem=problem,
        transitions=transitions,
    )

    # =========================================================================
    # Step 5: Display Results
    # =========================================================================
    print("\nStep 5: Estimation Results")
    print("-" * 40)

    # Full summary (StatsModels-style)
    print(result.summary())

    # Parameter comparison
    print("\nParameter Recovery:")
    print("-" * 40)
    true_params = env.get_true_parameter_vector()

    for i, name in enumerate(result.parameter_names):
        true_val = true_params[i].item()
        est_val = result.parameters[i].item()
        se = result.standard_errors[i].item()
        error = est_val - true_val
        t_stat = error / se if se > 0 else float('inf')

        print(f"{name}:")
        print(f"  True:      {true_val:.6f}")
        print(f"  Estimated: {est_val:.6f} (SE: {se:.6f})")
        print(f"  Error:     {error:.6f} ({error/true_val*100:.1f}%)")
        print(f"  t-stat:    {t_stat:.2f}")
        print()

    # =========================================================================
    # Step 6: Counterfactual Analysis
    # =========================================================================
    print("\nStep 6: Counterfactual Analysis")
    print("-" * 40)

    # What if replacement cost increases by 50%?
    print("Scenario: Replacement cost increases by 50%")

    new_params = result.parameters.clone()
    new_params[1] *= 1.5  # replacement_cost

    cf = counterfactual_policy(
        result=result,
        new_parameters=new_params,
        utility=utility,
        problem=problem,
        transitions=transitions,
    )

    # Policy changes
    baseline_replace_avg = cf.baseline_policy[:, 1].mean().item()
    cf_replace_avg = cf.counterfactual_policy[:, 1].mean().item()

    print(f"\nAverage replacement probability:")
    print(f"  Baseline:       {baseline_replace_avg:.4f}")
    print(f"  Counterfactual: {cf_replace_avg:.4f}")
    print(f"  Change:         {cf_replace_avg - baseline_replace_avg:.4f}")

    # Welfare effects
    welfare = compute_welfare_effect(cf, transitions, use_stationary=True)
    print(f"\nWelfare effects:")
    print(f"  Baseline EV:       {welfare['baseline_expected_value']:.2f}")
    print(f"  Counterfactual EV: {welfare['counterfactual_expected_value']:.2f}")
    print(f"  Total change:      {welfare['total_welfare_change']:.2f}")

    # =========================================================================
    # Step 7: Visualization
    # =========================================================================
    print("\nStep 7: Generating Visualizations")
    print("-" * 40)

    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)

    # 1. Choice probabilities
    fig = plot_choice_probabilities(
        result,
        action_labels=["Keep", "Replace"],
        xlabel="Mileage (bins)",
        title="Estimated Choice Probabilities",
    )
    fig.savefig("output/choice_probabilities.png", dpi=150, bbox_inches="tight")
    print("Saved: output/choice_probabilities.png")
    plt.close(fig)

    # 2. Model fit
    fig = plot_empirical_vs_predicted(
        panel,
        result,
        action_idx=1,
        action_label="Replace",
        xlabel="Mileage (bins)",
        title="Model Fit: Empirical vs Predicted",
    )
    fig.savefig("output/model_fit.png", dpi=150, bbox_inches="tight")
    print("Saved: output/model_fit.png")
    plt.close(fig)

    # 3. Counterfactual comparison
    fig = plot_counterfactual_comparison(
        cf,
        action_labels=["Keep", "Replace"],
        xlabel="Mileage (bins)",
    )
    fig.savefig("output/counterfactual.png", dpi=150, bbox_inches="tight")
    print("Saved: output/counterfactual.png")
    plt.close(fig)

    # 4. Value function summary
    fig = create_value_summary_figure(
        result,
        utility,
        problem,
        transitions,
        action_labels=["Keep", "Replace"],
    )
    fig.savefig("output/value_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: output/value_summary.png")
    plt.close(fig)

    # =========================================================================
    # Step 8: Export Results
    # =========================================================================
    print("\nStep 8: Exporting Results")
    print("-" * 40)

    # LaTeX table
    result.to_latex("output/estimation_results.tex", caption="Rust (1987) Replication")
    print("Saved: output/estimation_results.tex")

    # DataFrame
    df = result.to_dataframe()
    df.to_csv("output/estimation_results.csv")
    print("Saved: output/estimation_results.csv")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Replication Complete!")
    print("=" * 70)
    print(f"\nKey findings:")
    print(f"  - Operating cost: {result.parameters[0]:.6f} (true: {TRUE_OPERATING_COST})")
    print(f"  - Replacement cost: {result.parameters[1]:.4f} (true: {TRUE_REPLACEMENT_COST})")
    print(f"  - Log-likelihood: {result.log_likelihood:,.2f}")
    print(f"  - Prediction accuracy: {result.goodness_of_fit.prediction_accuracy:.1%}")
    print(f"\nOutput files saved to: output/")


if __name__ == "__main__":
    main()
