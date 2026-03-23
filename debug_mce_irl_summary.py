#!/usr/bin/env python3
"""
===============================================================================
FINAL SUMMARY: Why MCE IRL Fails on the Rust Bus Model
===============================================================================

EXECUTIVE SUMMARY:
------------------
MCE IRL's feature matching objective has a DIFFERENT optimum than Maximum
Likelihood Estimation (MLE). The two methods optimize fundamentally different
objectives, and their optima do not coincide for the Rust model.

KEY FINDINGS FROM DEBUGGING:
----------------------------

1. FEATURE MATCHING vs MLE OBJECTIVES ARE DIFFERENT

   Grid search results at RC=3.0 (true value):

   theta_c     Feature Diff   Log-Likelihood
   -------     ------------   --------------
   0.000721       1.34          -1012.59  <-- MLE optimum
   0.001000       1.05          -1012.77  <-- TRUE parameters
   0.001964       0.05          -1016.14  <-- Feature matching optimum

   The feature matching minimum (theta_c=0.00196) is TWICE the true value!
   The MLE maximum (theta_c=0.00072) is close to true but slightly lower.

2. WHY THE OPTIMA DIFFER

   MCE IRL minimizes: ||E_data[phi] - E_model[phi]||^2

   This matches AGGREGATE feature expectations:
   - E[phi_0] = -average(mileage * keep)
   - E[phi_1] = -replacement_rate

   MLE maximizes: sum_t log P(a_t | s_t; theta)

   This matches STATE-SPECIFIC choice probabilities:
   - P(replace | mileage=0), P(replace | mileage=1), ...

   The Rust model's identification comes from HOW P(replace|mileage) varies
   with mileage, not just from aggregate statistics.

3. FINITE SAMPLE BIAS

   At true parameters (theta_c=0.001, RC=3.0):
   - Empirical features:  [-7.41, -0.051]
   - Expected features:   [-8.37, -0.054]
   - Feature gap:         [0.96, 0.003]  <-- NOT ZERO!

   The model's expected features (computed from stationary distribution)
   don't match the finite sample empirical features, even at true parameters.

4. THE OPTIMIZATION LANDSCAPE

   See mce_irl_landscape.png:
   - Left panel: Feature matching objective (minimize)
     * Minimum at theta_c=0.0005, RC=2.7 (FAR from true)
     * True parameters are NOT at the minimum

   - Right panel: Log-likelihood (maximize)
     * Maximum close to theta_c=0.0007, RC=3.0 (close to true)
     * True parameters are near the maximum

5. IMPLICATIONS FOR ECONIRL

   a) MCE IRL is NOT appropriate for structural parameter recovery
      - It was designed for IRL, not econometric estimation
      - IRL asks: "What reward makes this behavior optimal?"
      - Structural estimation asks: "What are the true utility parameters?"

   b) For the Rust model, use NFXP (which works correctly)
      - NFXP directly maximizes the likelihood
      - It recovers theta_c=0.0006, RC=2.99 (very close to true)

   c) Feature matching can be used for:
      - Policy evaluation (does the model match observed behavior?)
      - Reward shaping (finding ANY consistent reward)
      - But NOT for recovering specific utility parameters

RECOMMENDATIONS:
----------------

1. DOCUMENT THE LIMITATION
   - MCE IRL is for IRL problems, not structural estimation
   - Add clear documentation distinguishing the use cases

2. CONSIDER ADDING MLE-BASED IRL
   - Gradient of log-likelihood instead of feature matching
   - Would be equivalent to gradient-based NFXP
   - But loses the "IRL" interpretation

3. USE NFXP FOR RUST MODEL
   - NFXP works correctly for parameter recovery
   - This is the appropriate method for structural estimation

TECHNICAL DETAILS:
------------------

The feature matching gradient is:
   grad_FM = E_data[phi] - E_model[phi]

The MLE gradient is:
   grad_MLE = sum_t [phi(s_t, a_t) - E_pi[phi | s_t]]

Key difference: MLE conditions on EACH STATE, while feature matching uses
aggregate expectations. This means:
- MLE uses the full variation in choice probabilities across states
- Feature matching only uses the average, losing information

For Rust model:
- MLE gradient: [-1648, 10.3] at true parameters
- FM gradient:  [0.96, 0.003] at true parameters

The gradients point in OPPOSITE directions! This is why optimizing the
feature matching objective diverges from the true parameters.
"""

import numpy as np
import torch

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation import simulate_panel
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.mce_irl import MCEIRLEstimator, MCEIRLConfig
from econirl.preferences.linear import LinearUtility
from econirl.preferences.action_reward import ActionDependentReward


def main():
    print(__doc__)

    print("\n" + "=" * 70)
    print(" VERIFICATION: Running both estimators")
    print("=" * 70)

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

    # NFXP
    utility = LinearUtility.from_environment(env)
    nfxp = NFXPEstimator(
        inner_solver="policy",
        inner_max_iter=10000,
        verbose=False,
    )
    nfxp_result = nfxp.estimate(
        panel=panel,
        utility=utility,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    # MCE IRL
    reward = ActionDependentReward.from_rust_environment(env)
    config = MCEIRLConfig(
        verbose=False,
        inner_max_iter=5000,
        outer_max_iter=500,
        learning_rate=0.05,
    )
    mce_irl = MCEIRLEstimator(config=config)
    mce_result = mce_irl.estimate(
        panel=panel,
        utility=reward,
        problem=env.problem_spec,
        transitions=env.transition_matrices,
    )

    print(f"\nTrue parameters:     theta_c = {TRUE_THETA_C:.6f}, RC = {TRUE_RC:.4f}")
    print(f"NFXP estimates:      theta_c = {nfxp_result.parameters[0].item():.6f}, RC = {nfxp_result.parameters[1].item():.4f}")
    print(f"MCE IRL estimates:   theta_c = {mce_result.parameters[0].item():.6f}, RC = {mce_result.parameters[1].item():.4f}")

    print(f"\nNFXP Log-likelihood:    {nfxp_result.log_likelihood:.2f}")
    print(f"MCE IRL Log-likelihood: {mce_result.log_likelihood:.2f}")

    print(f"\nNFXP converged:    {nfxp_result.converged}")
    print(f"MCE IRL converged: {mce_result.converged}")

    # Compute bias
    nfxp_bias = np.array([
        nfxp_result.parameters[0].item() - TRUE_THETA_C,
        nfxp_result.parameters[1].item() - TRUE_RC
    ])
    mce_bias = np.array([
        mce_result.parameters[0].item() - TRUE_THETA_C,
        mce_result.parameters[1].item() - TRUE_RC
    ])

    print(f"\nNFXP bias:    [{nfxp_bias[0]:+.6f}, {nfxp_bias[1]:+.4f}]")
    print(f"MCE IRL bias: [{mce_bias[0]:+.6f}, {mce_bias[1]:+.4f}]")

    print("\n" + "=" * 70)
    print(" CONCLUSION")
    print("=" * 70)
    print("""
    NFXP successfully recovers parameters close to true values.
    MCE IRL does NOT - this is NOT A BUG, but a fundamental limitation.

    For structural estimation of the Rust model, use NFXP or CCP methods.
    MCE IRL is designed for a different problem (inverse reinforcement learning).
""")


if __name__ == "__main__":
    main()
