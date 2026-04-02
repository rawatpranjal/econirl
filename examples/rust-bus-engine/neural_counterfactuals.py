#!/usr/bin/env python3
"""
Neural Counterfactuals on the Rust Bus Engine
==============================================

Demonstrates every neural counterfactual type on the Rust (1987) bus
engine replacement problem. Fits NFXP as the structural baseline and
NeuralGLADIUS as the neural estimator, then runs global perturbation,
local perturbation, transition counterfactuals, choice set restrictions,
sieve compression, policy Jacobian analysis, and (optionally) SHAP.

The structural estimator uses discount 0.9999 (Rust's original value).
NeuralGLADIUS uses discount 0.95 because Q-values become order 10,000
with 0.9999 and tiny reward differences get lost in numerical noise.

Usage:
    python examples/rust-bus-engine/neural_counterfactuals.py
"""

import numpy as np
import jax.numpy as jnp
import torch

from econirl import NFXP
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.datasets import load_rust_bus
from econirl.core.types import DDCProblem
from econirl.simulation import (
    neural_global_perturbation,
    neural_local_perturbation,
    neural_transition_counterfactual,
    neural_choice_set_counterfactual,
    neural_sieve_compression,
    neural_policy_jacobian,
    neural_perturbation_sweep,
    compute_stationary_distribution,
)


def banner(title: str) -> None:
    """Print a section banner."""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def main():
    # ==================================================================
    #  FITTING: structural (NFXP) and neural (NeuralGLADIUS) baselines
    # ==================================================================
    banner("FITTING MODELS")

    df = load_rust_bus()
    n_obs = len(df)
    n_buses = df["bus_id"].nunique()
    repl_rate = df["replaced"].mean()
    print(f"Data: {n_obs:,} observations, {n_buses} buses, "
          f"replacement rate {repl_rate:.4f}")
    print()

    # --- NFXP (structural baseline) ---
    print("Fitting NFXP (discount=0.9999) ...")
    nfxp = NFXP(n_states=90, n_actions=2, discount=0.9999, verbose=False)
    nfxp.fit(data=df, state="mileage_bin", action="replaced", id="bus_id")
    print(f"  theta_c = {nfxp.params_['theta_c']:.6f}")
    print(f"  RC      = {nfxp.params_['RC']:.4f}")
    print(f"  Log-lik = {nfxp.log_likelihood_:.2f}")
    print(f"  P(replace | s=50) = {nfxp.policy_[50, 1]:.4f}")
    print()

    # --- NeuralGLADIUS ---
    # With value_scale fix, GLADIUS can now train at beta=0.9999.
    # The networks predict in per-period utility units (~[-10, 10])
    # and multiply by 1/(1-beta)=10000 to reach the true Q-value scale.
    print("Fitting NeuralGLADIUS (discount=0.9999, value_scale=auto) ...")
    gladius = NeuralGLADIUS(
        n_actions=2,
        discount=0.9999,
        q_hidden_dim=64,
        q_num_layers=2,
        ev_hidden_dim=64,
        ev_num_layers=2,
        max_epochs=500,
        batch_size=256,
        patience=100,
        bellman_weight=1.0,
        alternating_updates=True,
        verbose=True,
    )
    gladius.fit(
        data=df, state="mileage_bin", action="replaced", id="bus_id",
    )
    print(f"  Epochs trained: {gladius.n_epochs_}")
    # NeuralGLADIUS infers n_states from the max observed state index.
    # The Rust bus data may not visit all 90 bins, so we override to
    # ensure the reward matrix covers the full state space.
    gladius._n_states = 90
    print(f"  P(replace | s=50) = {gladius.policy_[50, 1]:.4f}")
    print()

    # Extract neural reward matrix and build transition tensor
    reward_matrix = jnp.array(gladius.reward_matrix_)
    n_states = 90
    n_actions = 2

    # Build transition tensor (A, S, S) from NFXP's estimated transitions
    keep_trans = nfxp.transitions_  # (S, S)
    transitions = np.zeros((n_actions, n_states, n_states))
    transitions[0] = keep_trans
    for s in range(n_states):
        transitions[1, s, :] = keep_trans[0, :]  # replace resets to state 0
    transitions = jnp.array(transitions)

    problem = DDCProblem(
        num_states=n_states,
        num_actions=n_actions,
        discount_factor=0.9999,
        scale_parameter=1.0,
    )

    # Also compute the NFXP (linear) reward matrix for comparison.
    # This is R(s,a) = theta' * phi(s,a), the structural reward.
    nfxp_reward = jnp.array(nfxp.reward_matrix_)

    print(f"Neural reward shape: {reward_matrix.shape}")
    print(f"Neural reward range: [{float(reward_matrix.min()):.3f}, "
          f"{float(reward_matrix.max()):.3f}]")
    print(f"NFXP reward range:   [{float(nfxp_reward.min()):.3f}, "
          f"{float(nfxp_reward.max()):.3f}]")

    # ==================================================================
    #  SECTION A: Global perturbation sweep
    # ==================================================================
    banner("SECTION A: Global Perturbation Sweep")
    print("Question: How does the replacement rate respond to a uniform")
    print("cost penalty on the replace action?")
    print()

    deltas = [0.0, 1.0, 2.0, 3.0, 5.0, 10.0]
    delta_grid = jnp.array(deltas)
    sweep = neural_perturbation_sweep(
        reward_matrix, action=1, delta_grid=delta_grid,
        problem=problem, transitions=transitions,
    )

    # Run same sweep on NFXP reward for comparison
    sweep_nfxp = neural_perturbation_sweep(
        nfxp_reward, action=1, delta_grid=delta_grid,
        problem=problem, transitions=transitions,
    )

    print(f"{'Delta':>8s}  {'Neural P(repl)':>16s}  {'NFXP P(repl)':>14s}  {'Diff':>8s}")
    print(f"{'-----':>8s}  {'--------------':>16s}  {'------------':>14s}  {'----':>8s}")
    for i, delta in enumerate(deltas):
        n_p = sweep['mean_action_prob'][i]
        s_p = sweep_nfxp['mean_action_prob'][i]
        print(f"{delta:8.1f}  {n_p:16.4f}  {s_p:14.4f}  {n_p - s_p:+8.4f}")

    print()
    print(f"The neural model has higher baseline replacement (nonlinear")
    print(f"reward surface) but both respond in the same direction to the")
    print(f"penalty. The gap narrows at large delta because both converge")
    print(f"toward zero replacement.")

    # ==================================================================
    #  SECTION B: Local perturbation at high mileage
    # ==================================================================
    banner("SECTION B: Local Perturbation (High-Mileage States)")
    print("Question: What happens to the policy if operating costs increase")
    print("only at high-mileage states (bins above 60)?")
    print()

    high_mileage_mask = jnp.arange(n_states) > 60

    affected_states = jnp.where(high_mileage_mask)[0]
    print(f"  {'delta':>6s}  {'Neural P(r|s>60)':>18s}  {'NFXP P(r|s>60)':>16s}")
    print(f"  {'-----':>6s}  {'----------------':>18s}  {'--------------':>16s}")
    for delta in [0.0, 1.0, 2.0, 5.0]:
        result_n = neural_local_perturbation(
            reward_matrix, action=0, delta=delta,
            state_mask=high_mileage_mask,
            problem=problem, transitions=transitions,
        )
        result_s = neural_local_perturbation(
            nfxp_reward, action=0, delta=delta,
            state_mask=high_mileage_mask,
            problem=problem, transitions=transitions,
        )
        n_p = float(result_n.counterfactual_policy[affected_states, 1].mean())
        s_p = float(result_s.counterfactual_policy[affected_states, 1].mean())
        print(f"  {delta:6.0f}  {n_p:18.4f}  {s_p:16.4f}")
    print()
    print(f"  Both models shift replacement upward when operating costs rise")
    print(f"  at high mileage, but the neural baseline is already much higher.")

    # ==================================================================
    #  SECTION C: Transition counterfactual (faster depreciation)
    # ==================================================================
    banner("SECTION C: Transition Counterfactual")
    print("Question: What if engines deteriorate faster? We change the")
    print("mileage increment probabilities from (0.39, 0.60, 0.01) to")
    print("(0.20, 0.50, 0.30).")
    print()

    new_transitions = jnp.zeros_like(transitions)
    new_probs = [0.20, 0.50, 0.30]
    for a in range(n_actions):
        for s in range(n_states):
            if a == 1:
                # Replace resets to state 0, then transitions from 0
                row = jnp.zeros(n_states)
                for i, p in enumerate(new_probs):
                    dest = min(i, n_states - 1)
                    row = row.at[dest].add(p)
                new_transitions = new_transitions.at[a, s, :].set(row)
            else:
                # Keep: transition from current state
                row = jnp.zeros(n_states)
                for i, p in enumerate(new_probs):
                    dest = min(s + i, n_states - 1)
                    row = row.at[dest].add(p)
                new_transitions = new_transitions.at[a, s, :].set(row)

    result_n = neural_transition_counterfactual(
        reward_matrix, new_transitions, problem, transitions,
    )
    result_s = neural_transition_counterfactual(
        nfxp_reward, new_transitions, problem, transitions,
    )
    print(f"  {'':>30s}  {'Neural':>10s}  {'NFXP':>10s}")
    print(f"  {'':>30s}  {'------':>10s}  {'----':>10s}")
    print(f"  {'Baseline mean P(replace)':>30s}  "
          f"{float(result_n.baseline_policy[:, 1].mean()):10.4f}  "
          f"{float(result_s.baseline_policy[:, 1].mean()):10.4f}")
    print(f"  {'Counterfactual mean P(replace)':>30s}  "
          f"{float(result_n.counterfactual_policy[:, 1].mean()):10.4f}  "
          f"{float(result_s.counterfactual_policy[:, 1].mean()):10.4f}")
    print(f"  {'Welfare change':>30s}  "
          f"{result_n.welfare_change:10.2f}  "
          f"{result_s.welfare_change:10.2f}")
    print()

    # ==================================================================
    #  SECTION D: Choice set counterfactual
    # ==================================================================
    banner("SECTION D: Choice Set Counterfactual")
    print("Question: What if replacement is mandatory above bin 80, and a")
    print("warranty prevents replacement below bin 10?")
    print()

    action_mask = jnp.ones((n_states, n_actions), dtype=jnp.bool_)

    # Mandatory replacement above bin 80: block keep (action 0)
    action_mask = action_mask.at[81:, 0].set(False)

    # Warranty below bin 10: block replace (action 1)
    action_mask = action_mask.at[:10, 1].set(False)

    result_n = neural_choice_set_counterfactual(
        reward_matrix, action_mask, problem, transitions,
    )
    result_s = neural_choice_set_counterfactual(
        nfxp_reward, action_mask, problem, transitions,
    )

    n_blocked = int((~action_mask).sum())
    print(f"  State-action pairs blocked: {n_blocked}")
    print()

    print(f"  {'State':>6s}  {'Neural base':>12s}  {'Neural CF':>10s}  "
          f"{'NFXP base':>10s}  {'NFXP CF':>8s}  {'Note'}")
    print(f"  {'-----':>6s}  {'-----------':>12s}  {'---------':>10s}  "
          f"{'---------':>10s}  {'-------':>8s}  {'----'}")
    for s in [5, 9, 10, 50, 79, 80, 85]:
        nb = float(result_n.baseline_policy[s, 1])
        nc = float(result_n.counterfactual_policy[s, 1])
        sb = float(result_s.baseline_policy[s, 1])
        sc = float(result_s.counterfactual_policy[s, 1])
        note = ""
        if s < 10:
            note = "warranty"
        elif s > 80:
            note = "mandatory"
        print(f"  {s:6d}  {nb:12.4f}  {nc:10.4f}  {sb:10.4f}  {sc:8.4f}  {note}")

    print()
    print(f"  Neural welfare change: {result_n.welfare_change:.2f}")
    print(f"  NFXP welfare change:   {result_s.welfare_change:.2f}")
    print()
    print(f"  Both models agree on forced states (0.0 or 1.0). The gap at")
    print(f"  unconstrained states (10-79) reflects the nonlinear reward.")

    # ==================================================================
    #  SECTION E: Sieve compression
    # ==================================================================
    banner("SECTION E: Sieve Compression")
    print("Question: Can the neural reward surface be explained by the")
    print("same two features used in the structural model (operating cost")
    print("and replacement cost)?")
    print()

    # Build the Rust bus feature matrix (S, A, K) with K=2
    mileage = jnp.arange(n_states, dtype=jnp.float32)
    features = jnp.zeros((n_states, n_actions, 2))
    features = features.at[:, 0, 0].set(-mileage)   # keep: -theta_c * s
    features = features.at[:, 1, 1].set(-1.0)        # replace: -RC

    sieve = neural_sieve_compression(
        reward_matrix, features, parameter_names=["theta_c", "RC"],
    )

    print(f"  Sieve projection results:")
    for i, name in enumerate(sieve["parameter_names"]):
        theta_val = sieve["theta"][i]
        se_val = sieve["se"][i]
        print(f"    {name:>8s} = {theta_val:10.6f}  (SE = {se_val:.6f})")

    print(f"    R-squared = {sieve['r_squared']:.4f}")
    print()

    if nfxp.params_ is not None:
        print(f"  Structural (NFXP) for comparison:")
        print(f"    theta_c = {nfxp.params_['theta_c']:.6f}")
        print(f"    RC      = {nfxp.params_['RC']:.4f}")
        print()

    print("  A high R-squared means the linear basis captures the neural")
    print("  reward surface well, and the sieve coefficients are reliable.")

    # ==================================================================
    #  SECTION F: Policy Jacobian
    # ==================================================================
    banner("SECTION F: Policy Jacobian")
    print("Question: Which states' rewards most affect the replacement")
    print("probability at state 50?")
    print()
    print("Computing Jacobian (180 Bellman solves) ...")

    jacobian = neural_policy_jacobian(
        reward_matrix, problem, transitions,
        epsilon=1e-4, target_action=1,
    )

    # Sensitivity of P(replace | s=50) to reward perturbations
    # jacobian[s, s', a'] = d pi(s, target_action) / d r(s', a')
    sensitivity_at_50 = jacobian[50, :, :]  # shape (S, A)

    # Find the top 5 most influential (s', a') pairs
    flat_idx = jnp.argsort(jnp.abs(sensitivity_at_50).ravel())[::-1]
    print(f"  Top 10 reward perturbations affecting P(replace | s=50):")
    print(f"  {'State':>6s}  {'Action':>7s}  {'d pi / d r':>12s}")
    print(f"  {'-----':>6s}  {'------':>7s}  {'----------':>12s}")
    for rank in range(10):
        idx = int(flat_idx[rank])
        sp = idx // n_actions
        ap = idx % n_actions
        val = float(sensitivity_at_50[sp, ap])
        action_name = "keep" if ap == 0 else "replace"
        print(f"  {sp:6d}  {action_name:>7s}  {val:12.6f}")

    print()

    # Aggregate: total sensitivity by state (sum over actions)
    total_by_state = jnp.abs(sensitivity_at_50).sum(axis=1)
    top_states = jnp.argsort(total_by_state)[::-1][:5]
    print(f"  States with largest total influence on P(replace|s=50):")
    for s in top_states:
        s = int(s)
        print(f"    State {s:3d}: total |d pi / d r| = "
              f"{float(total_by_state[s]):.6f}")

    # ==================================================================
    #  SECTION G: SHAP (optional)
    # ==================================================================
    banner("SECTION G: SHAP Feature Attribution (Optional)")

    try:
        import shap  # noqa: F401

        print("SHAP is available. Computing feature attributions for the")
        print("neural Q-network ...")
        print()

        # Build input data for the Q-network: state features for all states
        with torch.no_grad():
            state_indices = torch.arange(n_states, dtype=torch.long)
            ctx_default = torch.zeros(n_states, dtype=torch.long)
            s_feat = gladius._state_encoder(state_indices)
            ctx_feat = gladius._context_encoder(ctx_default)

            # Compute Q-values for action 1 (replace) as a function of state
            def q_replace(state_feat_np):
                sf = torch.tensor(state_feat_np, dtype=torch.float32)
                cf = ctx_feat[:len(sf)]
                a_oh = torch.zeros(len(sf), n_actions)
                a_oh[:, 1] = 1.0  # replace action
                q_val = gladius._q_net(sf, cf, a_oh)
                return q_val.numpy()

            background = s_feat.numpy()[:20]
            explainer = shap.KernelExplainer(q_replace, background)
            shap_values = explainer.shap_values(s_feat.numpy())

            print(f"  SHAP values shape: {shap_values.shape}")
            print(f"  Mean |SHAP| across states: {np.abs(shap_values).mean():.4f}")

    except ImportError:
        print("SHAP is not installed. Skipping feature attribution.")
        print("Install with: pip install shap")

    # ==================================================================
    #  Validation: Do the neural counterfactuals agree with structural?
    # ==================================================================
    banner("VALIDATION: Neural vs Structural Ground Truth")
    print("Both NFXP and NeuralGLADIUS use beta=0.9999. We compare")
    print("policies state by state to check if the neural reward")
    print("produces the same optimal behavior as the structural model.")
    print()

    # Compare policies at key states
    print(f"  {'State':>6s}  {'NFXP P(repl)':>14s}  {'Neural P(repl)':>14s}  {'Diff':>8s}")
    print(f"  {'-----':>6s}  {'------------':>14s}  {'--------------':>14s}  {'----':>8s}")
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import value_iteration as vi
    operator = SoftBellmanOperator(problem, transitions)
    neural_sol = vi(operator, reward_matrix)
    for s in [0, 10, 20, 30, 40, 50, 60, 70, 80, 89]:
        nfxp_p = float(nfxp.policy_[s, 1])
        neural_p = float(neural_sol.policy[s, 1])
        diff = neural_p - nfxp_p
        print(f"  {s:6d}  {nfxp_p:14.4f}  {neural_p:14.4f}  {diff:+8.4f}")

    # Policy correlation
    nfxp_flat = nfxp.policy_[:, 1]
    neural_flat = np.asarray(neural_sol.policy[:, 1])
    corr = float(np.corrcoef(nfxp_flat, neural_flat)[0, 1])
    mae = float(np.abs(nfxp_flat - neural_flat).mean())
    print()
    print(f"  Policy correlation: {corr:.4f}")
    print(f"  Mean absolute error: {mae:.4f}")
    print()

    # Sieve compression should recover NFXP parameters if neural reward is correct
    print("Sieve compression comparison (both at beta=0.9999):")
    print(f"  NFXP structural:  theta_c={nfxp.params_['theta_c']:.6f}, "
          f"RC={nfxp.params_['RC']:.4f}")
    print(f"  Neural projected: theta_c={sieve['theta'][0]:.6f}, "
          f"RC={abs(sieve['theta'][1]):.4f} (R^2={sieve['r_squared']:.4f})")
    print()
    if sieve["r_squared"] > 0.95:
        print("  R-squared above 0.95: the neural reward is well-explained by")
        print("  the linear basis. Sieve coefficients are trustworthy.")
    else:
        print("  R-squared below 0.95: the neural reward captures nonlinearity")
        print("  that the linear basis misses. Sieve coefficients approximate")
        print("  but do not fully represent the neural reward surface.")

    # ==================================================================
    #  Summary
    # ==================================================================
    banner("SUMMARY")
    print("This script demonstrated seven neural counterfactual types on")
    print("the Rust bus engine problem. The neural reward from NeuralGLADIUS")
    print("can be perturbed, re-solved, projected, and differentiated without")
    print("requiring any parametric assumptions about the reward function.")
    print()
    print("The structural (NFXP) and neural (GLADIUS) estimators produce")
    print("complementary views of the same replacement behavior. The sieve")
    print("compression bridges the two by projecting the neural surface onto")
    print("the structural feature basis.")


if __name__ == "__main__":
    main()
