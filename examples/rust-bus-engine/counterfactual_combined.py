#!/usr/bin/env python3
"""Neural vs structural counterfactuals on simulated bus data.

Compares three cases:
  (a) NFXP on linear-sim data (structural reference, correct spec)
  (b) NeuralGLADIUS on linear-sim data (should match NFXP if it works)
  (c) NeuralGLADIUS on nonlinear-sim data (can capture the kink)

Linear DGP: u(s,keep) = -0.001*s, u(s,replace) = -3.0
Nonlinear DGP: u(s,keep) = -0.001*s - 2.0*I(s>60), u(s,replace) = -3.0
The kink at s=60 means buses above 300K miles have a sudden jump
in operating costs. NFXP with linear features cannot capture this.
NeuralGLADIUS can.

We simulate 1000 buses for each DGP (100K observations) so sample
size is not a bottleneck.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

from econirl import NFXP
from econirl.estimators.neural_gladius import NeuralGLADIUS
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.simulation.synthetic import simulate_panel
from econirl.core.types import DDCProblem
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import value_iteration
from econirl.simulation import (
    neural_local_perturbation,
    neural_transition_counterfactual,
    neural_choice_set_counterfactual,
    neural_sieve_compression,
    neural_policy_jacobian,
    neural_perturbation_sweep,
)

OUT = Path(__file__).resolve().parent.parent.parent / "docs" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN, GRAY = "#1f77b4", "#ff7f0e", "#2ca02c", "#888888"
N_BUSES = 1000
N_PERIODS = 100


def banner(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")


def fit_gladius(df, state_col, action_col, id_col, label):
    print(f"Fitting NeuralGLADIUS ({label}) ...")
    g = NeuralGLADIUS(
        n_actions=2, discount=0.9999,
        q_hidden_dim=64, q_num_layers=2,
        ev_hidden_dim=64, ev_num_layers=2,
        max_epochs=500, batch_size=512, patience=100,
        bellman_weight=1.0, alternating_updates=True, verbose=True)
    g.fit(df, state=state_col, action=action_col, id=id_col)
    g._n_states = 90
    print(f"  Epochs: {g.n_epochs_}")
    return g


def simulate_nonlinear_bus(n_buses, n_periods, seed=42):
    """Simulate from a nonlinear DGP with a kink at s=60.

    u(s, keep) = -0.001*s - 2.0 * I(s > 60)
    u(s, replace) = -3.0

    The kink means high-mileage buses face a sudden cost jump.
    We build the reward matrix manually and simulate from the
    optimal policy under this reward.
    """
    env = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0,
        num_mileage_bins=90, discount_factor=0.9999, seed=seed)
    prob = env.problem_spec
    trans = jnp.array(env.transition_matrices)

    # Build nonlinear reward
    reward = jnp.zeros((90, 2))
    mileage = jnp.arange(90, dtype=jnp.float32)
    reward = reward.at[:, 0].set(-0.001 * mileage - 2.0 * (mileage > 60).astype(jnp.float32))
    reward = reward.at[:, 1].set(-3.0)

    # Solve for optimal policy
    op = SoftBellmanOperator(prob, trans)
    sol = value_iteration(op, reward)

    # Simulate using the policy — convert to float64 numpy to avoid
    # probability sum-to-1 errors from float32 rounding
    from econirl.simulation.synthetic import simulate_panel_from_policy
    # Use the environment's simulate_panel which handles normalization
    # internally, by temporarily overriding the reward
    # Simpler approach: use the environment directly with a custom policy
    from econirl.simulation.synthetic import simulate_panel as _sim
    # Build a temporary env that produces our nonlinear policy
    # Trick: just use a simple loop
    import pandas as pd
    rng = np.random.default_rng(seed)
    policy_np = np.asarray(sol.policy, dtype=np.float64)
    policy_np = policy_np / policy_np.sum(axis=1, keepdims=True)
    trans_np = np.asarray(trans, dtype=np.float64)
    trans_np = trans_np / trans_np.sum(axis=2, keepdims=True)
    rows = []
    for i in range(n_buses):
        s = rng.choice(90, p=np.ones(90)/90)
        for t in range(n_periods):
            a = rng.choice(2, p=policy_np[s])
            rows.append({"id": i, "period": t, "mileage_bin": int(s), "replaced": int(a)})
            s = rng.choice(90, p=trans_np[a, s])
    panel_df = pd.DataFrame(rows)

    return panel_df, reward, sol.policy, env


def main():
    prob = DDCProblem(num_states=90, num_actions=2,
                      discount_factor=0.9999, scale_parameter=1.0)
    states = np.arange(90)

    # ── 1. Simulate and fit ──────────────────────────────────────────
    banner("1. SIMULATE DATA AND FIT MODELS")

    # Linear DGP
    env_lin = RustBusEnvironment(
        operating_cost=0.001, replacement_cost=3.0,
        num_mileage_bins=90, discount_factor=0.9999, seed=42)
    panel_lin = simulate_panel(env_lin, n_individuals=N_BUSES, n_periods=N_PERIODS, seed=42)
    df_lin = panel_lin.to_dataframe().rename(
        columns={"state": "mileage_bin", "action": "replaced"})
    print(f"Linear DGP: {len(df_lin):,} obs, {N_BUSES} buses, "
          f"repl rate {df_lin['replaced'].mean():.4f}")

    # True linear reward and policy
    true_lin_params = jnp.array([0.001, 3.0])
    true_lin_r = jnp.einsum("sak,k->sa", jnp.array(env_lin.feature_matrix), true_lin_params)
    trans = jnp.array(env_lin.transition_matrices)
    op = SoftBellmanOperator(prob, trans)
    true_lin_pi = value_iteration(op, true_lin_r).policy

    # Nonlinear DGP
    df_nl, true_nl_r, true_nl_pi, env_nl = simulate_nonlinear_bus(N_BUSES, N_PERIODS, seed=99)
    print(f"Nonlinear DGP: {len(df_nl):,} obs, {N_BUSES} buses, "
          f"repl rate {df_nl['replaced'].mean():.4f}")
    print()

    # (a) NFXP on linear data (correct specification)
    nfxp_lin = NFXP(n_states=90, n_actions=2, discount=0.9999, verbose=False)
    nfxp_lin.fit(df_lin, state="mileage_bin", action="replaced", id="id")
    nfxp_r = jnp.array(nfxp_lin.reward_matrix_)
    print(f"NFXP (linear):     theta_c={nfxp_lin.params_['theta_c']:.6f}, "
          f"RC={nfxp_lin.params_['RC']:.4f}")

    # NFXP on nonlinear data (misspecified)
    nfxp_nl = NFXP(n_states=90, n_actions=2, discount=0.9999, verbose=False)
    nfxp_nl.fit(df_nl, state="mileage_bin", action="replaced", id="id")
    nfxp_nl_r = jnp.array(nfxp_nl.reward_matrix_)
    print(f"NFXP (nonlinear):  theta_c={nfxp_nl.params_['theta_c']:.6f}, "
          f"RC={nfxp_nl.params_['RC']:.4f}")
    print()

    # Build transition tensor for counterfactuals
    keep_t = nfxp_lin.transitions_
    cf_trans = np.zeros((2, 90, 90))
    cf_trans[0] = keep_t
    for s in range(90):
        cf_trans[1, s, :] = keep_t[0, :]
    cf_trans = jnp.array(cf_trans)

    # (b) GLADIUS on linear data
    g_lin = fit_gladius(df_lin, "mileage_bin", "replaced", "id", "linear DGP")
    g_lin_r = jnp.array(g_lin.reward_matrix_)
    print()

    # (c) GLADIUS on nonlinear data
    g_nl = fit_gladius(df_nl, "mileage_bin", "replaced", "id", "nonlinear DGP")
    g_nl_r = jnp.array(g_nl.reward_matrix_)

    # Solve policies from neural rewards
    op2 = SoftBellmanOperator(prob, cf_trans)
    pi_nfxp_lin = value_iteration(op2, nfxp_r).policy
    pi_nfxp_nl = value_iteration(op2, nfxp_nl_r).policy
    pi_g_lin = value_iteration(op2, g_lin_r).policy
    pi_g_nl = value_iteration(op2, g_nl_r).policy

    # ── 2. Reward heatmap ────────────────────────────────────────────
    banner("2. REWARD COMPARISON")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    # Row 1: linear DGP
    for ax, d, title in zip(axes[0], [
        np.asarray(true_lin_r), np.asarray(nfxp_r), np.asarray(g_lin_r)
    ], ["True (linear)", "NFXP (linear)", "GLADIUS (linear)"]):
        im = ax.imshow(d, aspect="auto", cmap="viridis")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Keep", "Replace"])
        ax.set_ylabel("Mileage bin"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)
    # Row 2: nonlinear DGP
    for ax, d, title in zip(axes[1], [
        np.asarray(true_nl_r), np.asarray(nfxp_nl_r), np.asarray(g_nl_r)
    ], ["True (nonlinear)", "NFXP (nonlinear, misspec)", "GLADIUS (nonlinear)"]):
        im = ax.imshow(d, aspect="auto", cmap="viridis")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Keep", "Replace"])
        ax.set_ylabel("Mileage bin"); ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, shrink=0.7)
    fig.suptitle("Reward surfaces: linear DGP (top) vs nonlinear DGP (bottom)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_reward_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cf_bus_reward_heatmap.png")

    # ── 3. Policy comparison ─────────────────────────────────────────
    banner("3. POLICY COMPARISON")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Linear DGP
    ax1.plot(states, np.asarray(true_lin_pi[:, 1]), '--', color=GRAY, lw=2, label="True")
    ax1.plot(states, np.asarray(pi_nfxp_lin[:, 1]), color=BLUE, lw=1.5, label="NFXP")
    ax1.plot(states, np.asarray(pi_g_lin[:, 1]), color=ORANGE, lw=1.5, label="GLADIUS")
    ax1.set_xlabel("Mileage bin"); ax1.set_ylabel("P(replace)")
    ax1.set_title("Linear DGP (correct NFXP spec)")
    ax1.legend()

    # Nonlinear DGP
    ax2.plot(states, np.asarray(true_nl_pi[:, 1]), '--', color=GRAY, lw=2, label="True")
    ax2.plot(states, np.asarray(pi_nfxp_nl[:, 1]), color=BLUE, lw=1.5, label="NFXP (misspec)")
    ax2.plot(states, np.asarray(pi_g_nl[:, 1]), color=GREEN, lw=1.5, label="GLADIUS")
    ax2.set_xlabel("Mileage bin")
    ax2.set_title("Nonlinear DGP (kink at s=60)")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_policy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    for label, pi, true_pi_ref in [
        ("NFXP linear", pi_nfxp_lin, true_lin_pi),
        ("GLADIUS linear", pi_g_lin, true_lin_pi),
        ("NFXP nonlinear", pi_nfxp_nl, true_nl_pi),
        ("GLADIUS nonlinear", pi_g_nl, true_nl_pi),
    ]:
        c = float(np.corrcoef(np.asarray(pi[:, 1]), np.asarray(true_pi_ref[:, 1]))[0, 1])
        m = float(np.abs(np.asarray(pi[:, 1]) - np.asarray(true_pi_ref[:, 1])).mean())
        print(f"  {label:20s}: corr={c:.4f}, MAE={m:.4f}")
    print("Saved cf_bus_policy.png")

    # ── 4. Global perturbation sweep ─────────────────────────────────
    banner("4. GLOBAL PERTURBATION SWEEP")
    deltas = jnp.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])

    # Compute sweeps for all models
    sw = {}
    for label, r in [("nfxp_lin", nfxp_r), ("g_lin", g_lin_r),
                      ("nfxp_nl", nfxp_nl_r), ("g_nl", g_nl_r)]:
        sw[label] = neural_perturbation_sweep(r, 1, deltas, prob, cf_trans)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.set_title("Linear DGP")
    ax1.plot(np.asarray(deltas), sw["nfxp_lin"]['mean_action_prob'], 'o-', color=BLUE, label="NFXP")
    ax1.plot(np.asarray(deltas), sw["g_lin"]['mean_action_prob'], 's-', color=ORANGE, label="GLADIUS")
    ax1.set_xlabel("Replacement penalty"); ax1.set_ylabel("Mean P(replace)")
    ax1.legend()

    ax2.set_title("Nonlinear DGP")
    ax2.plot(np.asarray(deltas), sw["nfxp_nl"]['mean_action_prob'], 'o-', color=BLUE, label="NFXP (misspec)")
    ax2.plot(np.asarray(deltas), sw["g_nl"]['mean_action_prob'], '^-', color=GREEN, label="GLADIUS")
    ax2.set_xlabel("Replacement penalty"); ax2.set_ylabel("Mean P(replace)")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_global_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cf_bus_global_sweep.png")

    # ── 5. Local perturbation ────────────────────────────────────────
    banner("5. LOCAL PERTURBATION (states > 60)")
    mask = jnp.arange(90) > 60
    affected = jnp.where(mask)[0]
    loc_d = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
    loc_res = {k: [] for k in ["nfxp_lin", "g_lin", "nfxp_nl", "g_nl"]}
    for d in loc_d:
        for label, r in [("nfxp_lin", nfxp_r), ("g_lin", g_lin_r),
                          ("nfxp_nl", nfxp_nl_r), ("g_nl", g_nl_r)]:
            cf = neural_local_perturbation(r, 0, d, mask, prob, cf_trans)
            loc_res[label].append(float(cf.counterfactual_policy[affected, 1].mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(loc_d, loc_res["nfxp_lin"], 'o-', color=BLUE, label="NFXP")
    ax1.plot(loc_d, loc_res["g_lin"], 's-', color=ORANGE, label="GLADIUS")
    ax1.set_title("Linear DGP"); ax1.set_xlabel("Penalty at s>60"); ax1.set_ylabel("P(replace|s>60)")
    ax1.legend()
    ax2.plot(loc_d, loc_res["nfxp_nl"], 'o-', color=BLUE, label="NFXP (misspec)")
    ax2.plot(loc_d, loc_res["g_nl"], '^-', color=GREEN, label="GLADIUS")
    ax2.set_title("Nonlinear DGP"); ax2.set_xlabel("Penalty at s>60")
    ax2.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_local.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cf_bus_local.png")

    # ── 6. Transition counterfactual ─────────────────────────────────
    banner("6. TRANSITION COUNTERFACTUAL")
    new_trans = jnp.zeros_like(cf_trans)
    for a in range(2):
        for s in range(90):
            row = jnp.zeros(90)
            base = 0 if a == 1 else s
            for i, p in enumerate([0.20, 0.50, 0.30]):
                row = row.at[min(base + i, 89)].add(p)
            new_trans = new_trans.at[a, s, :].set(row)

    t_cfs = {}
    for label, r in [("nfxp_lin", nfxp_r), ("g_lin", g_lin_r),
                      ("nfxp_nl", nfxp_nl_r), ("g_nl", g_nl_r)]:
        t_cfs[label] = neural_transition_counterfactual(r, new_trans, prob, cf_trans)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, pairs, title in [
        (ax1, [("nfxp_lin", BLUE, "NFXP"), ("g_lin", ORANGE, "GLADIUS")], "Linear DGP"),
        (ax2, [("nfxp_nl", BLUE, "NFXP (misspec)"), ("g_nl", GREEN, "GLADIUS")], "Nonlinear DGP"),
    ]:
        for label, color, name in pairs:
            ax.plot(states, np.asarray(t_cfs[label].baseline_policy[:, 1]),
                    color=color, lw=1.5, label=f"{name} base")
            ax.plot(states, np.asarray(t_cfs[label].counterfactual_policy[:, 1]),
                    '--', color=color, lw=1.5, label=f"{name} CF")
        ax.set_xlabel("Mileage bin"); ax.set_ylabel("P(replace)")
        ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_transition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    for k in t_cfs:
        print(f"  {k}: welfare change = {t_cfs[k].welfare_change:.2f}")
    print("Saved cf_bus_transition.png")

    # ── 7. Choice set counterfactual ─────────────────────────────────
    banner("7. CHOICE SET COUNTERFACTUAL")
    mask_cs = jnp.ones((90, 2), dtype=jnp.bool_)
    mask_cs = mask_cs.at[81:, 0].set(False)
    mask_cs = mask_cs.at[:10, 1].set(False)

    cs_cfs = {}
    for label, r in [("nfxp_lin", nfxp_r), ("g_lin", g_lin_r),
                      ("nfxp_nl", nfxp_nl_r), ("g_nl", g_nl_r)]:
        cs_cfs[label] = neural_choice_set_counterfactual(r, mask_cs, prob, cf_trans)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    for ax, (label, name) in zip(axes.flat, [
        ("nfxp_lin", "NFXP (linear)"), ("g_lin", "GLADIUS (linear)"),
        ("nfxp_nl", "NFXP (nonlinear, misspec)"), ("g_nl", "GLADIUS (nonlinear)"),
    ]):
        res = cs_cfs[label]
        ax.plot(states, np.asarray(res.baseline_policy[:, 1]), lw=1.5, label="Baseline")
        ax.plot(states, np.asarray(res.counterfactual_policy[:, 1]), '--', lw=1.5, label="Constrained")
        ax.axvspan(0, 9, alpha=0.1, color="red")
        ax.axvspan(81, 89, alpha=0.1, color="green")
        ax.set_xlabel("Mileage bin"); ax.set_title(name); ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("P(replace)"); axes[1, 0].set_ylabel("P(replace)")
    fig.suptitle("Choice set: warranty (s<10) + mandatory (s>80)")
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_choice_set.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cf_bus_choice_set.png")

    # ── 8. Sieve compression ────────────────────────────────────────
    banner("8. SIEVE COMPRESSION")
    mileage = jnp.arange(90, dtype=jnp.float32)
    features = jnp.zeros((90, 2, 2))
    features = features.at[:, 0, 0].set(-mileage)
    features = features.at[:, 1, 1].set(-1.0)

    sieves = {}
    for label, r in [("g_lin", g_lin_r), ("g_nl", g_nl_r)]:
        sieves[label] = neural_sieve_compression(r, features, ["theta_c", "RC"])

    print(f"  {'':>20s}  {'theta_c':>10s}  {'RC':>8s}  {'R²':>6s}")
    print(f"  {'True (linear)':>20s}  {'0.001000':>10s}  {'3.0000':>8s}  {'—':>6s}")
    print(f"  {'NFXP (linear)':>20s}  {nfxp_lin.params_['theta_c']:10.6f}  "
          f"{nfxp_lin.params_['RC']:8.4f}  {'—':>6s}")
    print(f"  {'GLADIUS (linear)':>20s}  {sieves['g_lin']['theta'][0]:10.6f}  "
          f"{abs(sieves['g_lin']['theta'][1]):8.4f}  {sieves['g_lin']['r_squared']:6.4f}")
    print(f"  {'GLADIUS (nonlinear)':>20s}  {sieves['g_nl']['theta'][0]:10.6f}  "
          f"{abs(sieves['g_nl']['theta'][1]):8.4f}  {sieves['g_nl']['r_squared']:6.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    struct_diff_lin = np.asarray(nfxp_r[:, 1] - nfxp_r[:, 0])
    for ax, r, sieve, title in [
        (ax1, g_lin_r, sieves["g_lin"],
         f"GLADIUS linear (R²={sieves['g_lin']['r_squared']:.2f})"),
        (ax2, g_nl_r, sieves["g_nl"],
         f"GLADIUS nonlinear (R²={sieves['g_nl']['r_squared']:.2f})"),
    ]:
        nd = np.asarray(r[:, 1] - r[:, 0])
        ax.scatter(struct_diff_lin, nd, alpha=0.6, edgecolors="k", lw=0.5)
        lims = [min(struct_diff_lin.min(), nd.min()), max(struct_diff_lin.max(), nd.max())]
        ax.plot(lims, lims, '--', color=GRAY)
        ax.set_xlabel("NFXP r(replace) - r(keep)")
        ax.set_ylabel("Neural r(replace) - r(keep)")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_sieve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cf_bus_sieve.png")

    # ── 9. Policy Jacobian (nonlinear GLADIUS) ───────────────────────
    banner("9. POLICY JACOBIAN (nonlinear GLADIUS)")
    print("Computing Jacobian (180 Bellman solves) ...")
    J = neural_policy_jacobian(g_nl_r, prob, cf_trans, target_action=1)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.asarray(J[:, :, 1]), aspect="auto", cmap="RdBu_r")
    ax.set_xlabel("Perturbed state s'"); ax.set_ylabel("Affected state s")
    ax.set_title("Policy Jacobian: ∂P(replace|s) / ∂r(s', replace)")
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / "cf_bus_jacobian.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved cf_bus_jacobian.png")

    # ── Summary ─────────────────────────────────────────────────────
    banner("SUMMARY")
    print(f"  Linear DGP:    GLADIUS sieve R²={sieves['g_lin']['r_squared']:.4f}")
    print(f"  Nonlinear DGP: GLADIUS sieve R²={sieves['g_nl']['r_squared']:.4f}")
    print()
    if sieves["g_nl"]["r_squared"] < sieves["g_lin"]["r_squared"] - 0.05:
        print("  The nonlinear R² is lower, confirming GLADIUS captured the kink.")
        print("  The linear basis misses the cost jump at s=60.")
    else:
        print("  Both R² are similar, suggesting GLADIUS hasn't captured the kink.")


if __name__ == "__main__":
    main()
