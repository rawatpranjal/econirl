#!/usr/bin/env python3
"""NNES Primer — auto-generate results for the LaTeX document.

Demonstrates NNES's core advantage: valid standard errors from the
block-diagonal information matrix (Neyman orthogonality), despite
using a neural network to approximate the value function. On the
Rust (1987) bus engine, NNES matches NFXP's log-likelihood and
produces comparable standard errors without bias correction.

This replicates the Monte Carlo experiment from Nguyen (2025,
Section 6): NNES achieves the same precision as oracle NFXP.

Usage:
    python papers/econirl_package/primers/nnes/nnes_run.py
"""

import json
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "nnes_results.tex"

# ---------- DGP: Rust bus (standard benchmark) ----------
N_STATES = 90
DISCOUNT = 0.99
SEED = 42
N_BUSES = 200
N_PERIODS = 100


def main():
    import jax.numpy as jnp
    from econirl.environments.rust_bus import RustBusEnvironment
    from econirl.estimation.nfxp import NFXPEstimator
    from econirl.estimation.nnes import NNESConfig, NNESEstimator
    from econirl.preferences.linear import LinearUtility
    from econirl.simulation.synthetic import simulate_panel
    from econirl.core.types import DDCProblem

    env = RustBusEnvironment(
        num_mileage_bins=N_STATES, discount_factor=DISCOUNT, seed=SEED,
    )
    n_obs = N_BUSES * N_PERIODS
    true_params = env.get_true_parameter_vector()
    problem = env.problem_spec
    transitions = env.transition_matrices
    utility = LinearUtility.from_environment(env)

    # NNES needs a state encoder
    problem_nnes = DDCProblem(
        num_states=N_STATES,
        num_actions=problem.num_actions,
        discount_factor=DISCOUNT,
        scale_parameter=1.0,
        state_dim=1,
        state_encoder=lambda s: jnp.expand_dims(
            jnp.asarray(s, dtype=jnp.float32) / max(N_STATES - 1, 1),
            axis=-1,
        ),
    )

    print("NNES Primer — generating results")
    print(f"  Environment: Rust bus, {N_STATES} bins, beta={DISCOUNT}")
    print(f"  Data: {N_BUSES} x {N_PERIODS} = {n_obs:,} obs")

    panel = simulate_panel(
        env, n_individuals=N_BUSES, n_periods=N_PERIODS, seed=SEED + 1000,
    )

    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import policy_iteration as pi_solve

    # True policy
    operator = SoftBellmanOperator(problem, jnp.asarray(transitions, dtype=jnp.float64))
    true_reward = jnp.asarray(utility.compute(jnp.asarray(true_params, dtype=jnp.float32)), dtype=jnp.float64)
    true_result = pi_solve(operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix")
    true_actions = np.argmax(np.array(true_result.policy), axis=1)

    results = {}

    # -- BC baseline --
    print("\n  Running BC...")
    t0 = time.time()
    bc = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    bc_res = bc.estimate(panel, utility, problem, jnp.asarray(transitions))
    bc_time = time.time() - t0
    bc_acc = float(np.mean(np.argmax(np.array(bc_res.policy), axis=1) == true_actions))
    results["bc"] = {"ll": float(bc_res.log_likelihood), "acc": bc_acc, "time": bc_time}
    print(f"    LL={results['bc']['ll']:.2f}, acc={bc_acc:.2%}")

    # -- NFXP oracle --
    print("\n  Running NFXP...")
    nfxp = NFXPEstimator(
        inner_solver="hybrid", inner_tol=1e-12, inner_max_iter=100000,
        switch_tol=1e-3, outer_max_iter=200,
        compute_hessian=True, verbose=False,
    )
    t0 = time.time()
    nfxp_res = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    nfxp_p = np.asarray(nfxp_res.parameters)
    nfxp_se = np.asarray(nfxp_res.standard_errors)
    results["nfxp"] = {
        "params": [float(x) for x in nfxp_p],
        "se": [float(x) for x in nfxp_se],
        "ll": float(nfxp_res.log_likelihood),
        "time": nfxp_time,
    }
    print(f"    theta_c={nfxp_p[0]:.6f} (SE {nfxp_se[0]:.6f}), "
          f"RC={nfxp_p[1]:.4f} (SE {nfxp_se[1]:.4f}), "
          f"LL={nfxp_res.log_likelihood:.1f}, time={nfxp_time:.1f}s")

    # -- NNES (NPL Bellman, low-level API) --
    print("\n  Running NNES (NPL, 10 outer iterations)...")
    nnes_cfg = NNESConfig(
        hidden_dim=32, num_layers=2,
        v_lr=1e-3, v_epochs=500,
        n_outer_iterations=20,
        compute_se=True, se_method="asymptotic",
        verbose=False,
    )
    nnes_est = NNESEstimator(config=nnes_cfg)
    t0 = time.time()
    nnes_res = nnes_est.estimate(panel, utility, problem_nnes, transitions)
    nnes_time = time.time() - t0
    nnes_p = np.asarray(nnes_res.parameters)
    nnes_se = np.asarray(nnes_res.standard_errors)
    results["nnes"] = {
        "params": [float(x) for x in nnes_p],
        "se": [float(x) for x in nnes_se],
        "ll": float(nnes_res.log_likelihood),
        "time": nnes_time,
    }
    print(f"    theta_c={nnes_p[0]:.6f} (SE {nnes_se[0]:.6f}), "
          f"RC={nnes_p[1]:.4f} (SE {nnes_se[1]:.4f}), "
          f"LL={nnes_res.log_likelihood:.1f}, time={nnes_time:.1f}s")

    ll_gap = abs(float(nnes_res.log_likelihood) - float(nfxp_res.log_likelihood))
    print(f"\n  LL gap: {ll_gap:.2f}")

    # -- Write LaTeX macros + table --
    pnames = ["\\theta_c", "RC"]
    tex = []
    tex.append("% Auto-generated by nnes_run.py — do not edit by hand")
    tex.append(f"% Rust bus {N_STATES} bins, beta={DISCOUNT}, {n_obs} obs, seed={SEED}")
    tex.append("")
    tex.append(f"\\newcommand{{\\nnesNstates}}{{{N_STATES}}}")
    tex.append(f"\\newcommand{{\\nnesBeta}}{{{DISCOUNT}}}")
    tex.append(f"\\newcommand{{\\nnesNobs}}{{{n_obs:,}}}")
    tex.append(f"\\newcommand{{\\nnesLLgap}}{{{ll_gap:.2f}}}")
    tex.append(f"\\newcommand{{\\nnesNfxpTime}}{{{nfxp_time:.1f}}}")
    tex.append(f"\\newcommand{{\\nnesTime}}{{{nnes_time:.1f}}}")
    tex.append("")

    tex.append("\\begin{table}[h!]")
    tex.append("\\centering\\small")
    tex.append(f"\\caption{{Rust bus engine (\\nnesNstates\\ bins, "
               f"$\\beta=\\nnesBeta$, \\nnesNobs\\ obs). "
               f"NNES uses a 2-layer ReLU network (32 hidden), 10 outer iterations.}}")
    tex.append("\\label{tab:nnes_results}")
    tex.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r}")
    tex.append("\\toprule")
    tex.append("& True & NFXP & NNES \\\\")
    tex.append("\\midrule")
    true_vals = [0.001, 3.0]
    for i, (name, true_val) in enumerate(zip(pnames, true_vals)):
        nfxp_val = float(nfxp_p[i])
        nnes_val = float(nnes_p[i])
        nfxp_se_val = float(nfxp_se[i])
        nnes_se_val = float(nnes_se[i])
        tex.append(f"${name}$ & {true_val:.4f} & {nfxp_val:.6f} & {nnes_val:.6f} \\\\")
        se_nfxp = f"{nfxp_se_val:.6f}" if not np.isnan(nfxp_se_val) else "---"
        se_nnes = f"{nnes_se_val:.6f}" if not np.isnan(nnes_se_val) else "---"
        tex.append(f"\\quad SE & --- & {se_nfxp} & {se_nnes} \\\\")
    tex.append(f"Log-lik & --- & ${float(nfxp_res.log_likelihood):.1f}$ "
               f"& ${float(nnes_res.log_likelihood):.1f}$ \\\\")
    tex.append(f"Time (s) & --- & {nfxp_time:.1f} & {nnes_time:.1f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\end{tabular*}")
    tex.append("\\end{table}")

    OUT.write_text("\n".join(tex) + "\n")
    print(f"\n  Wrote {OUT}")
    OUT.with_suffix(".json").write_text(json.dumps(results, indent=2))
    print(f"  Wrote {OUT.with_suffix('.json')}")


if __name__ == "__main__":
    main()
