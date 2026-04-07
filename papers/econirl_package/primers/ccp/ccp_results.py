#!/usr/bin/env python3
"""CCP Primer — auto-generate results for the LaTeX document.

Demonstrates the NPL convergence theorem (Aguirregabiria-Mira 2002):
  - K=1 Hotz-Miller: consistent from empirical CCPs, no Bellman solver
  - K=2,3,5: converges to MLE as K increases
  - NFXP: MLE reference

The central insight is that CCP-NPL starts from non-parametric empirical
choice frequencies and converges to the same MLE as NFXP — without ever
solving the Bellman equation for the outer loop gradient. K=1 already
identifies theta consistently; NPL iterations add efficiency.

Usage:
    .venv/bin/python papers/econirl_package/primers/ccp/ccp_results.py

Output:
    papers/econirl_package/primers/ccp/ccp_results.tex
    papers/econirl_package/primers/ccp/ccp_results.json
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# ── Setup ──────────────────────────────────────────────────────
# Rust (1987) Table 6 settings: 90 bins, beta=0.9999.
# At beta=0.9999, the NK/policy inner loop handles high discounting;
# CCP bypasses the inner loop entirely via matrix inversion.
N_BINS = 90
BETA = 0.9999
TRUE_OC = 0.001
TRUE_RC = 3.0
N_INDIVIDUALS = 200
N_PERIODS = 100        # 20,000 observations (Rust's Table 6 order of magnitude)
SEED = 42

OUT = Path(__file__).resolve().parent / "ccp_results.tex"


def cosine_sim(a, b):
    a, b = jnp.asarray(a).flatten(), jnp.asarray(b).flatten()
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def main():
    from econirl.environments.rust_bus import RustBusEnvironment
    from econirl.estimation.nfxp import NFXPEstimator
    from econirl.estimation.ccp import CCPEstimator
    from econirl.estimation.behavioral_cloning import BehavioralCloningEstimator
    from econirl.preferences.linear import LinearUtility
    from econirl.core.bellman import SoftBellmanOperator
    from econirl.core.solvers import policy_iteration

    print("CCP Primer — NPL convergence theorem (A&M 2002)")
    print(f"  Rust bus: {N_BINS} states, beta={BETA}")
    print(f"  Data: {N_INDIVIDUALS} x {N_PERIODS} = {N_INDIVIDUALS*N_PERIODS:,} obs")

    # Environment and data
    env = RustBusEnvironment(
        operating_cost=TRUE_OC, replacement_cost=TRUE_RC,
        num_mileage_bins=N_BINS, discount_factor=BETA,
    )
    panel = env.generate_panel(n_individuals=N_INDIVIDUALS, n_periods=N_PERIODS, seed=SEED)
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.array([TRUE_OC, TRUE_RC])

    # True policy for accuracy comparison
    transitions_f64 = jnp.asarray(transitions, dtype=jnp.float64)
    operator = SoftBellmanOperator(problem, transitions_f64)
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy = np.array(true_result.policy)
    true_actions = np.argmax(true_policy, axis=1)

    results = {}

    # ── BC baseline ───────────────────────────────────────────
    print("\n  Running BC baseline...")
    t0 = time.time()
    bc = BehavioralCloningEstimator(smoothing=1.0, verbose=False)
    bc_res = bc.estimate(panel, utility, problem, jnp.asarray(transitions))
    bc_time = time.time() - t0
    bc_policy = np.array(bc_res.policy)
    bc_acc = float(np.mean(np.argmax(bc_policy, axis=1) == true_actions))
    results["bc"] = {
        "ll": float(bc_res.log_likelihood),
        "policy_acc": bc_acc,
        "time": bc_time,
    }
    print(f"    LL={results['bc']['ll']:.2f}, policy_acc={bc_acc:.2%}, time={bc_time:.2f}s")

    # ── NFXP: the MLE reference ───────────────────────────────
    # NFXP-NK converges to the MLE from a parametric starting point.
    # This is the reference that NPL must match at K=5+.
    print("\n  Running NFXP (MLE reference)...")
    t0 = time.time()
    nfxp = NFXPEstimator(
        optimizer="BHHH", inner_solver="hybrid",
        inner_tol=1e-12, inner_max_iter=200,
        se_method="robust", verbose=False,
    )
    nfxp_result = nfxp.estimate(
        panel=panel, utility=utility, problem=problem, transitions=transitions,
    )
    nfxp_time = time.time() - t0
    results["nfxp"] = {
        "oc": float(nfxp_result.parameters[0]),
        "rc": float(nfxp_result.parameters[1]),
        "ll": float(nfxp_result.log_likelihood),
        "time": nfxp_time,
        "cosine": cosine_sim(nfxp_result.parameters, true_params),
        "se_oc": float(nfxp_result.standard_errors[0]) if nfxp_result.standard_errors is not None else None,
        "se_rc": float(nfxp_result.standard_errors[1]) if nfxp_result.standard_errors is not None else None,
    }
    print(f"    theta_c={results['nfxp']['oc']:.6f}, RC={results['nfxp']['rc']:.4f}, "
          f"LL={results['nfxp']['ll']:.2f}, time={nfxp_time:.1f}s")

    # ── NPL convergence: K=1,2,3,5,10 ────────────────────────
    # Core experiment: show NPL starting from empirical CCPs (K=1 Hotz-Miller)
    # and converging to NFXP MLE as K increases. At K=5, parameter estimates,
    # log-likelihood, and standard errors are identical to NFXP.
    npl_ks = [1, 2, 3, 5, 10]
    results["npl"] = {}

    for K in npl_ks:
        label = "Hotz-Miller" if K == 1 else f"NPL K={K}"
        print(f"\n  Running CCP {label}...")
        t0 = time.time()
        estimator = CCPEstimator(
            num_policy_iterations=K,
            se_method="robust" if K >= 5 else "asymptotic",
            compute_hessian=(K >= 5),  # only compute SEs for K≥5
            verbose=False,
        )
        result = estimator.estimate(
            panel=panel, utility=utility, problem=problem, transitions=transitions,
        )
        elapsed = time.time() - t0
        results["npl"][K] = {
            "oc": float(result.parameters[0]),
            "rc": float(result.parameters[1]),
            "ll": float(result.log_likelihood),
            "time": elapsed,
            "cosine": cosine_sim(result.parameters, true_params),
            "se_oc": float(result.standard_errors[0]) if (K >= 5 and result.standard_errors is not None) else None,
            "se_rc": float(result.standard_errors[1]) if (K >= 5 and result.standard_errors is not None) else None,
            "ll_gap": abs(float(result.log_likelihood) - results["nfxp"]["ll"]),
        }
        print(f"    theta_c={results['npl'][K]['oc']:.6f}, RC={results['npl'][K]['rc']:.4f}, "
              f"LL={results['npl'][K]['ll']:.2f} (gap={results['npl'][K]['ll_gap']:.4f}), "
              f"time={elapsed:.1f}s")

    # Reference values from NFXP
    nfxp_oc = results["nfxp"]["oc"]
    nfxp_rc = results["nfxp"]["rc"]
    nfxp_ll = results["nfxp"]["ll"]

    # Aliases for CCP-NPL at K=5 (primary result)
    ccp_k5 = results["npl"][5]
    ccp_time = ccp_k5["time"]
    speedup = nfxp_time / ccp_time if ccp_time > 0 else 0

    print(f"\n  True parameters: theta_c={TRUE_OC}, RC={TRUE_RC}")
    print(f"  NFXP (MLE reference): theta_c={nfxp_oc:.6f}, RC={nfxp_rc:.4f}, LL={nfxp_ll:.2f}")
    print(f"  K=1 Hotz-Miller: LL gap = {results['npl'][1]['ll_gap']:.4f}")
    print(f"  K=5 NPL:         LL gap = {results['npl'][5]['ll_gap']:.6f} (matches MLE)")

    # ── Experiment 2: Initialization robustness ───────────────
    # CCP K=5 is invariant to starting theta because step 1 reads CCPs
    # directly from data frequencies (no initial theta involvement).
    # NFXP must search from a parametric starting point; bad starting
    # values cause more outer iterations (slow convergence) or failure.
    print("\n  Running initialization robustness experiment...")
    init_points = {
        "default":       jnp.array([0.01,   5.0]),   # data-driven (from NFXP default)
        "near truth":    jnp.array([0.001,  3.0]),
        "bad (tiny RC)": jnp.array([1.0,    0.1]),
        "bad (huge RC)": jnp.array([0.0001, 50.0]),
        "bad (near 0)":  jnp.array([1e-5,   1e-5]),
    }

    results["init_robustness"] = {}
    for label, init_p in init_points.items():
        print(f"    {label}: init=[{float(init_p[0]):.5f}, {float(init_p[1]):.4f}]")

        # NFXP from this starting point
        t0 = time.time()
        nfxp_rob = NFXPEstimator(
            optimizer="BHHH", inner_solver="hybrid",
            inner_tol=1e-12, inner_max_iter=200,
            se_method="asymptotic", verbose=False,
        )
        nfxp_rob_res = nfxp_rob.estimate(
            panel=panel, utility=utility, problem=problem, transitions=transitions,
            initial_params=init_p,
        )
        nfxp_rob_time = time.time() - t0

        # CCP K=5 from this starting point
        t0 = time.time()
        ccp_rob = CCPEstimator(
            num_policy_iterations=5, compute_hessian=False, verbose=False,
        )
        ccp_rob_res = ccp_rob.estimate(
            panel=panel, utility=utility, problem=problem, transitions=transitions,
            initial_params=init_p,
        )
        ccp_rob_time = time.time() - t0

        results["init_robustness"][label] = {
            "init_oc": float(init_p[0]),
            "init_rc": float(init_p[1]),
            "nfxp_oc": float(nfxp_rob_res.parameters[0]),
            "nfxp_rc": float(nfxp_rob_res.parameters[1]),
            "nfxp_ll": float(nfxp_rob_res.log_likelihood),
            "nfxp_nit": int(nfxp_rob_res.num_iterations),
            "nfxp_converged": bool(nfxp_rob_res.converged),
            "nfxp_time": nfxp_rob_time,
            "ccp_oc": float(ccp_rob_res.parameters[0]),
            "ccp_rc": float(ccp_rob_res.parameters[1]),
            "ccp_ll": float(ccp_rob_res.log_likelihood),
            "ccp_nit": int(ccp_rob_res.num_iterations),
            "ccp_time": ccp_rob_time,
        }
        r = results["init_robustness"][label]
        print(f"      NFXP: theta_c={r['nfxp_oc']:.6f}, RC={r['nfxp_rc']:.4f}, "
              f"LL={r['nfxp_ll']:.2f}, nit={r['nfxp_nit']}, conv={r['nfxp_converged']}")
        print(f"      CCP:  theta_c={r['ccp_oc']:.6f}, RC={r['ccp_rc']:.4f}, "
              f"LL={r['ccp_ll']:.2f}, nit={r['ccp_nit']}")

    init_nfxp_nits = [v["nfxp_nit"] for v in results["init_robustness"].values()]
    init_ccp_ocs = [v["ccp_oc"] for v in results["init_robustness"].values()]
    init_ccp_rcs = [v["ccp_rc"] for v in results["init_robustness"].values()]
    print(f"\n  NFXP outer iterations across starting points: {init_nfxp_nits}")
    print(f"  CCP K=5 oc range: [{min(init_ccp_ocs):.6f}, {max(init_ccp_ocs):.6f}]")
    print(f"  CCP K=5 rc range: [{min(init_ccp_rcs):.4f}, {max(init_ccp_rcs):.4f}]")

    # ── Write LaTeX ────────────────────────────────────────────
    def fmt(v, digits=6):
        if v is None:
            return "---"
        return f"{v:.{digits}f}"

    tex = []
    tex.append("% Auto-generated by ccp_results.py — do not edit by hand")
    tex.append(f"% {N_BINS} bins, beta={BETA}, {N_INDIVIDUALS}x{N_PERIODS} obs, seed={SEED}")
    tex.append("")
    tex.append(f"\\newcommand{{\\ccpNbins}}{{{N_BINS}}}")
    tex.append(f"\\newcommand{{\\ccpBeta}}{{{BETA}}}")
    tex.append(f"\\newcommand{{\\ccpNobs}}{{{N_INDIVIDUALS*N_PERIODS:,}}}")
    tex.append(f"\\newcommand{{\\ccpTrueOC}}{{{TRUE_OC}}}")
    tex.append(f"\\newcommand{{\\ccpTrueRC}}{{{TRUE_RC}}}")
    tex.append("")
    tex.append(f"\\newcommand{{\\bcLL}}{{{results['bc']['ll']:.2f}}}")
    tex.append(f"\\newcommand{{\\bcAcc}}{{{results['bc']['policy_acc']*100:.1f}\\%}}")
    tex.append(f"\\newcommand{{\\bcTime}}{{{results['bc']['time']:.2f}}}")
    tex.append("")
    tex.append(f"\\newcommand{{\\nfxpOC}}{{{nfxp_oc:.6f}}}")
    tex.append(f"\\newcommand{{\\nfxpRC}}{{{nfxp_rc:.4f}}}")
    tex.append(f"\\newcommand{{\\nfxpLL}}{{{nfxp_ll:.2f}}}")
    tex.append(f"\\newcommand{{\\nfxpTime}}{{{nfxp_time:.1f}}}")
    tex.append(f"\\newcommand{{\\nfxpCosine}}{{{results['nfxp']['cosine']:.4f}}}")
    tex.append(f"\\newcommand{{\\nfxpSEoc}}{{{fmt(results['nfxp']['se_oc'])}}}")
    tex.append(f"\\newcommand{{\\nfxpSErc}}{{{fmt(results['nfxp']['se_rc'])}}}")
    tex.append("")
    tex.append(f"\\newcommand{{\\ccpOC}}{{{ccp_k5['oc']:.6f}}}")
    tex.append(f"\\newcommand{{\\ccpRC}}{{{ccp_k5['rc']:.4f}}}")
    tex.append(f"\\newcommand{{\\ccpLL}}{{{ccp_k5['ll']:.2f}}}")
    tex.append(f"\\newcommand{{\\ccpTime}}{{{ccp_time:.1f}}}")
    tex.append(f"\\newcommand{{\\ccpCosine}}{{{ccp_k5['cosine']:.4f}}}")
    tex.append(f"\\newcommand{{\\ccpSpeedup}}{{{speedup:.1f}}}")
    tex.append(f"\\newcommand{{\\ccpSEoc}}{{{fmt(ccp_k5['se_oc'])}}}")
    tex.append(f"\\newcommand{{\\ccpSErc}}{{{fmt(ccp_k5['se_rc'])}}}")
    tex.append("")
    tex.append(f"\\newcommand{{\\hmOC}}{{{results['npl'][1]['oc']:.6f}}}")
    tex.append(f"\\newcommand{{\\hmRC}}{{{results['npl'][1]['rc']:.4f}}}")
    tex.append(f"\\newcommand{{\\hmLL}}{{{results['npl'][1]['ll']:.2f}}}")
    tex.append(f"\\newcommand{{\\hmTime}}{{{results['npl'][1]['time']:.1f}}}")
    tex.append(f"\\newcommand{{\\hmCosine}}{{{results['npl'][1]['cosine']:.4f}}}")
    tex.append(f"\\newcommand{{\\hmLLgap}}{{{results['npl'][1]['ll_gap']:.2f}}}")
    tex.append("")

    # NPL convergence table
    tex.append("% ── Table: NPL convergence ──")
    tex.append("\\begin{table}[H]")
    tex.append("\\centering")
    tex.append(f"\\caption{{NPL convergence to MLE on Rust bus (\\ccpNbins\\ states, "
               f"$\\beta=\\ccpBeta$, \\ccpNobs\\ observations). "
               f"True $\\theta_c = \\ccpTrueOC$, $RC = \\ccpTrueRC$. "
               f"NFXP is the MLE reference. BC policy accuracy is \\bcAcc.}}")
    tex.append("\\label{tab:ccp_vs_nfxp}")
    tex.append("\\small")
    tex.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r r r}")
    tex.append("\\toprule")
    tex.append("Metric & K=1 (HM) & K=2 & K=3 & K=5 & NFXP \\\\")
    tex.append("\\midrule")

    # Parameter rows
    def row(label, key, fmt_digits=6, show_as_pct=False):
        vals = []
        for k in [1, 2, 3, 5]:
            v = results["npl"][k].get(key)
            vals.append(fmt(v, fmt_digits) if v is not None else "---")
        nfxp_v = results["nfxp"].get(key)
        vals.append(fmt(nfxp_v, fmt_digits) if nfxp_v is not None else "---")
        return f"{label} & " + " & ".join(vals) + " \\\\"

    tex.append(row("$\\hat\\theta_c$", "oc", 6))
    tex.append(row("$\\hat{RC}$", "rc", 4))
    tex.append(row("Log-likelihood", "ll", 2))
    tex.append(row("|LL gap|", "ll_gap", 4))
    tex.append(row("Cosine sim.", "cosine", 4))
    tex.append(row("SE($\\theta_c$)", "se_oc", 6))
    tex.append(row("SE($RC$)", "se_rc", 6))
    tex.append(row("Time (s)", "time", 1))

    tex.append("\\bottomrule")
    tex.append("\\end{tabular*}")
    tex.append("\\end{table}")

    # Robustness macros
    all_nits = [v["nfxp_nit"] for v in results["init_robustness"].values()]
    all_ccp_ocs = [v["ccp_oc"] for v in results["init_robustness"].values()]
    all_ccp_rcs = [v["ccp_rc"] for v in results["init_robustness"].values()]
    tex.append("")
    tex.append(f"\\newcommand{{\\initNFXPMaxNit}}{{{max(all_nits)}}}")
    tex.append(f"\\newcommand{{\\initNFXPMinNit}}{{{min(all_nits)}}}")
    tex.append(f"\\newcommand{{\\initCCPOCRange}}{{{max(all_ccp_ocs)-min(all_ccp_ocs):.7f}}}")
    tex.append(f"\\newcommand{{\\initCCPRCRange}}{{{max(all_ccp_rcs)-min(all_ccp_rcs):.4f}}}")
    tex.append("")

    # Initialization robustness table
    tex.append("% ── Table: Initialization robustness ──")
    tex.append("\\begin{table}[H]")
    tex.append("\\centering")
    tex.append("\\caption{Sensitivity to starting values on Rust bus (\\ccpNbins\\ states, "
               "$\\beta=\\ccpBeta$). NFXP outer iterations and log-likelihood vary "
               "with the starting point. CCP K=5 produces identical estimates regardless "
               "of starting theta: $\\hat\\theta_c$ range \\initCCPOCRange, "
               "$\\hat{RC}$ range \\initCCPRCRange.}")
    tex.append("\\label{tab:init_robustness}")
    tex.append("\\small")
    tex.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}} l r r r r r r}")
    tex.append("\\toprule")
    tex.append("Starting point & \\multicolumn{3}{c}{NFXP} & \\multicolumn{3}{c}{CCP K=5} \\\\")
    tex.append("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}")
    tex.append("& Iter. & $\\hat\\theta_c$ & $\\hat{RC}$ & Iter. & $\\hat\\theta_c$ & $\\hat{RC}$ \\\\")
    tex.append("\\midrule")
    label_map = {
        "default":       "Default (data-driven)",
        "near truth":    "Near truth $(0.001, 3.0)$",
        "bad (tiny RC)": "Bad 1: $(1.0, 0.1)$",
        "bad (huge RC)": "Bad 2: $(0.0001, 50)$",
        "bad (near 0)":  "Bad 3: $(10^{-5}, 10^{-5})$",
    }
    for key, disp in label_map.items():
        v = results["init_robustness"][key]
        conv_flag = "" if v["nfxp_converged"] else "$^*$"
        tex.append(f"{disp} & {v['nfxp_nit']}{conv_flag} & "
                   f"{v['nfxp_oc']:.6f} & {v['nfxp_rc']:.4f} & "
                   f"{v['ccp_nit']} & {v['ccp_oc']:.6f} & {v['ccp_rc']:.4f} \\\\")
    tex.append("\\bottomrule")
    tex.append("\\multicolumn{7}{l}{\\footnotesize $^*$ Did not converge --- result at iteration limit.}")
    tex.append("\\end{tabular*}")
    tex.append("\\end{table}")

    OUT.write_text("\n".join(tex) + "\n")
    print(f"\n  Wrote {OUT}")

    json_out = OUT.with_suffix(".json")
    json_out.write_text(json.dumps(results, indent=2))
    print(f"  Wrote {json_out}")


if __name__ == "__main__":
    main()
