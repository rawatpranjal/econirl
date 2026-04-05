"""GLADIUS: Bi-conjugate Bellman decomposition and global convergence.

Demonstrates the key contributions of Kang, Yoganarasimhan and Jain (2025):
1. Bi-conjugate Bellman decomposition separates Q(s,a) and EV(s,a)
   into two neural networks, isolating true Bellman error from
   irreducible transition noise
2. Alternating optimization (odd batches update Q via NLL, even
   batches update zeta via Bellman MSE) prevents Q-value explosion
3. Post-hoc projection onto linear features with R-squared diagnostic
4. Honest comparison against NFXP showing the known ~40% operating
   cost bias in the IRL setting

Usage:
    python examples/rust-bus-engine/gladius_bellman_showcase.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.gladius import GLADIUSEstimator, GLADIUSConfig
from econirl.estimation.nfxp import NFXPEstimator
from econirl.preferences.linear import LinearUtility
from econirl.simulation.synthetic import simulate_panel
from econirl.core.bellman import SoftBellmanOperator
from econirl.core.solvers import policy_iteration


TRUE_OC = 0.01
TRUE_RC = 3.0


def main():
    print("=" * 70)
    print("GLADIUS: Bi-Conjugate Bellman Decomposition")
    print("Kang, Yoganarasimhan and Jain (2025)")
    print("=" * 70)

    env = RustBusEnvironment(
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        num_mileage_bins=90,
        discount_factor=0.95,
    )
    utility = LinearUtility.from_environment(env)
    problem = env.problem_spec
    transitions = env.transition_matrices
    true_params = jnp.array([TRUE_OC, TRUE_RC])

    panel = simulate_panel(env, n_individuals=500, n_periods=100, seed=42)
    n_obs = sum(len(t.states) for t in panel.trajectories)

    # Compute true policy for comparison
    operator = SoftBellmanOperator(
        problem, jnp.asarray(transitions, dtype=jnp.float64)
    )
    true_reward = jnp.asarray(
        utility.compute(jnp.asarray(true_params, dtype=jnp.float32)),
        dtype=jnp.float64,
    )
    true_result = policy_iteration(
        operator, true_reward, tol=1e-10, max_iter=200, eval_method="matrix"
    )
    true_policy = true_result.policy

    print(f"\n  States: {problem.num_states}, Observations: {n_obs:,}")
    print(f"  True: OC={TRUE_OC}, RC={TRUE_RC}")
    print(f"  Discount: {problem.discount_factor}")

    # ---- NFXP (gold standard) ----
    print("\n--- NFXP ---")
    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust", verbose=False)
    nfxp_result = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    print(f"  Time: {nfxp_time:.1f}s")

    # ---- GLADIUS (alternating, LR decay) ----
    print("\n--- GLADIUS (alternating Q/EV, LR decay) ---")
    t0 = time.time()
    gladius = GLADIUSEstimator(config=GLADIUSConfig(
        q_hidden_dim=64,
        v_hidden_dim=64,
        q_num_layers=2,
        v_num_layers=2,
        max_epochs=500,
        batch_size=256,
        alternating_updates=True,
        lr_decay_rate=0.001,
        bellman_penalty_weight=1.0,
        verbose=False,
    ))
    gladius_result = gladius.estimate(panel, utility, problem, transitions)
    gladius_time = time.time() - t0
    print(f"  Time: {gladius_time:.1f}s")

    # ---- Parameter comparison ----
    print("\n" + "=" * 70)
    print("Parameter Recovery")
    print("=" * 70)
    nfxp_p = np.asarray(nfxp_result.parameters)
    glad_p = np.asarray(gladius_result.parameters)

    print(f"\n{'':>18} {'True':>10} {'NFXP':>10} {'GLADIUS':>10}")
    print("-" * 50)
    print(f"{'operating_cost':>18} {TRUE_OC:>10.6f} {nfxp_p[0]:>10.6f} {glad_p[0]:>10.6f}")
    print(f"{'replacement_cost':>18} {TRUE_RC:>10.4f} {nfxp_p[1]:>10.4f} {glad_p[1]:>10.4f}")

    # Bias
    print(f"\n{'Bias':>18} {'':>10} {'NFXP':>10} {'GLADIUS':>10}")
    print("-" * 50)
    print(f"{'operating_cost':>18} {'':>10} {nfxp_p[0]-TRUE_OC:>+10.6f} {glad_p[0]-TRUE_OC:>+10.6f}")
    print(f"{'replacement_cost':>18} {'':>10} {nfxp_p[1]-TRUE_RC:>+10.4f} {glad_p[1]-TRUE_RC:>+10.4f}")

    oc_bias_pct = abs(glad_p[0] - TRUE_OC) / TRUE_OC * 100
    rc_bias_pct = abs(glad_p[1] - TRUE_RC) / TRUE_RC * 100
    print(f"\nGLADIUS bias: OC {oc_bias_pct:.0f}%, RC {rc_bias_pct:.1f}%")

    # ---- Policy comparison ----
    max_pdiff = float(jnp.max(jnp.abs(
        jnp.asarray(gladius_result.policy) - true_policy
    )))
    nfxp_pdiff = float(jnp.max(jnp.abs(
        jnp.asarray(nfxp_result.policy) - true_policy
    )))
    print(f"\nMax policy difference from true:")
    print(f"  NFXP:    {nfxp_pdiff:.6f}")
    print(f"  GLADIUS: {max_pdiff:.6f}")

    # ---- Bi-conjugate Bellman decomposition ----
    print("\n" + "=" * 70)
    print("Bi-Conjugate Bellman Decomposition")
    print("=" * 70)

    if "reward_table" in gladius_result.metadata:
        reward_table = np.array(gladius_result.metadata["reward_table"])
        q_table = np.array(gladius_result.metadata.get("q_table", []))
        ev_table = np.array(gladius_result.metadata.get("ev_table", []))

        print("\nThe decomposition: r(s,a) = Q(s,a) - beta * EV(s,a)")
        print("Q is trained via NLL of observed actions.")
        print("EV is trained via Bellman MSE against V(s').\n")

        print(f"{'Mileage':>10} {'Q(s,keep)':>12} {'EV(s,keep)':>12} {'r(s,keep)':>12}")
        print("-" * 48)
        for s in [0, 15, 30, 45, 60, 75, 89]:
            if len(q_table) > 0 and len(ev_table) > 0:
                print(f"{s:>10} {q_table[s,0]:>12.4f} {ev_table[s,0]:>12.4f} {reward_table[s,0]:>12.4f}")
            else:
                print(f"{s:>10} {'n/a':>12} {'n/a':>12} {reward_table[s,0]:>12.4f}")

    # ---- Projection R-squared ----
    print("\n" + "=" * 70)
    print("Linear Projection R-squared")
    print("=" * 70)

    if "reward_table" in gladius_result.metadata:
        reward_table = np.array(gladius_result.metadata["reward_table"])
        feature_matrix = np.array(env.feature_matrix)

        # Action-difference projection (same as GLADIUS internals)
        n_states = problem.num_states
        dphi_list, dr_list = [], []
        for s in range(n_states):
            dr = reward_table[s, 1] - reward_table[s, 0]
            dphi = feature_matrix[s, 1, :] - feature_matrix[s, 0, :]
            dphi_list.append(dphi)
            dr_list.append(dr)
        X = np.array(dphi_list)
        y = np.array(dr_list)
        theta_proj, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ theta_proj
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        print(f"\nR-squared of linear features explaining neural reward: {r_squared:.4f}")
        print(f"Projected params: OC={theta_proj[0]:.6f}, RC={theta_proj[1]:.4f}")

        if r_squared > 0.95:
            print("The linear features capture most of the neural reward surface.")
        else:
            print("The neural reward has structure beyond what linear features capture.")

    # ---- Known limitation ----
    print("\n" + "=" * 70)
    print("Known Limitation: Identification in IRL Setting")
    print("=" * 70)
    print(f"""
Without observed rewards, GLADIUS trains Q via NLL only (behavioral
cloning). This identifies Q up to a state-dependent constant c(s)
that leaks asymmetrically into implied rewards through the transition
structure. On the Rust bus with beta=0.95, this typically produces
about 40 percent bias on operating cost while recovering replacement
cost within 10 percent. This is structural, not a tuning problem.

NFXP solves the Bellman equation exactly and does not suffer this
identification issue. Use GLADIUS when the state space is continuous
and NFXP cannot operate, or when rewards are observed in the data.""")

    # ---- Save results ----
    out = {
        "true": {"operating_cost": TRUE_OC, "replacement_cost": TRUE_RC},
        "nfxp": {
            "params": [float(p) for p in nfxp_result.parameters],
            "se": [float(s) for s in nfxp_result.standard_errors],
            "time": nfxp_time,
            "max_policy_diff": nfxp_pdiff,
        },
        "gladius": {
            "params": [float(p) for p in gladius_result.parameters],
            "time": gladius_time,
            "max_policy_diff": max_pdiff,
        },
    }
    path = Path(__file__).parent / "gladius_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
