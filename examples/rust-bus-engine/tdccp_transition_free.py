"""TD-CCP: Transition-free estimation with valid inference.

Demonstrates the key contributions of Adusumilli and Eckardt (2025):
1. The semi-gradient algorithm estimates h and g from observed data
   tuples without using transition densities
2. Cross-fitting enables valid inference despite nonparametric
   first-stage estimation
3. Per-feature value decomposition reveals which utility components
   drive forward-looking behavior
4. Discretization robustness: TD-CCP maintains accuracy on coarsely
   binned data because its core algorithm works with data tuples

Usage:
    python examples/rust-bus-engine/tdccp_transition_free.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.core.types import DDCProblem, Panel, Trajectory
from econirl.environments.rust_bus import RustBusEnvironment
from econirl.estimation.ccp import CCPEstimator
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.td_ccp import TDCCPEstimator, TDCCPConfig
from econirl.preferences.linear import LinearUtility


TRUE_OC = 0.001
TRUE_RC = 3.0


def part1_equivalence():
    """Part 1: TD-CCP matches NFXP on the standard Rust bus."""
    print("=" * 70)
    print("Part 1: Parameter Recovery on Rust Bus (90 bins)")
    print("=" * 70)

    env = RustBusEnvironment(
        num_mileage_bins=90,
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        discount_factor=0.95,
    )
    panel = env.generate_panel(n_individuals=500, n_periods=100, seed=42)
    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    print(f"\n  States: {env.num_states}, Observations: {panel.num_observations}")
    print(f"  True: OC={TRUE_OC}, RC={TRUE_RC}")

    results = {}

    print("\n--- NFXP ---")
    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust")
    results["NFXP"] = nfxp.estimate(panel, utility, problem, transitions)
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n--- TD-CCP (cross-fitted) ---")
    t0 = time.time()
    tdccp = TDCCPEstimator(config=TDCCPConfig(
        method="semigradient",
        basis_dim=8,
        cross_fitting=True,
        robust_se=False,
    ))
    results["TD-CCP"] = tdccp.estimate(panel, utility, problem, transitions)
    print(f"  Time: {time.time() - t0:.1f}s")

    # Results
    print(f"\n{'':>18} {'True':>10} {'NFXP':>10} {'TD-CCP':>10}")
    print("-" * 50)
    for i, name in enumerate(env.parameter_names):
        true_val = [TRUE_OC, TRUE_RC][i]
        print(f"{name:>18} {true_val:>10.6f} "
              f"{float(results['NFXP'].parameters[i]):>10.6f} "
              f"{float(results['TD-CCP'].parameters[i]):>10.6f}")

    print(f"\n{'Standard Errors':>18} {'':>10} {'NFXP':>10} {'TD-CCP':>10}")
    print("-" * 50)
    for i, name in enumerate(env.parameter_names):
        print(f"{name:>18} {'':>10} "
              f"{float(results['NFXP'].standard_errors[i]):>10.6f} "
              f"{float(results['TD-CCP'].standard_errors[i]):>10.6f}")

    # Per-feature EV decomposition
    h_table = results["TD-CCP"].metadata["h_table"]
    params = np.array(results["TD-CCP"].parameters)
    print("\nPer-Feature EV Decomposition (TD-CCP only)")
    print(f"h_table shape: {h_table.shape}\n")
    print(f"{'Mileage':>10} {'OC contrib':>14} {'RC contrib':>14} {'Total EV':>12}")
    print("-" * 52)
    for s in [0, 15, 30, 45, 60, 75, 89]:
        h_keep = h_table[s, 0, :]
        weighted = h_keep * params
        print(f"{s:>10} {weighted[0]:>14.6f} {weighted[1]:>14.6f} {weighted.sum():>12.6f}")

    return results


def part2_discretization():
    """Part 2: TD-CCP is robust to coarse discretization."""
    print("\n\n" + "=" * 70)
    print("Part 2: Discretization Robustness")
    print("Fine data (90 bins) re-binned to 15 bins")
    print("=" * 70)

    env_fine = RustBusEnvironment(
        num_mileage_bins=90,
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        discount_factor=0.95,
    )
    panel_fine = env_fine.generate_panel(
        n_individuals=500, n_periods=100, seed=42
    )

    env_coarse = RustBusEnvironment(
        num_mileage_bins=15,
        operating_cost=TRUE_OC,
        replacement_cost=TRUE_RC,
        discount_factor=0.95,
    )

    # Re-bin states
    bin_ratio = 90 / 15
    coarse_trajs = []
    for traj in panel_fine.trajectories:
        cs = np.array([min(int(s / bin_ratio), 14) for s in np.array(traj.states)])
        cn = np.array([min(int(s / bin_ratio), 14) for s in np.array(traj.next_states)])
        ca = np.array(traj.actions)
        coarse_trajs.append(Trajectory(
            states=cs, actions=ca, next_states=cn,
            individual_id=traj.individual_id,
        ))
    panel_coarse = Panel(trajectories=coarse_trajs)

    utility_c = LinearUtility(
        feature_matrix=env_coarse.feature_matrix,
        parameter_names=env_coarse.parameter_names,
    )

    results = {}

    print("\n--- NFXP on coarse data ---")
    t0 = time.time()
    nfxp = NFXPEstimator(se_method="robust")
    results["NFXP"] = nfxp.estimate(
        panel_coarse, utility_c, env_coarse.problem_spec,
        env_coarse.transition_matrices,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    print("\n--- TD-CCP on coarse data ---")
    t0 = time.time()
    tdccp = TDCCPEstimator(config=TDCCPConfig(
        method="semigradient", basis_dim=8,
        cross_fitting=True, robust_se=False,
    ))
    results["TD-CCP"] = tdccp.estimate(
        panel_coarse, utility_c, env_coarse.problem_spec,
        env_coarse.transition_matrices,
    )
    print(f"  Time: {time.time() - t0:.1f}s")

    print(f"\n{'Coarse (15 bins)':>18} {'True':>10} {'NFXP':>10} {'TD-CCP':>10}")
    print("-" * 50)
    for i, name in enumerate(["operating_cost", "replacement_cost"]):
        true_val = [TRUE_OC, TRUE_RC][i]
        print(f"{name:>18} {true_val:>10.6f} "
              f"{float(results['NFXP'].parameters[i]):>10.6f} "
              f"{float(results['TD-CCP'].parameters[i]):>10.6f}")

    print(f"\n{'Bias (est - true)':>18} {'':>10} {'NFXP':>10} {'TD-CCP':>10}")
    print("-" * 50)
    for i, (name, tv) in enumerate([
        ("operating_cost", TRUE_OC), ("replacement_cost", TRUE_RC)
    ]):
        nb = float(results["NFXP"].parameters[i]) - tv
        tb = float(results["TD-CCP"].parameters[i]) - tv
        print(f"{name:>18} {'':>10} {nb:>+10.6f} {tb:>+10.6f}")

    return results


def main():
    results_p1 = part1_equivalence()
    results_p2 = part2_discretization()

    out = {"true": {"operating_cost": TRUE_OC, "replacement_cost": TRUE_RC}}
    for label, res in [("fine", results_p1), ("coarse", results_p2)]:
        out[label] = {}
        for name, r in res.items():
            out[label][name] = {
                "params": [float(p) for p in r.parameters],
                "se": [float(s) for s in r.standard_errors],
            }

    path = Path(__file__).parent / "tdccp_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
