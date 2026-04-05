"""NNES: Semiparametric efficiency without bias correction.

Demonstrates the key contributions of Nguyen (2025):
1. The zero-Jacobian property of NPL survives neural approximation,
   giving Neyman orthogonality without explicit bias correction
2. The neural V-network replaces NFXP's inner Bellman loop, scaling
   with data size rather than state space size
3. Standard errors from the numerical Hessian are valid because the
   block-diagonal information matrix structure is preserved
4. Comparison against NFXP on multi-component bus (900 states)

Usage:
    python examples/multi-component-bus/nnes_efficiency.py
"""

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from econirl.environments.multi_component_bus import MultiComponentBusEnvironment
from econirl.estimation.nfxp import NFXPEstimator
from econirl.estimation.nnes import NNESEstimator
from econirl.preferences.linear import LinearUtility


def main():
    print("=" * 70)
    print("NNES: Semiparametric Efficiency Without Bias Correction")
    print("Nguyen (2025)")
    print("=" * 70)

    # Multi-component bus: K=2 components, M=30 bins each = 900 states
    env = MultiComponentBusEnvironment(K=2, M=30, discount_factor=0.99)
    panel = env.generate_panel(n_individuals=500, n_periods=100, seed=42)

    transitions = env.transition_matrices
    problem = env.problem_spec
    utility = LinearUtility(
        feature_matrix=env.feature_matrix,
        parameter_names=env.parameter_names,
    )

    print(f"\n  Components: K=2, Bins per component: M=30")
    print(f"  State space: {env.num_states} states")
    print(f"  Transition matrix: {transitions.shape}, "
          f"{transitions.nbytes / 1e6:.1f} MB")
    print(f"  Observations: {panel.num_observations}")
    print(f"  True: {env.true_parameters}")

    results = {}

    # NFXP (exact Bellman inner loop)
    print("\n--- NFXP (exact Bellman, inner loop over 900 states) ---")
    t0 = time.time()
    nfxp = NFXPEstimator(
        inner_solver="hybrid",
        inner_tol=1e-10,
        se_method="robust",
    )
    results["NFXP"] = nfxp.estimate(panel, utility, problem, transitions)
    nfxp_time = time.time() - t0
    print(f"  Time: {nfxp_time:.1f}s")
    print(f"  Converged: {results['NFXP'].converged}")

    # NNES (neural V-network replaces inner loop)
    print("\n--- NNES (neural V-network, NPL variant) ---")
    t0 = time.time()
    nnes = NNESEstimator(
        hidden_dim=64,
        num_layers=2,
        v_epochs=200,
        v_lr=1e-3,
        n_outer_iterations=3,
        verbose=False,
    )
    results["NNES"] = nnes.estimate(panel, utility, problem, transitions)
    nnes_time = time.time() - t0
    print(f"  Time: {nnes_time:.1f}s")
    print(f"  Converged: {results['NNES'].converged}")

    # Parameter recovery
    print("\n" + "=" * 70)
    print("Parameter Recovery")
    print("=" * 70)
    print(f"\n{'':>18} {'True':>10} {'NFXP':>10} {'NNES':>10}")
    print("-" * 50)
    for i, name in enumerate(env.parameter_names):
        true_val = env.true_parameters[name]
        nfxp_val = float(results["NFXP"].parameters[i])
        nnes_val = float(results["NNES"].parameters[i])
        print(f"{name:>18} {true_val:>10.6f} {nfxp_val:>10.6f} {nnes_val:>10.6f}")

    # Standard errors
    print(f"\n{'Standard Errors':>18} {'':>10} {'NFXP':>10} {'NNES':>10}")
    print("-" * 50)
    for i, name in enumerate(env.parameter_names):
        nfxp_se = float(results["NFXP"].standard_errors[i])
        nnes_se = float(results["NNES"].standard_errors[i])
        print(f"{name:>18} {'':>10} {nfxp_se:>10.6f} {nnes_se:>10.6f}")

    # Timing comparison
    print(f"\n{'Wall time':>18} {'':>10} {nfxp_time:>9.1f}s {nnes_time:>9.1f}s")
    if nfxp_time > nnes_time:
        print(f"{'Speedup':>18} {'':>10} {'':>10} {nfxp_time/nnes_time:>9.1f}x")
    else:
        print(f"{'Speedup':>18} {'':>10} {nnes_time/nfxp_time:>9.1f}x {'':>10}")

    # Key insight: standard errors comparison
    print("\n" + "=" * 70)
    print("Standard Error Comparison")
    print("=" * 70)
    print("""
The zero-Jacobian property (Proposition 1 in Nguyen 2025) ensures that
first-order errors in the neural V-network drop out of the structural
parameter score. This means the standard errors from the numerical
Hessian are valid without explicit bias correction, unlike double ML
methods that require debiasing terms. The NFXP and NNES standard
errors should be similar when both converge to the same parameters.""")

    se_ratio = []
    for i, name in enumerate(env.parameter_names):
        nfxp_se = float(results["NFXP"].standard_errors[i])
        nnes_se = float(results["NNES"].standard_errors[i])
        if nfxp_se > 0 and nnes_se > 0:
            ratio = nnes_se / nfxp_se
            se_ratio.append(ratio)
            print(f"  {name}: NNES/NFXP SE ratio = {ratio:.3f}")
    if se_ratio:
        print(f"  Mean ratio: {np.mean(se_ratio):.3f}")

    # Save results
    out = {
        "env": {"K": 2, "M": 30, "n_states": env.num_states},
        "true": env.true_parameters,
        "nfxp": {
            "params": {n: float(results["NFXP"].parameters[i])
                       for i, n in enumerate(env.parameter_names)},
            "se": {n: float(results["NFXP"].standard_errors[i])
                   for i, n in enumerate(env.parameter_names)},
            "time": nfxp_time,
        },
        "nnes": {
            "params": {n: float(results["NNES"].parameters[i])
                       for i, n in enumerate(env.parameter_names)},
            "se": {n: float(results["NNES"].standard_errors[i])
                   for i, n in enumerate(env.parameter_names)},
            "time": nnes_time,
        },
    }
    path = Path(__file__).parent / "nnes_results.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {path}")


if __name__ == "__main__":
    main()
