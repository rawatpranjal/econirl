"""The econirl landing page demo.

NFXP and NNES recover the same structural parameters from the same data.
One uses exact Bellman iteration, the other uses a neural V-network.
Both give you standard errors. Same answer, different algorithms.

Run: python examples/landing_demo.py
"""

from econirl import NFXP, NNES
from econirl.datasets import load_rust_bus

df = load_rust_bus()
print(f"Loaded Rust bus data: {len(df):,} observations, {df['bus_id'].nunique()} buses\n")

print("Fitting NFXP (structural MLE with exact Bellman)...")
nfxp = NFXP(discount=0.9999).fit(df, state="mileage_bin", action="replaced", id="bus_id")

print("Fitting NNES (neural V-network with semiparametric efficiency)...")
nnes = NNES(discount=0.9999, v_epochs=300, n_outer_iterations=2).fit(
    df, state="mileage_bin", action="replaced", id="bus_id"
)

print("\n=== NFXP (Structural MLE) ===")
print(nfxp.summary())

print("\n=== NNES (Neural Network Estimation) ===")
print(nnes.summary())

print("\n--- Same problem, different algorithms. Both with standard errors. ---")
print(f"NFXP params: theta_c={nfxp.params_['theta_c']:.6f}, RC={nfxp.params_['RC']:.4f}")
print(f"NNES params: theta_c={nnes.params_['theta_c']:.6f}, RC={nnes.params_['RC']:.4f}")
