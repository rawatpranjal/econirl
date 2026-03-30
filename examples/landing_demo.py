"""The econirl landing page demo -- 4 estimators, same data, consistent interface.

Run: python examples/landing_demo.py
"""

from econirl import NFXP, NNES, CCP, TDCCP
from econirl.datasets import load_rust_bus

df = load_rust_bus()
print(f"Loaded Rust bus data: {len(df):,} observations, {df['bus_id'].nunique()} buses\n")

estimators = {
    "NFXP": NFXP(discount=0.9999),
    "CCP": CCP(discount=0.9999, num_policy_iterations=3),
    "NNES": NNES(discount=0.9999, v_epochs=300, n_outer_iterations=2),
    "TDCCP": TDCCP(
        discount=0.9999, avi_iterations=15, epochs_per_avi=20, n_policy_iterations=2
    ),
}

for name, model in estimators.items():
    print(f"Fitting {name}...")
    model.fit(df, state="mileage_bin", action="replaced", id="bus_id")

print("\n" + "=" * 70)
print("PARAMETER COMPARISON")
print("=" * 70)
print(
    f"{'Estimator':<10} {'theta_c':>12} {'RC':>12} {'SE(theta_c)':>14} {'SE(RC)':>12}"
)
print("-" * 70)

for name, model in estimators.items():
    tc = model.params_["theta_c"]
    rc = model.params_["RC"]
    se_tc = model.se_.get("theta_c", float("nan")) if model.se_ else float("nan")
    se_rc = model.se_.get("RC", float("nan")) if model.se_ else float("nan")
    print(f"{name:<10} {tc:>12.6f} {rc:>12.4f} {se_tc:>14.6f} {se_rc:>12.4f}")

print("=" * 70)
print("\nSame problem, four algorithms, one API. All with standard errors.")
