"""
NFXP Estimation on Rust Bus Engine Data (for comparison)

This demonstrates the structural estimation approach that MaxEntIRL should match.
"""

import numpy as np
import pandas as pd

from econirl.datasets import load_rust_bus

# Load the bus data
print("=" * 60)
print("Loading Rust Bus Engine Data")
print("=" * 60)
df = load_rust_bus()
print(f"Observations: {len(df):,}")
print(f"Buses: {df['bus_id'].nunique()}")
print(f"Replacement rate: {df['replaced'].mean():.2%}")
print()

# Try to import NFXP
try:
    from econirl.estimators import NFXP

    print("=" * 60)
    print("Running NFXP Estimation")
    print("=" * 60)

    model = NFXP(
        n_states=90,
        n_actions=2,
        discount=0.9999,
        verbose=True,
    )

    model.fit(
        data=df,
        state="mileage_bin",
        action="replaced",
        id="bus_id",
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print(model.summary())

except ImportError as e:
    print(f"NFXP not available: {e}")
except Exception as e:
    print(f"Error running NFXP: {e}")
    import traceback
    traceback.print_exc()
