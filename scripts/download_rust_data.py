#!/usr/bin/env python3
"""Download and process original Rust (1987) bus data.

Sources:
- NFXP Software data files: https://github.com/b-rodrigues/rust/tree/master/datasets
- Original source: John Rust's NFXP software package

The data consists of monthly odometer readings for 162 buses from the
Madison Metropolitan Bus Company, December 1974 to May 1985.

For Rust (1987) replication, Groups 1-4 are used:
- Group 1: g870.asc (Grumman model 870)
- Group 2: rt50.asc (Chance model RT50)
- Group 3: t8h203.asc (GMC model T8H203)
- Group 4: a530875.asc (GMC model A5308, 1975)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from urllib.request import urlopen
from io import StringIO


# NFXP data files for Groups 1-4 used in Rust (1987)
NFXP_BASE_URL = "https://raw.githubusercontent.com/b-rodrigues/rust/master/datasets"
GROUP_FILES = {
    1: ("g870.asc", 36, 15),      # 36 rows x 15 buses
    2: ("rt50.asc", 60, 4),       # 60 rows x 4 buses
    3: ("t8h203.asc", 81, 48),    # 81 rows x 48 buses
    4: ("a530875.asc", 128, 37),  # 128 rows x 37 buses
}


def download_nfxp_file(filename: str) -> list:
    """Download an ASC file from the NFXP repository."""
    url = f"{NFXP_BASE_URL}/{filename}"
    try:
        with urlopen(url) as response:
            content = response.read().decode('utf-8')
            # Parse the single-column format (values separated by newlines/whitespace)
            values = [int(float(x.strip())) for x in content.split() if x.strip()]
            return values
    except Exception as e:
        print(f"Failed to download {filename}: {e}")
        return None


def parse_nfxp_data(values: list, n_rows: int, n_buses: int, group: int) -> pd.DataFrame:
    """Parse NFXP ASC format into a DataFrame.

    NFXP format (from README):
    - Data is a matrix vectorized column-by-column
    - First 11 rows of each bus column are header info:
        1. Bus number
        2. Month purchased
        3. Year purchased
        4. Month of 1st engine replacement
        5. Year of 1st engine replacement
        6. Odometer at 1st replacement
        7. Month of 2nd replacement
        8. Year of 2nd replacement
        9. Odometer at 2nd replacement
        10. Month odometer data begins
        11. Year odometer data begins
    - Remaining rows are monthly odometer readings
    """
    records = []
    bus_id_offset = sum(GROUP_FILES[g][2] for g in range(1, group))  # Offset for unique IDs

    for bus_idx in range(n_buses):
        # Extract this bus's column
        start_idx = bus_idx * n_rows
        end_idx = start_idx + n_rows
        bus_data = values[start_idx:end_idx]

        if len(bus_data) < 11:
            continue

        # Parse header
        original_bus_id = bus_data[0]
        month_purchased = bus_data[1]
        year_purchased = bus_data[2]
        replace1_month = bus_data[3]
        replace1_year = bus_data[4]
        replace1_odo = bus_data[5]
        replace2_month = bus_data[6]
        replace2_year = bus_data[7]
        replace2_odo = bus_data[8]
        start_month = bus_data[9]
        start_year = bus_data[10]

        # Odometer readings start at index 11
        odometer_readings = bus_data[11:]

        # Create unique bus_id across all groups
        bus_id = bus_id_offset + bus_idx + 1

        # Generate date sequence
        current_month = start_month
        current_year = start_year

        for period, odo in enumerate(odometer_readings, start=1):
            if odo == 0 and period > 1:
                # End of data for this bus
                break

            # Check if replacement occurred this period
            replaced = 0
            if replace1_year > 0 and replace1_month > 0:
                if current_year == replace1_year and current_month == replace1_month:
                    replaced = 1
            if replace2_year > 0 and replace2_month > 0:
                if current_year == replace2_year and current_month == replace2_month:
                    replaced = 1

            records.append({
                'bus_id': bus_id,
                'period': period,
                'mileage': odo / 1000.0,  # Convert to thousands
                'replaced': replaced,
                'group': group,
                'original_bus_id': original_bus_id,
            })

            # Advance month
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

    return pd.DataFrame(records)


def download_rust_data() -> pd.DataFrame:
    """Download and parse all Groups 1-4 bus data from NFXP files."""
    all_data = []

    for group, (filename, n_rows, n_buses) in GROUP_FILES.items():
        print(f"Downloading Group {group}: {filename}...")
        values = download_nfxp_file(filename)

        if values is None:
            print(f"  Failed to download {filename}")
            continue

        expected_size = n_rows * n_buses
        actual_size = len(values)
        print(f"  Downloaded {actual_size} values (expected {expected_size})")

        if actual_size < expected_size:
            print(f"  Warning: file smaller than expected")

        df = parse_nfxp_data(values, n_rows, n_buses, group)
        print(f"  Parsed {len(df)} observations for {df['bus_id'].nunique()} buses")
        all_data.append(df)

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True)


def process_rust_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process raw Rust data into econirl format.

    Target format:
    - bus_id: int (unique across all groups)
    - period: int (1-indexed)
    - mileage: float (in thousands)
    - mileage_bin: int (5000-mile bins, 0-89)
    - replaced: int (0/1)
    - group: int (1-4 for paper analysis)
    """
    df = df.copy()

    # Create mileage bins (5000 miles = 5 units in thousands)
    df['mileage_bin'] = (df['mileage'] / 5.0).astype(int).clip(0, 89)

    # Sort and reset index
    df = df.sort_values(['bus_id', 'period']).reset_index(drop=True)

    return df[['bus_id', 'period', 'mileage', 'mileage_bin', 'replaced', 'group']]


def save_data(df: pd.DataFrame, output_path: Path):
    """Save processed data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} observations to {output_path}")

    # Print summary
    print(f"\nDataset Summary:")
    print(f"  Buses: {df['bus_id'].nunique()}")
    print(f"  Observations: {len(df):,}")
    print(f"  Groups: {sorted(df['group'].unique())}")
    print(f"  Replacement rate: {df['replaced'].mean():.2%}")
    print(f"  Mean mileage: {df['mileage'].mean():.1f} (thousands)")
    print(f"  Mean mileage bin: {df['mileage_bin'].mean():.1f}")

    # Per-group summary
    print(f"\nPer-Group Summary:")
    for group in sorted(df['group'].unique()):
        g_df = df[df['group'] == group]
        print(f"  Group {group}: {g_df['bus_id'].nunique()} buses, "
              f"{len(g_df):,} obs, "
              f"{g_df['replaced'].sum()} replacements")


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "src/econirl/datasets/rust_bus_original.csv"

    df = download_rust_data()
    if df is not None:
        df = process_rust_data(df)
        save_data(df, output_path)
    else:
        print("Failed to download data. Please try again or download manually.")
