#!/usr/bin/env python3
"""Download and process RDW vehicle and inspection data.

Downloads data from the Dutch RDW (Rijksdienst voor het Wegverkeer) open
data API, which uses the Socrata/SODA platform. The data is freely available
under a CC-0 license at https://opendata.rdw.nl.

Three datasets are queried:
    1. Gekentekende voertuigen (m9d7-ebf2): vehicle attributes
    2. Keuringen (vkij-7mwc): APK inspection expiry dates
    3. Geconstateerde gebreken (2u8a-sfar): defect codes per inspection

The script merges these into a car-year panel suitable for dynamic discrete
choice estimation, with columns for vehicle age, defect severity, and a
scrappage indicator.

Usage:
    python scripts/download_rdw.py
    python scripts/download_rdw.py --brand TOYOTA --model COROLLA
    python scripts/download_rdw.py --min-year 2010 --max-year 2020
"""

import argparse
import json
import time
from pathlib import Path
from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

import pandas as pd
import numpy as np


# RDW SODA API endpoints
VEHICLES_URL = "https://opendata.rdw.nl/resource/m9d7-ebf2.json"
INSPECTIONS_URL = "https://opendata.rdw.nl/resource/vkij-7mwc.json"
DEFECTS_URL = "https://opendata.rdw.nl/resource/2u8a-sfar.json"

# SODA API pagination limit
PAGE_SIZE = 50000

# Defect codes in the 500-799 range are structural/safety-critical:
# 5xx = chassis/body, 6xx = steering, 7xx = braking
MAJOR_DEFECT_MIN = 500
MAJOR_DEFECT_MAX = 799


def soda_query(base_url: str, params: dict, max_retries: int = 3) -> list[dict]:
    """Execute a SODA API query with pagination and retry logic.

    Args:
        base_url: SODA API endpoint URL
        params: SoQL query parameters ($where, $select, $limit, etc.)
        max_retries: Number of retry attempts for failed requests

    Returns:
        List of result dictionaries
    """
    all_results = []
    offset = params.get("$offset", 0)
    limit = params.get("$limit", PAGE_SIZE)

    while True:
        query_params = {**params, "$limit": limit, "$offset": offset}
        url = f"{base_url}?{urlencode(query_params)}"

        for attempt in range(max_retries):
            try:
                req = Request(url)
                req.add_header("Accept", "application/json")
                with urlopen(req, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except (HTTPError, URLError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    print(f"    Retry {attempt + 1}/{max_retries} after {wait}s: {e}")
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Failed after {max_retries} attempts: {e}")

        if not data:
            break

        all_results.extend(data)
        n = len(all_results)
        print(f"    Downloaded {n:,} records...    ", end="\r")

        if len(data) < limit:
            break

        offset += limit

    print(f"    Downloaded {len(all_results):,} records total    ")
    return all_results


def download_vehicles(
    brand: str = "VOLKSWAGEN",
    model: str = "GOLF",
    min_year: int = 2005,
    max_year: int = 2015,
) -> pd.DataFrame:
    """Download vehicle registration data from RDW.

    Filters for a specific brand and model with first registration dates
    in the specified year range.
    """
    print(f"\n  Downloading vehicles: {brand} {model} ({min_year}-{max_year})")

    min_date = f"{min_year}0101"
    max_date = f"{max_year}1231"

    where_clause = (
        f"merk='{brand}' AND "
        f"handelsbenaming like '%{model}%' AND "
        f"datum_eerste_toelating >= '{min_date}' AND "
        f"datum_eerste_toelating <= '{max_date}'"
    )

    # Available columns in the vehicles resource (no brandstof_omschrijving,
    # that lives in the linked brandstof resource 8ys7-d773)
    select_cols = (
        "kenteken,"
        "datum_eerste_toelating,"
        "merk,"
        "handelsbenaming,"
        "massa_ledig_voertuig,"
        "cilinderinhoud,"
        "catalogusprijs,"
        "vervaldatum_apk,"
        "export_indicator"
    )

    params = {
        "$where": where_clause,
        "$select": select_cols,
        "$order": "kenteken",
    }

    results = soda_query(VEHICLES_URL, params)

    if not results:
        print("  No vehicles found matching criteria")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    print(f"  Found {len(df):,} vehicles")
    return df


def download_defects_for_plates(
    plates: list[str], batch_size: int = 100
) -> pd.DataFrame:
    """Download defect records for a list of license plates.

    Queries the geconstateerde gebreken (detected defects) dataset.
    Each record is one defect found during an inspection, with a defect
    code (gebrek_identificatie) and date.

    Args:
        plates: List of license plate strings
        batch_size: Number of plates per API query

    Returns:
        DataFrame with defect records
    """
    print(f"\n  Downloading defects for {len(plates):,} plates")

    all_results = []
    n_batches = (len(plates) + batch_size - 1) // batch_size

    for i in range(0, len(plates), batch_size):
        batch = plates[i : i + batch_size]
        batch_num = i // batch_size + 1

        # Build IN-style filter using OR
        plate_filter = " OR ".join(f"kenteken='{p}'" for p in batch)
        params = {
            "$where": plate_filter,
            "$select": "kenteken,gebrek_identificatie,meld_datum_door_keuringsinstantie",
        }

        try:
            results = soda_query(DEFECTS_URL, params)
            all_results.extend(results)
        except RuntimeError as e:
            print(f"    Batch {batch_num}/{n_batches} failed: {e}")

        if batch_num % 10 == 0 or batch_num == n_batches:
            print(f"  Defect progress: batch {batch_num}/{n_batches}, "
                  f"{len(all_results):,} defects so far")

    if not all_results:
        print("  No defect records found")
        return pd.DataFrame()

    df = pd.DataFrame(all_results)
    print(f"  Total defect records: {len(df):,}")
    return df


def classify_defects(defects_df: pd.DataFrame) -> pd.DataFrame:
    """Classify defects by severity and aggregate to vehicle-year level.

    Defect codes in the 500-799 range (chassis, steering, braking) are
    classified as major. All other codes are classified as minor.

    Returns:
        DataFrame with columns: kenteken, year, defect_level
        where defect_level = 0 (no defects), 1 (minor), 2 (major)
    """
    if defects_df.empty:
        return pd.DataFrame(columns=["kenteken", "year", "defect_level"])

    df = defects_df.copy()

    # Parse defect code as integer
    df["defect_code"] = pd.to_numeric(df["gebrek_identificatie"], errors="coerce")

    # Parse year from meld_datum (report date, format YYYYMMDD)
    df["year"] = pd.to_numeric(
        df["meld_datum_door_keuringsinstantie"].astype(str).str[:4],
        errors="coerce",
    )
    df = df.dropna(subset=["year", "defect_code"])
    df["year"] = df["year"].astype(int)

    # Classify: major if code in 500-799
    df["is_major"] = (
        (df["defect_code"] >= MAJOR_DEFECT_MIN)
        & (df["defect_code"] <= MAJOR_DEFECT_MAX)
    )

    # Aggregate to vehicle-year: take worst defect level
    grouped = df.groupby(["kenteken", "year"]).agg(
        has_major=("is_major", "any"),
        n_defects=("defect_code", "count"),
    ).reset_index()

    grouped["defect_level"] = 1  # minor by default (has at least one defect)
    grouped.loc[grouped["has_major"], "defect_level"] = 2  # major

    return grouped[["kenteken", "year", "defect_level"]]


def build_panel(
    vehicles_df: pd.DataFrame,
    defect_summary: pd.DataFrame,
    num_age_bins: int = 25,
    num_defect_levels: int = 3,
    current_year: int = 2025,
) -> pd.DataFrame:
    """Build a car-year panel from vehicle and defect data.

    For each vehicle, creates annual observations from the first APK
    eligibility year (age 3) through the last observed year. Exit events
    are defined as:

    1. APK expired without renewal (expired before current_year)
    2. Vehicle exported (export_indicator = Ja)
    3. No APK on record (likely scrapped early)

    Vehicles with APK valid beyond current_year are right-censored
    (their last observation has action=0).

    Args:
        vehicles_df: Vehicle attributes from download_vehicles()
        defect_summary: Vehicle-year defect levels from classify_defects()
        num_age_bins: Maximum age bin (years)
        num_defect_levels: Number of defect severity levels
        current_year: Current calendar year (vehicles with APK valid
            beyond this are right-censored)

    Returns:
        Panel DataFrame with columns: vehicle_id, year, age_bin,
        defect_level, scrapped, state
    """
    print("\n  Building car-year panel")

    vdf = vehicles_df.copy()

    # Parse registration year
    vdf["reg_year"] = pd.to_numeric(
        vdf["datum_eerste_toelating"].astype(str).str[:4],
        errors="coerce",
    )

    # Parse APK expiry year
    vdf["apk_year"] = pd.to_numeric(
        vdf["vervaldatum_apk"].astype(str).str[:4],
        errors="coerce",
    )

    # Determine if exported
    export_col = vdf.get("export_indicator", pd.Series(["Nee"] * len(vdf)))
    vdf["exported"] = export_col.fillna("Nee").str.strip().str.lower() == "ja"

    vdf = vdf.dropna(subset=["reg_year"]).copy()
    vdf["reg_year"] = vdf["reg_year"].astype(int)

    # Create defect lookup: (kenteken, year) -> defect_level
    defect_lookup = {}
    if not defect_summary.empty:
        for _, row in defect_summary.iterrows():
            defect_lookup[(row["kenteken"], int(row["year"]))] = int(row["defect_level"])

    records = []
    vid = 0
    n_exited = 0
    n_censored = 0

    for _, row in vdf.iterrows():
        reg_year = int(row["reg_year"])
        kenteken = row["kenteken"]

        # Start observations at age 3 (before first mandatory APK at age 4)
        first_obs_year = reg_year + 3

        # Determine exit status and last observed year
        apk_year = row["apk_year"]
        exported = row["exported"]

        if exported:
            # Exported vehicle: exited the Dutch fleet
            # Use APK year as approximate exit year, or reg_year + 5
            if pd.isna(apk_year):
                last_year = reg_year + 5
            else:
                last_year = int(apk_year)
            is_exit = True
        elif pd.isna(apk_year):
            # No APK on record at all: likely never inspected or removed early
            last_year = first_obs_year
            is_exit = True
        elif int(apk_year) < current_year:
            # APK expired and not renewed: vehicle exited
            last_year = int(apk_year)
            is_exit = True
        else:
            # APK still valid: right-censored observation
            last_year = current_year
            is_exit = False

        # Skip vehicles with no valid observation window
        if last_year < first_obs_year:
            continue

        # Cap at current year
        last_year = min(last_year, current_year)

        if is_exit:
            n_exited += 1
        else:
            n_censored += 1

        vid += 1

        for year in range(first_obs_year, last_year + 1):
            age = year - reg_year
            age_bin = min(age, num_age_bins - 1)

            # Look up actual defect level from defect data
            defect_level = defect_lookup.get((kenteken, year), 0)

            # Mark exited in the final observed year only
            scrapped = 1 if (year == last_year and is_exit) else 0

            records.append({
                "vehicle_id": vid,
                "year": year,
                "age_bin": age_bin,
                "defect_level": defect_level,
                "scrapped": scrapped,
                "state": age_bin * num_defect_levels + defect_level,
            })

    df = pd.DataFrame(records)

    if df.empty:
        print("  WARNING: No panel records generated")
        return df

    print(f"  Panel: {len(df):,} observations, {df['vehicle_id'].nunique():,} vehicles")
    print(f"  Exited vehicles: {n_exited:,} ({n_exited/(n_exited+n_censored):.1%})")
    print(f"  Censored vehicles: {n_censored:,} ({n_censored/(n_exited+n_censored):.1%})")
    print(f"  Per-year scrappage rate: {df['scrapped'].mean():.2%}")
    print(f"  Mean age: {df['age_bin'].mean():.1f} years")
    print(f"  Defect distribution:")
    for level in range(num_defect_levels):
        pct = (df["defect_level"] == level).mean()
        labels = {0: "pass", 1: "minor", 2: "major"}
        print(f"    Level {level} ({labels[level]}): {pct:.1%}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download and process RDW vehicle scrappage data"
    )
    parser.add_argument(
        "--brand", type=str, default="VOLKSWAGEN",
        help="Vehicle brand to filter (default: VOLKSWAGEN)"
    )
    parser.add_argument(
        "--model", type=str, default="GOLF",
        help="Model name substring to match (default: GOLF)"
    )
    parser.add_argument(
        "--min-year", type=int, default=2005,
        help="Minimum first registration year (default: 2005)"
    )
    parser.add_argument(
        "--max-year", type=int, default=2015,
        help="Maximum first registration year (default: 2015)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: src/econirl/datasets/rdw_scrappage_data.csv)"
    )
    parser.add_argument(
        "--save-intermediate", action="store_true",
        help="Save intermediate vehicle and defect files"
    )
    parser.add_argument(
        "--skip-defects", action="store_true",
        help="Skip defect download (use age-based proxy instead)"
    )
    args = parser.parse_args()

    output_path = Path(args.output) if args.output else (
        Path(__file__).parent.parent / "src/econirl/datasets/rdw_scrappage_data.csv"
    )

    print("=" * 72)
    print("RDW Vehicle Scrappage Data Download")
    print("=" * 72)
    print(f"  Brand: {args.brand}")
    print(f"  Model: {args.model}")
    print(f"  Years: {args.min_year}-{args.max_year}")
    print(f"  Output: {output_path}")

    # Step 1: Download vehicle data
    vehicles_df = download_vehicles(
        brand=args.brand,
        model=args.model,
        min_year=args.min_year,
        max_year=args.max_year,
    )

    if vehicles_df.empty:
        print("\nNo data to process. Exiting.")
        return

    if args.save_intermediate:
        intermediate_dir = output_path.parent / "rdw_intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        vehicles_df.to_csv(intermediate_dir / "vehicles.csv", index=False)
        print(f"  Saved vehicles to {intermediate_dir / 'vehicles.csv'}")

    # Step 2: Download defect records
    plates = vehicles_df["kenteken"].unique().tolist()

    if args.skip_defects:
        print("\n  Skipping defect download (--skip-defects)")
        defect_summary = pd.DataFrame(columns=["kenteken", "year", "defect_level"])
    else:
        defects_df = download_defects_for_plates(plates)

        if args.save_intermediate and not defects_df.empty:
            defects_df.to_csv(intermediate_dir / "defects.csv", index=False)
            print(f"  Saved defects to {intermediate_dir / 'defects.csv'}")

        defect_summary = classify_defects(defects_df)

    # Step 3: Build panel
    panel_df = build_panel(vehicles_df, defect_summary)

    if panel_df.empty:
        print("\nNo panel data generated. Exiting.")
        return

    # Step 4: Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(panel_df):,} observations to {output_path}")

    # Summary
    print(f"\nDataset Summary:")
    print(f"  Vehicles:       {panel_df['vehicle_id'].nunique():,}")
    print(f"  Observations:   {len(panel_df):,}")
    print(f"  Scrappage rate: {panel_df['scrapped'].mean():.2%}")
    print(f"  Age range:      {panel_df['age_bin'].min()}-{panel_df['age_bin'].max()}")
    print(f"  Mean age:       {panel_df['age_bin'].mean():.1f} years")


if __name__ == "__main__":
    main()
