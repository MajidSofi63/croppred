#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 04: Load the prepared crop + weather data into PostgreSQL.

Default behaviour is a full refresh: the two tables are truncated and rebuilt
so stale rows from an earlier run (for example the old fabricated-weather file
or the excluded Coconut crop) cannot survive. Pass --keep to upsert without
truncating.
"""

# Import os for environment variables.
import os
# Import sys for clean fatal exits.
import sys
# Import argparse for the --keep flag.
import argparse
# Import pandas for CSV handling.
import pandas as pd
# Import Path for paths.
from pathlib import Path
# Import psycopg2 for PostgreSQL.
import psycopg2
# Import execute_values for fast bulk inserts.
from psycopg2.extras import execute_values
# Import dotenv for .env loading.
from dotenv import load_dotenv

# Resolve project paths.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
# Load .env from the project root.
load_dotenv(PROJECT_ROOT / ".env")


def ensure_schema(cur):
    """Create the tables, add the coordinate-source column, and add indexes."""
    # Districts table: one row per real state-district pair.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS districts (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            district VARCHAR(100) NOT NULL,
            lat FLOAT NOT NULL,
            lon FLOAT NOT NULL,
            coord_source VARCHAR(20),
            UNIQUE(state, district)
        );
        """
    )
    # Add the column for existing databases created before this change.
    cur.execute("ALTER TABLE districts ADD COLUMN IF NOT EXISTS coord_source VARCHAR(20);")
    # Crop statistics table: one row per district-year-season-crop.
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS crop_stats (
            id SERIAL PRIMARY KEY,
            district_id INTEGER REFERENCES districts(id) ON DELETE CASCADE,
            year INTEGER NOT NULL,
            season VARCHAR(50) NOT NULL,
            crop VARCHAR(100) NOT NULL,
            area_ha FLOAT NOT NULL,
            production_t FLOAT NOT NULL,
            yield_t_ha FLOAT NOT NULL,
            temp_mean FLOAT NOT NULL,
            rain_total FLOAT NOT NULL,
            UNIQUE(district_id, year, season, crop)
        );
        """
    )
    # Indexes that match the API's query patterns.
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_crop_stats_district_season "
        "ON crop_stats(district_id, season);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_crop_stats_district_crop_season "
        "ON crop_stats(district_id, crop, season);"
    )
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_crop_stats_district_year "
        "ON crop_stats(district_id, year);"
    )


def main():
    """Load the real data into PostgreSQL."""
    # Parse the optional flag.
    parser = argparse.ArgumentParser(description="Load crop data into PostgreSQL")
    parser.add_argument("--keep", action="store_true", help="Upsert instead of truncating")
    args = parser.parse_args()
    # Banner.
    print("=" * 80)
    print("STEP 4: DATABASE LOADING")
    print("=" * 80)
    # Read the database URL.
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("[ERROR] DATABASE_URL is not set.")
        print("[ACTION] Copy .env.example to .env and fill in your database URL.")
        sys.exit(1)
    # Connect.
    print("[INFO] Connecting to PostgreSQL")
    try:
        conn = psycopg2.connect(database_url)
        cur = conn.cursor()
    except Exception as exc:
        print(f"[ERROR] Could not connect: {exc}")
        sys.exit(1)
    # Create or migrate the schema.
    print("[INFO] Creating/verifying tables and indexes")
    ensure_schema(cur)
    conn.commit()
    # Load the real Step 2 output.
    data_path = DATA_DIR / "crop_data_with_weather.csv"
    if not data_path.exists():
        print(f"[ERROR] Missing {data_path}")
        print("[ACTION] Run: python scripts/02_fetch_weather.py")
        sys.exit(1)
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded {len(df):,} rows from {data_path.name}")
    # Refuse to load synthetic placeholder weather.
    if df["temp_mean"].nunique() <= 1 or df["rain_total"].nunique() <= 1:
        print("[ERROR] Weather columns are constant; refusing to load fake data.")
        sys.exit(1)
    # Drop the Total roll-up defensively.
    if (df["season"] == "Total").any():
        removed = int((df["season"] == "Total").sum())
        df = df[df["season"] != "Total"].copy()
        print(f"[INFO] Dropped {removed} 'Total' rows")
    # Optionally clear existing rows so the database exactly matches this file.
    if not args.keep:
        print("[INFO] Full refresh: truncating districts and crop_stats")
        cur.execute("TRUNCATE TABLE crop_stats, districts RESTART IDENTITY CASCADE;")
        conn.commit()
    # Prepare the unique district table.
    districts_df = (
        df[["state", "district", "latitude", "longitude", "coord_source"]]
        .drop_duplicates(subset=["state", "district"])
        .reset_index(drop=True)
    )
    print(f"[INFO] Inserting {len(districts_df):,} districts")
    district_rows = [
        (
            row["state"],
            row["district"],
            float(row["latitude"]),
            float(row["longitude"]),
            row["coord_source"],
        )
        for _, row in districts_df.iterrows()
    ]
    execute_values(
        cur,
        """
        INSERT INTO districts (state, district, lat, lon, coord_source)
        VALUES %s
        ON CONFLICT (state, district) DO UPDATE
        SET lat = EXCLUDED.lat, lon = EXCLUDED.lon, coord_source = EXCLUDED.coord_source
        """,
        district_rows,
        page_size=500,
    )
    conn.commit()
    # Build a (state, district) -> id map.
    cur.execute("SELECT id, state, district FROM districts")
    district_map = {(state, district): district_id for district_id, state, district in cur.fetchall()}
    # Prepare the statistics rows.
    print(f"[INFO] Inserting {len(df):,} crop statistics rows")
    crop_rows = []
    for _, row in df.iterrows():
        district_id = district_map.get((row["state"], row["district"]))
        # Skip a row if its district somehow was not inserted.
        if district_id is None:
            continue
        crop_rows.append((
            int(district_id),
            int(row["year"]),
            row["season"],
            row["crop"],
            float(row["area_ha"]),
            float(row["production_t"]),
            float(row["yield_t_ha"]),
            float(row["temp_mean"]),
            float(row["rain_total"]),
        ))
    # Bulk upsert in batches.
    batch_size = 5000
    inserted = 0
    for start in range(0, len(crop_rows), batch_size):
        batch = crop_rows[start:start + batch_size]
        execute_values(
            cur,
            """
            INSERT INTO crop_stats
                (district_id, year, season, crop, area_ha, production_t,
                 yield_t_ha, temp_mean, rain_total)
            VALUES %s
            ON CONFLICT (district_id, year, season, crop) DO UPDATE
            SET area_ha = EXCLUDED.area_ha,
                production_t = EXCLUDED.production_t,
                yield_t_ha = EXCLUDED.yield_t_ha,
                temp_mean = EXCLUDED.temp_mean,
                rain_total = EXCLUDED.rain_total
            """,
            batch,
            page_size=1000,
        )
        # Commit each batch so progress is durable.
        conn.commit()
        inserted += len(batch)
        print(f"  Progress: {inserted:,}/{len(crop_rows):,}")
    # Refresh planner statistics.
    cur.execute("ANALYZE districts;")
    cur.execute("ANALYZE crop_stats;")
    conn.commit()
    # Print final counts.
    print("\n" + "=" * 80)
    print("DATABASE STATISTICS")
    print("=" * 80)
    cur.execute("SELECT COUNT(*) FROM districts")
    print(f"[INFO] districts: {cur.fetchone()[0]:,} rows")
    cur.execute("SELECT COUNT(*) FROM crop_stats")
    print(f"[INFO] crop_stats: {cur.fetchone()[0]:,} rows")
    cur.execute("SELECT MIN(year), MAX(year) FROM crop_stats")
    min_year, max_year = cur.fetchone()
    print(f"[INFO] Years: {min_year} to {max_year}")
    cur.execute("SELECT COUNT(DISTINCT season) FROM crop_stats")
    print(f"[INFO] Seasons: {cur.fetchone()[0]}")
    cur.execute("SELECT COUNT(DISTINCT crop) FROM crop_stats")
    print(f"[INFO] Crops: {cur.fetchone()[0]}")
    print("=" * 80)
    # Close resources.
    cur.close()
    conn.close()
    print("[INFO] Database loading complete")


# Run only when executed as a script.
if __name__ == "__main__":
    main()
