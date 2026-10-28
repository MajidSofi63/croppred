#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 02: Build Weather Features for India Crop Recommendation System
Purpose:
  - Attach real coordinates to every state-district pair.
  - Fetch REAL historical daily weather (no synthetic values, ever) from
    NASA POWER, which is free and needs no API key.
  - Aggregate daily weather into season-level temperature and rainfall.
  - Merge those aggregates onto the prepared crop rows.

Why NASA POWER instead of Open-Meteo archive?
  Open-Meteo's archive host (archive-api.open-meteo.com) is unreachable from
  this machine, and its geocoding API returns wrong places for many Indian
  districts (e.g. MYSORE -> "Mysore Road Tolgate"). NASA POWER provides real
  daily T2M (2 m air temperature) and PRECTOTCORR (corrected precipitation)
  for 1981-present at any point, with no key.
"""

# Import operating-system helpers.
import os
# Import sys for clean fatal-error exits.
import sys
# Import time for retry backoff.
import time
# Import re for safe filename slugs.
import re
# Import json for reading the district GeoJSON.
import json
# Import difflib for fuzzy district-name matching.
import difflib
# Import pandas for tabular processing.
import pandas as pd
# Import numpy for NaN handling.
import numpy as np
# Import requests for the NASA POWER HTTP API.
import requests
# Import Path for filesystem operations.
from pathlib import Path
# Import date helpers for season windows.
from datetime import date
# Import the thread pool for concurrent, polite fetching.
from concurrent.futures import ThreadPoolExecutor, as_completed

# Resolve the scripts/ directory.
SCRIPT_DIR = Path(__file__).resolve().parent
# Resolve the project root.
PROJECT_ROOT = SCRIPT_DIR.parent
# Resolve the data directory.
DATA_DIR = PROJECT_ROOT / "data"

# NASA POWER daily point endpoint (real observations / reanalysis).
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# Real India district boundary file used to derive district centroids.
GEOJSON_URL = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
# Local copy of the GeoJSON (downloaded only when coordinates must be rebuilt).
GEOJSON_PATH = DATA_DIR / "india_district.geojson"
# Cached coordinate table (state, district, latitude, longitude, ...).
COORDS_CACHE = DATA_DIR / "district_coords.csv"
# Directory holding one compressed daily-weather file per district.
WEATHER_CACHE_DIR = DATA_DIR / "weather_cache"
# Create the cache directory if needed.
WEATHER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
# Canonical prepared-data input.
PREPARED_PATH = DATA_DIR / "crop_data_prepared.csv"
# Final output with weather columns.
OUTPUT_PATH = DATA_DIR / "crop_data_with_weather.csv"

# Number of concurrent NASA POWER requests. Kept modest to stay polite.
MAX_WORKERS = 5
# Number of HTTP retries per district.
MAX_RETRIES = 4
# Base backoff in seconds between retries.
RETRY_BACKOFF = 2.0

# Canonical season windows. Months are 1-based; end_month is inclusive.
# "crosses_year" means the season ends in March of the following calendar year.
SEASON_WINDOWS = {
    "Kharif": {
        "start": (6, 1),
        "end": (10, 31),
        "crosses_year": False,
        "description": "Jun-Oct monsoon season",
    },
    "Rabi": {
        "start": (11, 1),
        "end": (3, 31),
        "crosses_year": True,
        "description": "Nov-Mar, harvested in the following calendar year",
    },
    "Summer": {
        "start": (3, 1),
        "end": (6, 30),
        "crosses_year": False,
        "description": "Mar-Jun summer/zaid window",
    },
    "Autumn": {
        "start": (9, 1),
        "end": (11, 30),
        "crosses_year": False,
        "description": "Sep-Nov post-monsoon window",
    },
    "Winter": {
        "start": (12, 1),
        "end": (3, 31),
        "crosses_year": True,
        "description": "Dec-Mar winter window",
    },
    "Whole Year": {
        "start": (1, 1),
        "end": (12, 31),
        "crosses_year": False,
        "description": "Jan-Dec full year",
    },
}

# Map lower-case spellings from the raw dataset onto canonical names.
SEASON_ALIASES = {
    "kharif": "Kharif",
    "rabi": "Rabi",
    "summer": "Summer",
    "autumn": "Autumn",
    "winter": "Winter",
    "whole year": "Whole Year",
}

# Explicit renames for historical district names that the boundary file uses
# under a different spelling. Keys and values are normalised upper-case names.
DISTRICT_ALIASES = {
    "D VALLEY": "LOWER DIBANG VALLEY",
    "E KAMENG": "EAST KAMENG",
    "W KAMENG": "WEST KAMENG",
    "E SIANG": "EAST SIANG",
    "U SIANG": "UPPER SIANG",
    "L SUBANSIRI": "LOWER SUBANSIRI",
    "U SUBANSIRI": "UPPER SUBANSIRI",
    "L/SUBABSIRI": "LOWER SUBANSIRI",
    "U/SIANG": "UPPER SIANG",
    "K KUMEY": "KURUNG KUMEY",
    "P PARE": "PAPUM PARE",
    "N C HILLS": "NORTH CACHAR HILLS",
    "24 PARGANAS NORTH": "NORTH 24 PARGANAS",
    "24 PARGANAS SOUTH": "SOUTH 24 PARGANAS",
    "BROACH": "BHARUCH",
    "DOHAD": "DAHOD",
    "BULSAR": "VALSAD",
    "KUTCH": "KACHCHH",
    "DANGS": "THE DANGS",
    "QUILON": "KOLLAM",
    "TRIVANDRUM": "THIRUVANANTHAPURAM",
    "MONGHYR": "MUNGER",
    "PALAMAU": "PALAMU",
    "CHAMPARAN EAST": "PURBA CHAMPARAN",
    "CHAMPARAN WEST": "PASCHIM CHAMPARAN",
    "ANANTNAGH": "ANANTNAG KASHMIR SOUTH",
    "BARAMULLA": "BARAMULA KASHMIR NORTH",
    "BUDGAM": "BAGDAM",
    "LEH": "LADAKH LEH",
    "KODAGU COORG": "KODAGU",
    "KAWARDHA KABIRDHAM": "KAWARDHA",
    "CHICKBALLAPUR": "CHIKBALLAPUR",
    "RAMANAGAR": "RAMANAGARAM",
    "UTTARAKANNADA": "UTTAR KANNAD",
    "DAKSHINAKANNADA": "DAKSHIN KANNAD",
    "NARSIMPUR": "NARSINGHPUR",
    "NEEMACH": "NEEMUCH",
    "SHEOPUR KALA": "SHEOPUR",
    "MANDURBAR": "NANDURBAR",
    "BURAGARH": "BARGARH",
    "KHURDA": "KHORDHA",
    "NAWORANGPUR": "NABARANGPUR",
    "BHATINDA": "BATHINDA",
    "ARHASIA": "ARARIA",
    "BAGHALPUR": "BHAGALPUR",
    "BHABHA": "BHABUA",
    "JAHANABAD": "JEHANABAD",
    "MADHUPURA": "MADHEPURA",
    "SHKHPURA": "SHEIKHPURA",
    "SUMAL": "SUPAUL",
    "ZAMUI": "JAMUI",
    "ORANGABAD": "AURANGABAD",
    "KAMRUP M": "KAMRUP",
    "MORIGON": "MARIGAON",
    "NEW LAKHIMPUR": "LAKHIMPUR",
    "KULU": "KULLU",
    "POONCH": "PUNCH",
    "PULWANNA": "PULWAMA",
    "RAJOURI": "RAJAURI",
    "EAST SINGHBHUM": "PURBA SINGHBHUM",
    "WEST SINGHBHUM": "PASCHIM SINGHBHUM",
    "SINGHBHUM": "PURBA SINGHBHUM",
    "SINGHBHUR WEST": "PASCHIM SINGHBHUM",
    "SERAIKELA": "SARAIKELA KHARSAWAN",
    "GADCHIROLI": "GARHCHIROLI",
    "RAIGAD": "RAIGARH",
    "SARGUJA": "SURGUJA",
    "DANTEWARA": "DANTEWADA",
    "AHMEDABAD": "AHMADABAD",
    "MEHSANA": "MAHESANA",
    "JUNAGARH": "JUNAGADH",
    "CHAMARAJANNAGAR": "CHAMRAJNAGAR",
    "PATHANAMTHITTA": "PATTANAMTITTA",
    "WYNAD": "WAYANAD",
    "MAHABOOBNAGAR": "MAHBUBNAGAR",
    "KHANDWA": "EAST NIMAR",
    "DHAMANTARI": "DHAMTARI",
    "JANGIR": "JANJGIR CHAMPA",
    "JUSHPUR": "JASHPUR",
    "KANKAR": "KANKER",
    "KHARGAON": "KHARGONE",
    "KORIYA": "KOREA",
    "KWARDHA": "KAWARDHA",
    "BEED": "BID",
    "BALASORE": "BALESHWAR",
    "NAWAPARA": "NUAPADA",
    "PHULBANI": "KANDHAMAL",
    "N SHAHAR": "NAWAN SHEHAR",
    "ROPAR RUPNAGAR": "RUPNAGAR",
    "KRISHNAGIRI": "KRISHNAGIRI",
    "THIRUPPUR": "TIRUPPUR",
    "TIRUNELVELI": "TIRUNELVELI KATTABO",
    "CHHATRAPTI SAHUJI MAHARAJ NGR": "AMBEDKAR NAGAR",
    "G BUDDHA NGR": "GAUTAM BUDDHA NAGAR",
    "J B PHULE NGR": "JYOTIBA PHULE NAGAR",
    "KANPUR CITY": "KANPUR",
    "KHERI": "LAKHIMPUR KHERI",
    "MAHAMAYA NAGAR": "HATHRAS",
    "RAMABAI NAGAR": "KANPUR DEHAT",
    "U S NAGAR": "UDHAM SINGH NAGAR",
    "BURDWAN": "BARDDHAMAN",
    "COOCH BEHAR": "KOCHBIHAR",
    "DINAJPUR SOUTH": "DAKSHIN DINAJPUR",
    "DINAJUR NORTH": "UTTAR DINAJPUR",
    "DINAJPUR NORTH": "UTTAR DINAJPUR",
    "HOOGHLY": "HUGLI",
    "HOWRAH": "HAORA",
    "MIDNAPUR EAST": "EAST MIDNAPORE",
    "MIDNAPUR WEST": "WEST MIDNAPORE",
    "WEST DINAJPUR": "UTTAR DINAJPUR",
    "NORTH": "NORTH SIKKIM",
    "SOUTH": "SOUTH SIKKIM",
    "WEST": "WEST SIKKIM",
    "EAST": "EAST SIKKIM",
    "NICOBAR": "NICOBAR ISLANDS",
    "NORTH AND MIDDLE ANDAMAN": "ANDAMAN ISLANDS",
    "SOUTH ANDAMAN": "ANDAMAN ISLANDS",
    "CHIRANG": "BONGAIGAON",
    "UDALGURI": "DARRANG",
    "ARVAL": "ARWAL",
    "SHIVHAR": "SHEOHAR",
    "TAPI": "SURAT",
    "MEWAT": "GURGAON",
    "PALWAL": "FARIDABAD",
    "KUPWARA": "KUPWARA MUZAFFARABAD",
    "ANJAW": "LOHIT",
    "PONDYCHERRY DIST": "PUDUCHERRY",
    "CHHIMTUIPUI": "SAIHA",
    "EAST AIZAWL": "AIZAWL",
    "WEST AIZAWL": "AIZAWL",
}

# State aliases for the boundary file's names.
STATE_ALIASES = {
    "A AND N ISLANDS": "ANDAMAN AND NICOBAR",
    "D AND N HAVELI": "DADRA AND NAGAR HAVELI",
    "PONDICHERRY": "PUDUCHERRY",
}


def canonical_season(value: str) -> str:
    """Return the canonical season name, or the trimmed original if unknown."""
    # Normalise casing and spacing.
    key = re.sub(r"\s+", " ", str(value).strip().lower())
    # Return the canonical spelling when known.
    return SEASON_ALIASES.get(key, str(value).strip().title())


def _norm_name(value: str) -> str:
    """Normalise a place name for fuzzy matching."""
    # Upper-case and turn ampersands into the word AND.
    text = str(value).upper().replace("&", " AND ")
    # Replace punctuation with spaces.
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    # Collapse whitespace.
    return re.sub(r"\s+", " ", text).strip()


def _ring_centroid(ring: list) -> tuple:
    """Compute a polygon centroid with the shoelace formula."""
    # Fall back to the coordinate mean for degenerate rings.
    if len(ring) < 3:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    # Accumulate signed area and centroid numerators.
    area = 0.0
    cx = 0.0
    cy = 0.0
    for i in range(len(ring) - 1):
        x0, y0 = ring[i][0], ring[i][1]
        x1, y1 = ring[i + 1][0], ring[i + 1][1]
        cross = x0 * y1 - x1 * y0
        area += cross
        cx += (x0 + x1) * cross
        cy += (y0 + y1) * cross
    # Handle zero-area rings.
    if abs(area) < 1e-12:
        xs = [p[0] for p in ring]
        ys = [p[1] for p in ring]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    # Convert to a real centroid.
    area *= 0.5
    return (cx / (6 * area), cy / (6 * area))


def _geometry_centroid(geometry: dict) -> tuple:
    """Return (longitude, latitude) for the largest polygon in a geometry."""
    # MultiPolygons contain several polygons; single Polygons contain one.
    polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    # Keep the largest polygon because small islands can bias the centroid.
    best_ring = None
    best_area = 0.0
    for polygon in polygons:
        ring = polygon[0]
        area = 0.0
        for i in range(len(ring) - 1):
            area += ring[i][0] * ring[i + 1][1] - ring[i + 1][0] * ring[i][1]
        area = abs(area) / 2.0
        if area > best_area:
            best_area = area
            best_ring = ring
    # Return the centroid of the largest ring.
    return _ring_centroid(best_ring)


def _download_geojson() -> dict:
    """Download and cache the real India district boundary file."""
    # Reuse the local copy when present.
    if GEOJSON_PATH.exists() and GEOJSON_PATH.stat().st_size > 1_000_000:
        print(f"[INFO] Using cached district boundaries: {GEOJSON_PATH.name}")
        return json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
    # Download the file from GitHub.
    print(f"[INFO] Downloading district boundaries from {GEOJSON_URL}")
    response = requests.get(GEOJSON_URL, timeout=300)
    # Fail loudly if the download did not work.
    response.raise_for_status()
    # Persist it for future runs.
    GEOJSON_PATH.write_bytes(response.content)
    # Return the parsed payload.
    return response.json()


def _build_coords_from_geojson(pairs: pd.DataFrame) -> pd.DataFrame:
    """Derive a coordinate for every state-district pair from real boundaries."""
    # Load the boundary file.
    geojson = _download_geojson()
    # Build a flat table of boundary centroids.
    rows = []
    for feature in geojson["features"]:
        # Compute the representative point of the polygon.
        lon, lat = _geometry_centroid(feature["geometry"])
        # Read the real state and district names.
        properties = feature["properties"]
        rows.append({
            "geo_state": properties["NAME_1"],
            "geo_district": properties["NAME_2"],
            "latitude": round(float(lat), 5),
            "longitude": round(float(lon), 5),
        })
    # Convert to a DataFrame.
    geo = pd.DataFrame(rows)
    # Normalise names for matching.
    geo["nd"] = geo["geo_district"].map(_norm_name)
    geo["ns"] = geo["geo_state"].map(_norm_name)
    # Map the boundary file's state names onto the dataset's spelling.
    geo["ns"] = geo["ns"].map(lambda s: STATE_ALIASES.get(s, s))
    # Pre-compute a fallback centroid per state (mean of its district centroids).
    state_centroids = geo.groupby("ns")[["latitude", "longitude"]].mean().to_dict("index")
    # Accumulate one output row per dataset pair.
    output = []
    district_matches = 0
    state_fallbacks = 0
    for _, pair in pairs.iterrows():
        # Normalise the dataset names and apply the explicit renames.
        state_key = _norm_name(pair["state"])
        state_key = STATE_ALIASES.get(state_key, state_key)
        district_key = _norm_name(pair["district"])
        district_key = DISTRICT_ALIASES.get(district_key, district_key)
        # Score candidate districts within the same state.
        def best_match(frame):
            best_row = None
            best_score = 0.0
            for _, candidate in frame.iterrows():
                score = difflib.SequenceMatcher(None, district_key, candidate["nd"]).ratio()
                # Treat identical token sets as a near-perfect match.
                if set(district_key.split()) == set(candidate["nd"].split()):
                    score = max(score, 0.99)
                # Treat prefix matches as strong matches.
                if district_key and (
                    candidate["nd"].startswith(district_key + " ")
                    or district_key.startswith(candidate["nd"] + " ")
                ):
                    score = max(score, 0.93)
                if score > best_score:
                    best_score = score
                    best_row = candidate
            return best_row, best_score
        # First search inside the correct state.
        match, score = best_match(geo[geo["ns"] == state_key])
        # If that fails, allow a very strong national match (protects against
        # pre-2000 state boundaries such as Bihar/Jharkhand and MP/Chhattisgarh).
        if score < 0.86:
            national_match, national_score = best_match(geo)
            if national_score >= 0.90 and national_score > score:
                match = national_match
                score = national_score
        # Accept the district match when confident enough.
        if match is not None and score >= 0.82:
            output.append({
                "state": pair["state"],
                "district": pair["district"],
                "latitude": float(match["latitude"]),
                "longitude": float(match["longitude"]),
                "geo_state": match["geo_state"],
                "geo_district": match["geo_district"],
                "coord_source": "district",
            })
            district_matches += 1
        else:
            # Otherwise fall back to the real state centroid and label it.
            centroid = state_centroids.get(state_key)
            if centroid is None:
                print(f"[WARN] No centroid available for {pair['state']} / {pair['district']}")
                continue
            output.append({
                "state": pair["state"],
                "district": pair["district"],
                "latitude": float(centroid["latitude"]),
                "longitude": float(centroid["longitude"]),
                "geo_state": pair["state"],
                "geo_district": None,
                "coord_source": "state_fallback",
            })
            state_fallbacks += 1
    # Report coverage honestly.
    print(f"[INFO] Coordinates: {district_matches} district-level matches, "
          f"{state_fallbacks} state-centroid fallbacks")
    # Return the coordinate table.
    return pd.DataFrame(output)


def build_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Load or build the coordinate table for every state-district pair."""
    # List the pairs that the crop data actually needs.
    pairs = df[["state", "district"]].drop_duplicates().copy()
    # Load the cached table when it exists.
    if COORDS_CACHE.exists():
        cached = pd.read_csv(COORDS_CACHE)
        # Compare place names case-insensitively so "ANDHRA PRADESH" and
        # "Andhra Pradesh" are treated as the same state.
        pair_keys = (
            pairs["state"].astype(str).str.strip().str.upper()
            + "|"
            + pairs["district"].astype(str).str.strip().str.upper()
        )
        cached_keys = (
            cached["state"].astype(str).str.strip().str.upper()
            + "|"
            + cached["district"].astype(str).str.strip().str.upper()
        )
        # Accept the cache only if it covers every needed pair.
        missing = sorted(set(pair_keys) - set(cached_keys))
        if not missing:
            print(f"[INFO] Loaded complete coordinate cache: {COORDS_CACHE.name}")
            return cached
        # Otherwise rebuild everything to keep the file consistent.
        print(f"[INFO] Coordinate cache is missing {len(missing)} pairs; rebuilding")
    # Build from real boundaries.
    coords = _build_coords_from_geojson(pairs)
    # Save the cache.
    coords.to_csv(COORDS_CACHE, index=False)
    print(f"[INFO] Saved coordinates to {COORDS_CACHE.name}")
    # Return the table.
    return coords


def _slug(value: str) -> str:
    """Turn a place name into a filename-safe slug."""
    # Upper-case, replace anything unsafe with an underscore.
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def _cache_path(state: str, district: str, lat: float, lon: float, start: date, end: date) -> Path:
    """Build a cache path that cannot collide across states or date ranges."""
    # Include state, district, coordinates, and range in the filename.
    name = f"{_slug(state)}_{_slug(district)}_{lat:.5f}_{lon:.5f}_{start:%Y%m%d}_{end:%Y%m%d}.csv.gz"
    # Return the full path.
    return WEATHER_CACHE_DIR / name


def _parse_nasa_payload(payload: dict) -> pd.DataFrame:
    """Convert a NASA POWER JSON payload into a clean daily DataFrame."""
    # Read the two parameters.
    parameters = payload["properties"]["parameter"]
    temperatures = parameters["T2M"]
    rainfall = parameters["PRECTOTCORR"]
    # Build a date index from the returned keys.
    dates = sorted(temperatures.keys())
    # Convert YYYYMMDD strings to real dates.
    parsed_dates = pd.to_datetime(dates, format="%Y%m%d")
    # Read the values in the same order.
    temp_values = pd.to_numeric(pd.Series([temperatures[d] for d in dates]), errors="coerce")
    rain_values = pd.to_numeric(pd.Series([rainfall[d] for d in dates]), errors="coerce")
    # NASA POWER uses large negative sentinels for missing values.
    temp_values = temp_values.mask(temp_values <= -900)
    rain_values = rain_values.mask(rain_values <= -900)
    # Assemble the final frame.
    return pd.DataFrame({
        "date": parsed_dates,
        "temp_mean": temp_values.values,
        "rain_total": rain_values.values,
    })


def fetch_district_daily(state: str, district: str, lat: float, lon: float,
                         start: date, end: date) -> tuple:
    """Fetch (or load) daily weather for one district, with retries."""
    # Build the cache path for this exact district and range.
    cache_path = _cache_path(state, district, lat, lon, start, end)
    # Load from cache when available.
    if cache_path.exists():
        try:
            cached = pd.read_csv(cache_path, parse_dates=["date"])
            # A cache with no rows is unusable; fall through to a re-fetch.
            if len(cached) > 0:
                return (state, district, cached, "cache")
        except Exception:
            # A corrupt cache is simply overwritten below.
            pass
    # Prepare the NASA POWER query.
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "latitude": lat,
        "longitude": lon,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    # Try the request several times with backoff.
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(NASA_POWER_URL, params=params, timeout=180)
            response.raise_for_status()
            # Parse the real payload.
            daily = _parse_nasa_payload(response.json())
            # Refuse to cache an empty result.
            if daily.empty:
                raise ValueError("NASA POWER returned no daily rows")
            # Add the coordinate columns for traceability.
            daily["latitude"] = lat
            daily["longitude"] = lon
            # Save compressed to keep the cache small.
            daily.to_csv(cache_path, index=False, compression="gzip")
            # Return the frame.
            return (state, district, daily, "fetch")
        except Exception as exc:
            # Remember the error and back off before retrying.
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    # Return a clear failure marker after all retries.
    return (state, district, None, f"error: {last_error}")


def season_bounds(season: str, year: int) -> tuple:
    """Return the inclusive (start_date, end_date) for a season and crop year."""
    # Look up the canonical window.
    window = SEASON_WINDOWS[season]
    # Build the start date in the crop year.
    start = date(year, window["start"][0], window["start"][1])
    # Cross-year seasons end in the following calendar year.
    end_year = year + 1 if window["crosses_year"] else year
    # Build the end date.
    end = date(end_year, window["end"][0], window["end"][1])
    # Return both.
    return start, end


def compute_season_aggregates(daily_by_district: dict, seasons: list, years: list) -> pd.DataFrame:
    """Aggregate daily weather into one row per (state, district, year, season)."""
    # Accumulate output rows.
    rows = []
    # Process each district independently.
    for (state, district), daily in daily_by_district.items():
        # Index by date so slicing is fast and vectorised.
        indexed = daily.set_index("date").sort_index()
        # Loop over years and seasons (small: ~14 years x 6 seasons).
        for year in years:
            for season in seasons:
                # Compute the exact season window.
                start, end = season_bounds(season, year)
                # Slice using label-based indexing (inclusive).
                window = indexed.loc[str(start):str(end)]
                # Skip windows with no observations.
                if window.empty:
                    continue
                # Mean temperature across the season.
                temp_mean = window["temp_mean"].mean()
                # Total rainfall across the season.
                rain_total = window["rain_total"].sum(min_count=1)
                # Keep only usable aggregates.
                if pd.isna(temp_mean) or pd.isna(rain_total):
                    continue
                # Append the aggregate row.
                rows.append({
                    "state": state,
                    "district": district,
                    "year": int(year),
                    "season": season,
                    "temp_mean": float(temp_mean),
                    "rain_total": float(rain_total),
                })
    # Return a tidy DataFrame.
    return pd.DataFrame(rows)


def main() -> pd.DataFrame:
    """Run Step 2 end to end."""
    # Banner.
    print("=" * 80)
    print("STEP 2: WEATHER FEATURES (NASA POWER)")
    print("=" * 80)
    # Verify that Step 1 produced its output.
    if not PREPARED_PATH.exists():
        print(f"[ERROR] Prepared data not found: {PREPARED_PATH}")
        print("[ACTION] Run: python scripts/01_prepare_data.py")
        sys.exit(1)
    # Load the prepared data.
    df = pd.read_csv(PREPARED_PATH)
    print(f"[INFO] Loaded {len(df):,} prepared rows")
    # Normalise the season column defensively.
    df["season"] = df["season"].map(canonical_season)
    # Remove a Total season if an old prepared file still contains it.
    if (df["season"] == "Total").any():
        dropped = int((df["season"] == "Total").sum())
        df = df[df["season"] != "Total"].copy()
        print(f"[INFO] Dropped {dropped} legacy 'Total' rows")
    # Discover the seasons actually present.
    seasons = sorted(df["season"].unique().tolist())
    unknown = [s for s in seasons if s not in SEASON_WINDOWS]
    if unknown:
        print(f"[ERROR] No season window defined for: {unknown}")
        sys.exit(1)
    print(f"[INFO] Seasons: {seasons}")
    # Attach real coordinates.
    coords = build_coordinates(df)
    # Merge coordinates onto the crop rows.
    df = df.merge(
        coords[["state", "district", "latitude", "longitude", "coord_source"]],
        on=["state", "district"],
        how="left",
    )
    # Report any pair that still lacks coordinates.
    missing_coords = int(df["latitude"].isna().sum())
    if missing_coords:
        print(f"[WARN] Dropping {missing_coords} rows without coordinates")
        df = df.dropna(subset=["latitude", "longitude"]).copy()
    # Determine the real year range from the crop data.
    min_year = int(df["year"].min())
    max_year = int(df["year"].max())
    # Extend the end date so Rabi/Winter of the final year are complete.
    start_date = date(min_year, 1, 1)
    end_date = date(max_year + 1, 3, 31)
    print(f"[INFO] Weather range: {start_date} to {end_date}")
    # Build the unique district list with coordinates.
    districts = (
        df[["state", "district", "latitude", "longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"[INFO] Fetching daily weather for {len(districts)} state-district pairs")
    # Fetch concurrently.
    daily_by_district = {}
    failures = []
    completed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit one job per district.
        futures = {
            executor.submit(
                fetch_district_daily,
                row["state"], row["district"], float(row["latitude"]), float(row["longitude"]),
                start_date, end_date,
            ): (row["state"], row["district"])
            for _, row in districts.iterrows()
        }
        # Collect results as they finish.
        for future in as_completed(futures):
            state, district, daily, status = future.result()
            completed += 1
            # Store successful results.
            if daily is not None:
                daily_by_district[(state, district)] = daily
            else:
                failures.append((state, district, status))
            # Print progress every 25 districts.
            if completed % 25 == 0 or completed == len(districts):
                print(f"  Progress: {completed}/{len(districts)} | cached/fetched OK: "
                      f"{len(daily_by_district)} | failed: {len(failures)}")
    # Fail loudly if nothing could be fetched: never invent weather values.
    if not daily_by_district:
        print("\n[ERROR] NO REAL WEATHER COULD BE FETCHED.")
        print("[INFO] This script will NOT fabricate weather values.")
        print("[INFO] Check your internet connection or NASA POWER availability, then re-run.")
        for state, district, status in failures[:10]:
            print(f"  - {state} / {district}: {status}")
        sys.exit(1)
    # Report partial failures honestly.
    if failures:
        print(f"\n[WARN] {len(failures)} districts failed; their crop rows will be dropped")
        for state, district, status in failures[:15]:
            print(f"  - {state} / {district}: {status}")
    # Compute season aggregates from the real daily data.
    print("\n[INFO] Aggregating daily weather into season windows")
    aggregates = compute_season_aggregates(
        daily_by_district,
        seasons=seasons,
        years=list(range(min_year, max_year + 1)),
    )
    print(f"[INFO] Built {len(aggregates):,} district-year-season weather aggregates")
    # Merge the aggregates onto the crop rows.
    merged = df.merge(
        aggregates[["state", "district", "year", "season", "temp_mean", "rain_total"]],
        on=["state", "district", "year", "season"],
        how="left",
    )
    # Count rows that could not be matched to weather.
    missing_weather = int(merged["temp_mean"].isna().sum())
    if missing_weather:
        print(f"[WARN] Dropping {missing_weather} rows with no matching weather window")
        # Show which seasons lose the most rows.
        lost = merged[merged["temp_mean"].isna()]["season"].value_counts()
        print(lost.to_string())
        merged = merged.dropna(subset=["temp_mean", "rain_total"]).copy()
    # Reorder and keep only the columns the rest of the pipeline needs.
    final = merged[[
        "state", "district", "year", "season", "crop",
        "area_ha", "production_t", "yield_t_ha",
        "temp_mean", "rain_total", "latitude", "longitude", "coord_source",
    ]].copy()
    # Round weather values for readability (not for modelling accuracy).
    final["temp_mean"] = final["temp_mean"].round(3)
    final["rain_total"] = final["rain_total"].round(2)
    # Print a final summary.
    print("\n" + "=" * 80)
    print("WEATHER FEATURE SUMMARY")
    print("=" * 80)
    print(f"[INFO] Rows: {len(final):,}")
    print(f"[INFO] State-district pairs: {final[['state', 'district']].drop_duplicates().shape[0]}")
    print(f"[INFO] Years: {int(final['year'].min())} to {int(final['year'].max())}")
    print(f"[INFO] Temperature range: {final['temp_mean'].min():.1f} to {final['temp_mean'].max():.1f} C")
    print(f"[INFO] Seasonal rainfall range: {final['rain_total'].min():.0f} to {final['rain_total'].max():.0f} mm")
    print(f"[INFO] Coordinate sources: {final['coord_source'].value_counts().to_dict()}")
    print("=" * 80)
    # Save the real, non-synthetic output.
    final.to_csv(OUTPUT_PATH, index=False)
    print(f"[INFO] Saved {OUTPUT_PATH}")
    # Return the frame.
    return final


# Run only when executed as a script.
if __name__ == "__main__":
    main()
