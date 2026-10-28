#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main FastAPI application for the India Crop Recommendation System.
Serves the JSON API under /api and the static frontend at /.
"""

# Import os for environment variables.
import os
# Import sys so the project root can be added to the import path.
import sys
# Import json for reading report metadata.
import json
# Import Path for filesystem paths.
from pathlib import Path
# Import typing helpers.
from typing import List, Optional, Dict, Any
# Import datetime for response timestamps.
from datetime import datetime

# Import numpy for medians, percentiles, and spreads.
import numpy as np
# Import requests for the optional Gemini narrative call.
import requests
# Import FastAPI primitives.
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

# Import PostgreSQL adapter and dict cursor.
import psycopg2
from psycopg2 import pool as psycopg2_pool
from psycopg2.extras import RealDictCursor
# Import threading for the pooled-connection and reason caches.
import threading

# Load environment variables from .env.
from dotenv import load_dotenv
load_dotenv()

# Resolve the project root from this file, then make the "backend" package
# importable. This keeps BOTH `uvicorn backend.main:app` (run from the project
# root) and `python backend/main.py` (run from anywhere) working.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the two service layers using the package-qualified path.
from backend.services.weather_service import WeatherService
from backend.services.model_service import ModelService

# Resolve project paths.
FRONTEND_DIR = PROJECT_ROOT / "frontend"
MODELS_DIR = BACKEND_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create the FastAPI application.
app = FastAPI(
    title="India Crop Recommendation System",
    description="District + season crop suitability using three ML models and a weighted ensemble",
    version="1.0.0",
)

# Allow the frontend to call the API from any origin during development.
# Credentials are disabled because "*" and credentialed requests are an
# invalid combination in the CORS specification.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialise the weather service (live conditions + 7-day forecast only).
weather_service = WeatherService()

# Initialise the model service, tolerating a first run before training exists.
try:
    model_service = ModelService(MODELS_DIR)
    print("[OK] Model service initialised")
except Exception as exc:
    print(f"[WARNING] Model service failed to initialise: {exc}")
    model_service = None


# Hold one shared connection pool so requests do not pay a fresh TCP+TLS
# handshake to the hosted database every time.
db_pool = None
# Guard lazy pool creation against concurrent first requests.
db_pool_lock = threading.Lock()


def get_db_connection():
    """Borrow a PostgreSQL connection from the shared pool."""
    # The pool needs the connection string.
    global db_pool
    # Read the connection string from the environment.
    database_url = os.getenv("DATABASE_URL")
    # Fail clearly when the variable is missing.
    if not database_url:
        raise HTTPException(status_code=500, detail="DATABASE_URL is not configured")
    # Create the pool once, with one warm connection and a small ceiling.
    if db_pool is None:
        with db_pool_lock:
            if db_pool is None:
                try:
                    db_pool = psycopg2_pool.ThreadedConnectionPool(
                        1, 5, database_url, cursor_factory=RealDictCursor
                    )
                except Exception as exc:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Database connection failed: {exc}",
                    )
    # Borrow a connection.
    try:
        conn = db_pool.getconn()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {exc}")
    # Discard connections the server has already closed.
    if conn.closed:
        db_pool.putconn(conn, close=True)
        conn = db_pool.getconn()
    # This API only reads, so autocommit avoids an extra COMMIT/ROLLBACK
    # round-trip to the hosted database on every request.
    if not conn.autocommit:
        conn.autocommit = True
    # Return the usable connection.
    return conn


def release_db_connection(conn):
    """Return a connection to the pool, discarding broken ones."""
    # Nothing to do when the connection is already gone.
    if conn is None:
        return
    try:
        # An aborted transaction must be rolled back before reuse.
        if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
            conn.rollback()
        # Return the healthy connection to the pool.
        db_pool.putconn(conn)
    except Exception:
        # Never return a broken connection to the pool.
        try:
            db_pool.putconn(conn, close=True)
        except Exception:
            pass


class RecommendationRequest(BaseModel):
    """Request body for POST /api/recommend."""
    # The database id of the chosen district.
    district_id: int = Field(..., description="District id from /api/districts")
    # The canonical planting season.
    season: str = Field(..., description="Planting season, e.g. Kharif or Rabi")


def make_district_key(state: str, district: str) -> str:
    """Build the location key used during training (state | district)."""
    # The separator matches scripts/03_train_models.py exactly.
    return f"{state} | {district}"


# Cache rewritten reasons so repeat requests do not pay for a second LLM call.
gemini_cache: Dict[str, List[str]] = {}
# Guard the cache across request threads.
gemini_cache_lock = threading.Lock()


def generate_gemini_reasons(prompt: str, fallback: List[str]) -> List[str]:
    """
    Optionally rewrite computed reasons in natural language with Gemini.

    This is optional and never blocks the recommendation: without an API key,
    or on any error, the deterministic computed reasons are returned.
    """
    # Read the optional key.
    api_key = os.getenv("GEMINI_API_KEY")
    # Narrative rewriting is opt-in: on a slow link it adds several seconds,
    # while the deterministic reasons below are always accurate and instant.
    enabled = os.getenv("GEMINI_ENABLED", "false").strip().lower() in {
        "1", "true", "yes", "on"
    }
    # Skip the call unless the feature is explicitly enabled with a key.
    if not api_key or not enabled:
        return fallback
    # Read the optional model name. The lite model answers in ~1-2 s, which
    # keeps the optional narrative from dominating request latency.
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    # Identical facts always produce the same prompt, so reuse the result.
    cache_key = f"{model_name}:{prompt}"
    with gemini_cache_lock:
        cached = gemini_cache.get(cache_key)
    if cached is not None:
        return cached
    # Build the REST endpoint.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    # Build a tightly scoped payload: only rewrite, never invent facts.
    payload = {
        "contents": [{
            "parts": [{
                "text": (
                    "Rewrite the following computed crop-recommendation facts as 3-5 short, "
                    "plain-English bullet points. Do not add any facts, numbers, or claims "
                    "that are not present. Do not mention profit or guarantees.\n\n" + prompt
                )
            }]
        }],
        # The lite model does not use a thinking pass. Keep the budget modest
        # because the request only needs 3-5 short bullets.
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 600,
        },
    }
    try:
        # Keep the optional call bounded so it cannot dominate request latency.
        # On a slow or overloaded API, the computed reasons are returned instead.
        response = requests.post(url, params={"key": api_key}, json=payload, timeout=10)
        response.raise_for_status()
        # Extract the generated text and finish reason.
        data = response.json()
        candidate = data["candidates"][0]
        # A truncated answer must never replace the complete computed reasons.
        if candidate.get("finishReason") != "STOP":
            return fallback
        text = candidate["content"]["parts"][0]["text"].strip()
        # Keep only explicit bullet lines so a preamble is not shown as a reason.
        bullets = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("-", "*", "•")):
                cleaned = stripped.lstrip("-*• ").replace("**", "").strip()
                if cleaned:
                    bullets.append(cleaned)
        # If the model ignored the bullet format, drop a one-line preamble.
        if not bullets:
            lines = [line.strip().replace("**", "") for line in text.splitlines() if line.strip()]
            if len(lines) > 1 and lines[0].endswith(":"):
                lines = lines[1:]
            bullets = lines
        # Require at least three useful bullets, otherwise keep the fallback.
        result = bullets[:5]
        if len(result) < 3:
            return fallback
        # Remember the result, keeping the cache small.
        with gemini_cache_lock:
            if len(gemini_cache) > 500:
                gemini_cache.clear()
            gemini_cache[cache_key] = result
        return result
    except Exception:
        # Any failure returns the deterministic computed reasons.
        return fallback


@app.get("/api/health")
async def health():
    """Simple health endpoint for local checks and Render."""
    # Report whether models are loaded without exposing secrets.
    return {
        "status": "ok",
        "models_loaded": model_service is not None,
        "database_configured": bool(os.getenv("DATABASE_URL")),
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/districts")
def get_districts(q: Optional[str] = Query(None, description="District search text")):
    """Return up to 15 Indian districts matching the search text."""
    # Open a connection.
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Use a parameterised ILIKE search; never interpolate user text.
        if q:
            cur.execute(
                """
                SELECT id, state, district, lat, lon
                FROM districts
                WHERE district ILIKE %s OR state ILIKE %s
                ORDER BY district, state
                LIMIT 15
                """,
                (f"%{q}%", f"%{q}%"),
            )
        else:
            cur.execute(
                """
                SELECT id, state, district, lat, lon
                FROM districts
                ORDER BY district, state
                LIMIT 15
                """
            )
        # Fetch and return the rows.
        districts = cur.fetchall()
        return {"districts": districts, "count": len(districts)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always release the cursor and return the connection to the pool.
        cur.close()
        release_db_connection(conn)


@app.get("/api/seasons")
def get_seasons():
    """Return the canonical seasons present in the database."""
    # Open a connection.
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Exclude the Total roll-up defensively.
        cur.execute(
            "SELECT DISTINCT season FROM crop_stats WHERE season <> 'Total' ORDER BY season"
        )
        # Return a simple list.
        return {"seasons": [row["season"] for row in cur.fetchall()]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        cur.close()
        release_db_connection(conn)


@app.post("/api/recommend")
def recommend_crops(request: RecommendationRequest):
    """
    Recommend and rank crops for one district and season.

    Ranking is by suitability = predicted yield / that crop's own historical
    median. Raw yields are never compared across different crops.
    """
    # Refuse to serve recommendations before training has run.
    if model_service is None:
        raise HTTPException(status_code=500, detail="Models are not loaded. Run Step 3 first.")
    # Open a pooled connection.
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Load the chosen district and the dataset's latest year in one query.
        cur.execute(
            """
            SELECT d.*, (SELECT MAX(year) FROM crop_stats) AS max_year
            FROM districts d
            WHERE d.id = %s
            """,
            (request.district_id,),
        )
        district_info = cur.fetchone()
        # 404 when the id is unknown.
        if not district_info:
            raise HTTPException(status_code=404, detail="District not found")
        # Pull out the fields used below.
        district_name = district_info["district"]
        state_name = district_info["state"]
        lat = district_info["lat"]
        lon = district_info["lon"]
        # The latest year comes from the same round-trip.
        max_year = district_info["max_year"]
        # Build the same location key used during training.
        district_key = make_district_key(state_name, district_name)

        # Fetch live conditions and the 7-day forecast for display only.
        live_weather = weather_service.get_live_weather_and_forecast(lat, lon)

        # Fetch every historical row for this district-season in ONE query.
        # Yields, per-year weather averages, and candidate eligibility are all
        # derived from this result, avoiding four extra cloud round-trips.
        cur.execute(
            """
            SELECT crop, year, yield_t_ha, temp_mean, rain_total
            FROM crop_stats
            WHERE district_id = %s AND season = %s
            ORDER BY crop, year
            """,
            (request.district_id, request.season),
        )
        rows = cur.fetchall()
        # Refuse when there is no history at all.
        if not rows:
            raise HTTPException(
                status_code=404,
                detail="No historical weather for this district and season",
            )

        # Group yields, distinct years, and weather by crop/year in one pass.
        history_by_crop: Dict[str, List[float]] = {}
        years_by_crop: Dict[str, set] = {}
        temp_by_year: Dict[int, List[float]] = {}
        rain_by_year: Dict[int, List[float]] = {}
        for row in rows:
            crop_name = row["crop"]
            year = int(row["year"])
            history_by_crop.setdefault(crop_name, []).append(float(row["yield_t_ha"]))
            years_by_crop.setdefault(crop_name, set()).add(year)
            temp_by_year.setdefault(year, []).append(float(row["temp_mean"]))
            rain_by_year.setdefault(year, []).append(float(row["rain_total"]))

        # Compute the model-input weather from the 5 most recent years.
        # Average within a year first, so duplicated per-crop rows cannot
        # weight a year more heavily than it should.
        years_used = sorted(temp_by_year, reverse=True)[:5]
        model_temp = float(np.mean([np.mean(temp_by_year[year]) for year in years_used]))
        model_rain = float(np.mean([np.mean(rain_by_year[year]) for year in years_used]))
        # Label the estimate honestly. Some district-season pairs end before the
        # dataset's final year, so say "most recent available", not "last 5 years".
        if len(years_used) == 5:
            weather_note = (
                "Seasonal estimate: average of the 5 most recent years available "
                "for this district and season, not a forecast"
            )
        else:
            weather_note = (
                f"Seasonal estimate: average of {len(years_used)} available year(s) "
                "for this district and season, not a forecast"
            )

        # Candidate crops need at least 3 DISTINCT years of history.
        candidates = [
            {"crop": crop_name, "year_count": len(years_by_crop[crop_name])}
            for crop_name in sorted(history_by_crop)
            if len(years_by_crop[crop_name]) >= 3
        ]
        # 404 when nothing qualifies.
        if not candidates:
            raise HTTPException(
                status_code=404,
                detail="No candidate crops with at least 3 distinct years of history",
            )

        # Predict every candidate crop in one batched call per model.
        batch_predictions = model_service.predict_many(
            district_key,
            request.season,
            [candidate["crop"] for candidate in candidates],
            model_temp,
            model_rain,
        )

        # Build one recommendation per candidate crop.
        recommendations = []
        for candidate in candidates:
            crop_name = candidate["crop"]
            # Read this crop's predictions from the batch.
            values = batch_predictions.get(crop_name)
            # Skip a crop if the encoder has never seen it.
            if values is None:
                continue
            # Historical distribution for this crop in this district-season.
            hist_yields = history_by_crop.get(crop_name, [])
            if not hist_yields:
                continue
            hist_median = float(np.median(hist_yields))
            hist_p25 = float(np.percentile(hist_yields, 25))
            hist_p75 = float(np.percentile(hist_yields, 75))
            num_years = len(years_by_crop.get(crop_name, hist_yields))
            # Use the ensemble as the headline number.
            ensemble_pred = float(values.get("Ensemble", 0.0))
            # Suitability is relative to this crop's own median, never raw yield.
            suitability = (ensemble_pred / hist_median * 100.0) if hist_median > 0 else 0.0

            # Agreement is the coefficient of variation across the three models.
            model_values = [
                float(values.get("RandomForest", 0.0)),
                float(values.get("XGBoost", 0.0)),
                float(values.get("HistGradientBoosting", 0.0)),
            ]
            mean_value = float(np.mean(model_values))
            std_value = float(np.std(model_values))
            cv = (std_value / mean_value * 100.0) if mean_value > 0 else 999.0
            # Map the CV onto a simple label.
            if cv < 10:
                agreement = "High"
            elif cv < 20:
                agreement = "Medium"
            else:
                agreement = "Low"

            # Build deterministic, computed reasons first.
            reasons = [
                f"Recorded in {request.season} season in {district_name}, {state_name} "
                f"for {num_years} distinct years",
                f"Seasonal temperature estimate: {model_temp:.1f} C "
                f"(average of {len(years_used)} years)",
                f"Seasonal rainfall estimate: {model_rain:.1f} mm "
                f"(average of {len(years_used)} years)",
                f"Predicted yield {ensemble_pred:.2f} t/ha versus this crop's "
                f"historical median of {hist_median:.2f} t/ha",
                f"Model agreement: {agreement} (coefficient of variation {cv:.1f}%)",
            ]
            # Append the recommendation payload.
            recommendations.append({
                "crop": crop_name,
                "expected_yield": round(ensemble_pred, 3),
                "suitability": round(float(suitability), 2),
                "models": {
                    "RandomForest": round(float(values.get("RandomForest", 0.0)), 3),
                    "XGBoost": round(float(values.get("XGBoost", 0.0)), 3),
                    "HistGradientBoosting": round(float(values.get("HistGradientBoosting", 0.0)), 3),
                    "Ensemble": round(ensemble_pred, 3),
                },
                "agreement": agreement,
                "historical_median": round(hist_median, 3),
                "historical_range": {
                    "p25": round(hist_p25, 3),
                    "p75": round(hist_p75, 3),
                },
                "years_of_evidence": num_years,
                "reasons": reasons,
            })

        # Rank by suitability, highest first.
        recommendations.sort(key=lambda item: item["suitability"], reverse=True)
        top_five = recommendations[:5]

        # Optionally rewrite ONLY the top crop's reasons, once per request.
        # Calling Gemini for every crop would add up to five blocking calls.
        if os.getenv("GEMINI_API_KEY") and top_five:
            top_five[0]["reasons"] = generate_gemini_reasons(
                "\n".join(top_five[0]["reasons"]), top_five[0]["reasons"]
            )

        # Build the limitations list from real metadata.
        model_info = model_service.get_model_info()
        limitations = [
            f"Historical data ends at {max_year}",
            "District-level averages, not field-level predictions",
            "No soil, irrigation, price, or satellite data is used",
            "Weather input is a seasonal estimate, not a forecast",
            "Suitability is based on history; it is not a profit estimate or a guarantee",
        ]
        # Warn when models disagree on any top crop.
        if any(item["agreement"] == "Low" for item in top_five):
            limitations.append("Low-confidence warning: models disagree on at least one top crop")
        # Warn when evidence is thin.
        if any(item["years_of_evidence"] < 5 for item in top_five):
            limitations.append("Low-confidence warning: fewer than 5 years of history for at least one top crop")

        # Return the complete response.
        return {
            "district": {
                "id": district_info["id"],
                "name": district_name,
                "state": state_name,
            },
            "season": request.season,
            "weather_used": {
                "model_input": {
                    "temperature": round(model_temp, 2),
                    "rainfall": round(model_rain, 2),
                    "years_used": years_used,
                    "note": weather_note,
                },
                "live_conditions": live_weather.get("current"),
                "forecast_7day": live_weather.get("forecast", []),
            },
            "recommendations": top_five,
            "limitations": limitations,
            "model_info": {
                "weights": model_service.weights,
                "test_metrics": model_info.get("metrics", {}),
                "val_maes": model_info.get("val_maes", {}),
                "test_years": model_info.get("test_years", []),
            },
        }
    except HTTPException:
        # Preserve deliberate 4xx responses.
        raise
    except Exception as exc:
        # Convert unexpected failures into a 500 with a readable message.
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        # Always release database resources.
        cur.close()
        release_db_connection(conn)


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the single-page frontend."""
    # Read the index file when present.
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    # Friendly message when the frontend is missing.
    return HTMLResponse(content="<h1>Frontend not found</h1>")


# Mount static assets (CSS and JavaScript).
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# Allow `python backend/main.py` for quick local runs.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



