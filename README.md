# India Crop Recommendation System

A standalone web app that recommends crops for an Indian district and planting
season. The user picks a district (searchable) and a season; the app fetches
weather automatically, runs three ML models plus a weighted ensemble, and
explains each recommendation with computed facts.

The system is deliberately honest: it reports test metrics, shows the baseline,
and warns when the data is thin or the models disagree.

## What it does

- India only, district-level.
- User input is just **district + season**.
- Historical model weather comes from **NASA POWER** (real daily observations,
  no API key). Live conditions and a 7-day forecast come from **Open-Meteo**
  and are displayed only, never used as model features.
- Three regressors: **RandomForest**, **XGBoost**, **HistGradientBoosting**.
- One **weighted ensemble** whose weights come from validation MAE.
- Every model's own prediction is shown in an expandable sidebar, together
  with its test R2 / MAE / RMSE and ensemble weight. The historical baseline is
  shown too.
- Ranking uses **suitability = predicted yield / that crop's own historical
  median**, so a low-yield crop is never penalised just for having a lower
  absolute yield than another crop.

## Tech stack

- Python 3.12, FastAPI, Uvicorn
- Plain HTML, CSS, JavaScript (no frontend framework)
- pandas, numpy, scikit-learn, XGBoost, joblib
- PostgreSQL (Neon or any Postgres)
- NASA POWER (historical weather), Open-Meteo (live weather)
- matplotlib (training comparison chart)

## Project structure

```
crop-recommendation-india/
|-- backend/
|   |-- main.py                      # FastAPI app: JSON API + static frontend
|   |-- services/
|   |   |-- model_service.py         # loads models, encodes inputs, ensembles
|   |   `-- weather_service.py       # live weather + 7-day forecast
|   `-- models/                      # trained artefacts (generated)
|-- frontend/
|   |-- index.html
|   |-- style.css
|   `-- app.js
|-- scripts/
|   |-- 01_prepare_data.py           # clean, de-duplicate, winsorise
|   |-- 02_fetch_weather.py          # coordinates + NASA POWER + season aggregates
|   |-- 03_train_models.py           # tune, train, evaluate, save
|   `-- 04_load_database.py          # create tables, indexes, bulk load
|-- data/                            # datasets + caches (generated)
|-- reports/                         # metrics.json, comparison report, chart
|-- tests/                           # pytest unit tests
|-- requirements.txt
|-- .env.example
`-- README.md
```

## Data used

- Source: `data/district_apy.csv` (district-level area, production, yield).
- Raw rows: 177,600; years 1998-99 through 2011-12.
- After cleaning: **153,374 rows**, **666 state-district pairs**, **48 crops**,
  **6 seasons**.
- The `Total` season is a Kharif + Rabi roll-up and is excluded (5,113 rows).
- Coconut is excluded (1,380 rows) because the source reports production in
  nuts rather than tonnes, so derived t/ha is meaningless.
- Duplicate business keys are collapsed by summing area and production and
  recomputing yield.
- Yields are winsorised at the 1st/99th percentile within each crop, with a
  hard physical sanity bound of 250 t/ha.
- Coordinates: 642 of 666 pairs are matched to real district boundaries
  (GADM-level GeoJSON). The remaining 24 use the real **state centroid** and
  are labelled `state_fallback` in `data/district_coords.csv`.
- Weather: **666 of 666** pairs fetched successfully from NASA POWER (zero
  failures), producing 55,944 district-year-season aggregates. No row had to be
  dropped for missing weather.
- Temperature range: -16.0 C (Lahaul & Spiti, Rabi) to 35.2 C (Ganganagar,
  Kharif). Seasonal rainfall range: 0 mm to 4,152 mm (Kasaragod, whole year).
- Split sizes: 122,383 train rows, 23,429 validation rows, 7,562 test rows.

### Data limits (shown in the UI as well)

- History ends in **2011-12**; recommendations describe historical suitability,
  not the present-day economy.
- District-level averages, not field-level predictions.
- No soil type, irrigation, fertiliser, price, market access, or satellite data.
- Weather input is a **seasonal estimate from the 5 most recent years available**
  for that district and season, not a forecast.
- The model does not predict profit.

## Pipeline

### Step 1 - Prepare data

```bash
python scripts/01_prepare_data.py
```

- Selects `district_apy.csv` explicitly (never re-reads its own output).
- Drops the `Total` roll-up and the Coconut unit mismatch.
- Collapses duplicate keys and winsorises outliers.
- Writes `data/crop_data_prepared.csv`.

### Step 2 - Build weather features

```bash
python scripts/02_fetch_weather.py
```

- Loads or builds `data/district_coords.csv` from a real India district
  boundary file.
- Fetches real daily `T2M` and `PRECTOTCORR` from NASA POWER for 1998-01-01 to
  2012-03-31 (the extended end date makes Rabi/Winter of 2011 complete).
- Caches one compressed daily file per state-district pair in
  `data/weather_cache/`, so the script is resumable.
- Aggregates into season-level mean temperature and total rainfall.
- **Never fabricates weather.** If no real data can be fetched, it stops with an
  explicit error.
- Writes `data/crop_data_with_weather.csv`.

### Step 3 - Train and compare models

```bash
python scripts/03_train_models.py
```

- Location feature is `state | district`, so same-named districts in different
  states stay distinct.
- Chronological split: last 2 years = test, prior 2 = validation, rest = train.
- Small hyper-parameter grid tuned on validation years.
- Target is `log1p(yield_t_ha)`; predictions are converted back with `expm1`.
- Ensemble weights are inverse validation MAE.
- The script intentionally **stops here and prints the metrics table** before
  Step 4.

### Step 4 - Load PostgreSQL

```bash
python scripts/04_load_database.py
```

- Creates `districts` and `crop_stats`, plus query indexes.
- Default is a full refresh (truncate then load), so stale rows cannot survive.
- Use `python scripts/04_load_database.py --keep` to upsert instead.

## Model results

Real output from `reports/metrics.json` (test years **2010-2011**, 7,562 rows,
trained on 1998-2007 and validated on 2008-2009).

| Model | Test R2 | Test MAE (t/ha) | Test RMSE (t/ha) |
|---|---:|---:|---:|
| RandomForest | 0.9063 | 1.0915 | 3.5843 |
| XGBoost | 0.8908 | 1.1233 | 3.8702 |
| HistGradientBoosting | 0.8943 | 1.1059 | 3.8074 |
| **Ensemble** | **0.9035** | 1.0748 | **3.6366** |
| Baseline (district x crop median) | 0.9003 | **0.8587** | 3.6973 |

Ensemble weights (inverse validation MAE):

| Model | Validation MAE | Weight |
|---|---:|---:|
| RandomForest | 1.3998 | 0.3117 |
| XGBoost | 1.3288 | 0.3283 |
| HistGradientBoosting | 1.2119 | 0.3600 |

**Honest reading of this table:** the ensemble beats the historical-median
baseline on R2 (0.9035 vs 0.9003) and on RMSE (3.6366 vs 3.6973), but it **loses
on MAE** (1.0748 vs 0.8587 t/ha). In plain terms, the baseline is closer on a
typical prediction, while the ML models handle the large errors better. This is
reported rather than hidden, and it is a reason the UI shows both the model
predictions and the historical median.

## Local setup

1. Create a virtual environment and install dependencies.

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

2. Configure environment variables.

```bash
cp .env.example .env
```

Edit `.env` and set `DATABASE_URL`. `GEMINI_API_KEY` is optional; without it the
app uses deterministic computed reasons. Gemini wording is only used when
`GEMINI_ENABLED=true`; the default lite model often adds 1-3 seconds, and every
district/season pair is cached after the first request. If Gemini is slow or
unavailable, the same computed reasons are returned without an error.

3. Put `district_apy.csv` in `data/` (or keep the copy already there).

4. Run the pipeline.

```bash
python scripts/01_prepare_data.py
python scripts/02_fetch_weather.py
python scripts/03_train_models.py   # stops and prints metrics
python scripts/04_load_database.py
```

5. Start the app.

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## API

- `GET /api/health` - health and model-load status
- `GET /api/districts?q=<text>` - up to 15 matching districts
- `GET /api/seasons` - canonical seasons in the database
- `POST /api/recommend` with body:

```json
{
  "district_id": 1,
  "season": "Kharif"
}
```

## Recommendation logic

- Candidate crops need at least **3 distinct years** of history in that
  district-season.
- Suitability is relative to each crop's own historical median.
- Model agreement uses the coefficient of variation across the three models:
  High < 10%, Medium 10-20%, Low > 20%.
- Model-input weather is the per-year average of the 5 most recent years
  available for that district and season, not a forecast.
- Low-confidence warnings are added when models disagree or evidence is thin.

## Performance

Measured on this machine with the hosted PostgreSQL database:

| Operation | Time |
|---|---:|
| `POST /api/recommend`, first call after start | ~2.0 s (opens the DB pool) |
| `POST /api/recommend`, subsequent calls | ~0.7-1.0 s |
| Gemini narrative rewrite, cache miss | ~1-3 s typical; capped at 10 s |
| Live weather on a cache miss | ~1-2 s (cached for 10 minutes) |
| App startup | ~30 s (loads the 30.5 MB of models) |

What keeps it fast:

- All candidate crops are scored in **one batched prediction per model**, not
  one call per crop.
- PostgreSQL connections come from a **pool** with autocommit enabled, so a
  request does not pay a fresh TLS handshake or a COMMIT round-trip.
- Yields, per-year weather, and candidate filtering are derived from **one
  query** per district-season.
- Live weather and Gemini narratives are cached per district/season; Gemini
  rewriting is **opt-in** and best-effort, so a slow or failed call falls back
  to the computed reasons.
- Models load once at startup, not per request.

## Tests

```bash
pytest tests/ -v
```

The tests are self-contained: no database, models, or internet required. They
cover season windows, cross-year boundaries, suitability ranking, distinct-year
candidate filtering, agreement thresholds, and district-key collisions.

## Deploy on Render

Build command:

```bash
pip install -r requirements.txt
```

Start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Environment variables:

- `DATABASE_URL` (required)
- `GEMINI_API_KEY` (optional)
- `GEMINI_MODEL` (optional, default `gemini-3.5-flash-lite`)
- `GEMINI_ENABLED` (optional, default `false`; set `true` to rewrite reasons)
- `PYTHON_VERSION=3.12`

Notes:

- The free tier has limited RAM. Models are compressed and loaded once.
- Run the pipeline (Steps 1-4) before the first deploy, or run it once in a
  Render shell. Model files and generated CSVs are intentionally gitignored.
- On the free tier, keep `MAX_WORKERS` in `02_fetch_weather.py` modest.

## Security

- `.env` is gitignored and must never be committed.
- `.env.example` contains placeholders only.
- If a real database password was ever pasted into chat, in a log, or in a
  committed file, rotate it in the database provider immediately.

## Honest caveats

- The dataset stops at 2011-12. This is a historical suitability demo, not a
  current advisory service.
- District-level averages hide within-district variation.
- Weather is a seasonal estimate, not a forecast.
- No economic modelling: suitability is not profit.
- If the ML ensemble loses to the simple baseline, the app says so.

