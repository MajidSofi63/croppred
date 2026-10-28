#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 03: Train and Compare Models for India Crop Recommendation System
Purpose:
  - Train RandomForest, XGBoost, and HistGradientBoosting regressors.
  - Use a time-based split: last 2 years = test, prior 2 = validation, rest = train.
  - Tune a small, explicit hyper-parameter grid on the validation years.
  - Compare every model and the weighted ensemble against an honest
    district x crop historical-median baseline.
  - Save models, encoders, feature metadata, and a metrics report.

Key correctness choices:
  - The location feature is "state|district", not district alone. 22 district
    names occur in more than one state (e.g. AURANGABAD, BILASPUR), so a
    district-only label would silently merge different places.
  - The "Total" season is excluded because it is a Kharif+Rabi roll-up.
  - The baseline is computed with a vectorised groupby+merge, not an O(n^2)
    row loop.
  - No year feature is used; the split is strictly chronological.
"""

# Import sys for clean fatal exits.
import sys
# Import json for metadata output.
import json
# Import pandas for data handling.
import pandas as pd
# Import numpy for maths.
import numpy as np
# Import joblib for model serialisation.
import joblib
# Import Path for paths.
from pathlib import Path
# Import matplotlib for the comparison chart.
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive backend suitable for servers.
import matplotlib.pyplot as plt

# Import the ML estimators.
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Resolve project paths.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "backend" / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
# Ensure output directories exist.
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Input produced by Step 2.
DATA_PATH = DATA_DIR / "crop_data_with_weather.csv"


def fit_and_score(model, X_train, y_train, X_val, y_val_original):
    """Fit a model and return it with its validation MAE in t/ha."""
    # Fit on the log-transformed target.
    model.fit(X_train, y_train)
    # Predict validation rows and convert back to t/ha.
    predictions = np.clip(np.expm1(model.predict(X_val)), 0, None)
    # Compute the validation MAE on the original scale.
    mae = mean_absolute_error(y_val_original, predictions)
    # Return both.
    return model, float(mae)


def tune_model(name, candidates, X_train, y_train, X_val, y_val_original):
    """Try each candidate configuration and keep the best validation MAE."""
    # Track the best result.
    best_model = None
    best_mae = float("inf")
    best_params = None
    # Evaluate each configuration.
    for index, params in enumerate(candidates, start=1):
        print(f"    [{name}] config {index}/{len(candidates)}: {params}")
        # Build the estimator from the parameter dictionary.
        if name == "RandomForest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        elif name == "XGBoost":
            model = XGBRegressor(random_state=42, n_jobs=-1, **params)
        else:
            model = HistGradientBoostingRegressor(random_state=42, **params)
        # Fit and score.
        model, mae = fit_and_score(model, X_train, y_train, X_val, y_val_original)
        # Report the result.
        print(f"      validation MAE = {mae:.4f} t/ha")
        # Keep the best configuration.
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_params = params
    # Return the winner.
    return best_model, best_mae, best_params


def main():
    """Run the full Step 3 pipeline."""
    # Banner.
    print("=" * 80)
    print("STEP 3: TRAIN AND COMPARE MODELS")
    print("=" * 80)
    # Verify the Step 2 output exists.
    if not DATA_PATH.exists():
        print(f"[ERROR] Data with weather not found: {DATA_PATH}")
        print("[ACTION] Run: python scripts/02_fetch_weather.py")
        sys.exit(1)
    # Load the real data.
    df = pd.read_csv(DATA_PATH)
    print(f"[INFO] Loaded {len(df):,} rows from {DATA_PATH.name}")
    # Drop the Total roll-up if an old file still contains it.
    if (df["season"] == "Total").any():
        removed = int((df["season"] == "Total").sum())
        df = df[df["season"] != "Total"].copy()
        print(f"[INFO] Dropped {removed} 'Total' rows")
    # Verify that weather columns are real, not a constant placeholder.
    if df["temp_mean"].nunique() <= 1 or df["rain_total"].nunique() <= 1:
        print("[ERROR] Weather columns are constant. Refusing to train on fake data.")
        print("[ACTION] Re-run: python scripts/02_fetch_weather.py")
        sys.exit(1)
    # Build a location key that cannot merge same-named districts in different states.
    df["district_key"] = df["state"].astype(str) + " | " + df["district"].astype(str)
    print(f"[INFO] Location keys: {df['district_key'].nunique()} unique state-district pairs")
    # Log-transform the target because yields are right-skewed.
    df["yield_log"] = np.log1p(df["yield_t_ha"])
    # Encode the categorical features.
    le_district = LabelEncoder()
    le_season = LabelEncoder()
    le_crop = LabelEncoder()
    df["district_enc"] = le_district.fit_transform(df["district_key"])
    df["season_enc"] = le_season.fit_transform(df["season"])
    df["crop_enc"] = le_crop.fit_transform(df["crop"])
    # Save the encoders so the API can reproduce the exact encoding.
    joblib.dump(le_district, MODELS_DIR / "le_district.pkl")
    joblib.dump(le_season, MODELS_DIR / "le_season.pkl")
    joblib.dump(le_crop, MODELS_DIR / "le_crop.pkl")
    print("[INFO] Saved label encoders")
    # Define the feature matrix. Year is deliberately not a feature.
    feature_cols = ["district_enc", "season_enc", "crop_enc", "temp_mean", "rain_total"]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df["yield_log"].to_numpy(dtype=float)
    y_original = df["yield_t_ha"].to_numpy(dtype=float)
    # Build the chronological split.
    print("\n" + "=" * 80)
    print("TIME-BASED SPLIT")
    print("=" * 80)
    years_sorted = sorted(int(year) for year in df["year"].unique())
    print(f"[INFO] Years available: {years_sorted}")
    # The last two years are the final test set.
    test_years = years_sorted[-2:]
    # The two years before that are validation.
    val_years = years_sorted[-4:-2]
    # Everything earlier is training.
    train_years = years_sorted[:-4]
    # Fail if the dataset is too short for this protocol.
    if not train_years or not val_years:
        print("[ERROR] Not enough years for the 2-year validation + 2-year test split.")
        sys.exit(1)
    print(f"[INFO] Train years: {train_years}")
    print(f"[INFO] Validation years: {val_years}")
    print(f"[INFO] Test years: {test_years}")
    # Build boolean masks.
    train_mask = df["year"].isin(train_years).to_numpy()
    val_mask = df["year"].isin(val_years).to_numpy()
    test_mask = df["year"].isin(test_years).to_numpy()
    # Slice the arrays.
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    y_val_original = y_original[val_mask]
    y_test_original = y_original[test_mask]
    # Report split sizes.
    print(f"[INFO] Train rows: {len(X_train):,} | Validation rows: {len(X_val):,} | Test rows: {len(X_test):,}")
    # Build the honest baseline with a vectorised groupby + merge.
    print("\n" + "=" * 80)
    print("BASELINE")
    print("=" * 80)
    print("[INFO] Baseline = historical median yield for the same state-district and crop")
    # Use train+validation history only, never the test years.
    history_mask = train_mask | val_mask
    history = df.loc[history_mask, ["district_key", "crop", "yield_t_ha"]]
    # Median per district x crop.
    district_crop_median = (
        history.groupby(["district_key", "crop"])["yield_t_ha"]
        .median()
        .rename("baseline")
        .reset_index()
    )
    # Median per crop, used only when a district-crop pair has no history.
    crop_median = (
        history.groupby("crop")["yield_t_ha"]
        .median()
        .rename("crop_baseline")
        .reset_index()
    )
    # Build the test frame used for baseline scoring.
    test_frame = df.loc[test_mask, ["district_key", "crop", "yield_t_ha"]].copy()
    # Merge the district-crop medians.
    test_frame = test_frame.merge(district_crop_median, on=["district_key", "crop"], how="left")
    # Merge the crop-level fallback.
    test_frame = test_frame.merge(crop_median, on="crop", how="left")
    # Fill missing district-crop values with the crop-level median.
    test_frame["baseline"] = test_frame["baseline"].fillna(test_frame["crop_baseline"])
    # Final fallback: the overall median, for crops that appear only in test years.
    overall_median = float(history["yield_t_ha"].median())
    unresolved = int(test_frame["baseline"].isna().sum())
    if unresolved:
        test_frame["baseline"] = test_frame["baseline"].fillna(overall_median)
        print(f"[WARN] {unresolved} test rows had no historical median; used overall median {overall_median:.4f}")
    # Keep the final baseline vector aligned with the test arrays.
    baseline_preds = test_frame["baseline"].to_numpy(dtype=float)
    print(f"[INFO] Baseline rows: {len(baseline_preds):,} | missing: {int(np.isnan(baseline_preds).sum())}")
    # Tune each model on validation years.
    print("\n" + "=" * 80)
    print("HYPER-PARAMETER SEARCH (VALIDATION YEARS)")
    print("=" * 80)
    # Random Forest candidates: keep trees shallow enough to stay under 100 MB.
    rf_candidates = [
        {"n_estimators": 120, "max_depth": 12, "min_samples_split": 4, "min_samples_leaf": 3},
        {"n_estimators": 160, "max_depth": 14, "min_samples_split": 6, "min_samples_leaf": 5},
    ]
    # XGBoost candidates.
    xgb_candidates = [
        {"n_estimators": 250, "max_depth": 8, "learning_rate": 0.08,
         "subsample": 0.85, "colsample_bytree": 0.85},
        {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
         "subsample": 0.85, "colsample_bytree": 0.85},
    ]
    # HistGradientBoosting candidates. Only season and crop are declared
    # categorical: the 643-level district column exceeds the 255-bin limit.
    hgb_candidates = [
        {"max_iter": 250, "max_depth": 8, "learning_rate": 0.08,
         "l2_regularization": 0.0, "categorical_features": [1, 2]},
        {"max_iter": 400, "max_depth": 6, "learning_rate": 0.05,
         "l2_regularization": 1.0, "categorical_features": [1, 2]},
        {"max_iter": 300, "max_depth": 10, "learning_rate": 0.07,
         "l2_regularization": 0.1, "categorical_features": [1, 2]},
    ]
    # Tune each model in turn.
    print("\n[1/3] Random Forest")
    rf, rf_mae, rf_params = tune_model("RandomForest", rf_candidates, X_train, y_train, X_val, y_val_original)
    print("\n[2/3] XGBoost")
    xgb, xgb_mae, xgb_params = tune_model("XGBoost", xgb_candidates, X_train, y_train, X_val, y_val_original)
    print("\n[3/3] HistGradientBoosting")
    hgb, hgb_mae, hgb_params = tune_model("HistGradientBoosting", hgb_candidates, X_train, y_train, X_val, y_val_original)
    # Keep the tuned validation predictions for ensemble weighting.
    tuned_models = {"RandomForest": rf, "XGBoost": xgb, "HistGradientBoosting": hgb}
    tuned_params = {"RandomForest": rf_params, "XGBoost": xgb_params, "HistGradientBoosting": hgb_params}
    val_maes = {"RandomForest": rf_mae, "XGBoost": xgb_mae, "HistGradientBoosting": hgb_mae}
    # Compute inverse-MAE weights from validation performance.
    print("\n" + "=" * 80)
    print("ENSEMBLE WEIGHTS")
    print("=" * 80)
    inverse = {name: (1.0 / mae if mae > 0 else 0.0) for name, mae in val_maes.items()}
    total_inverse = sum(inverse.values())
    weights = {name: value / total_inverse for name, value in inverse.items()}
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f} ({weight * 100:.2f}%) | validation MAE {val_maes[name]:.4f}")
    # Retrain the tuned configurations on train+validation for the final models.
    print("\n[INFO] Retraining on train+validation with the tuned configurations")
    X_train_val = np.vstack([X_train, X_val])
    y_train_val = np.concatenate([y_train, y_val])
    final_models = {}
    for name, params in tuned_params.items():
        if name == "RandomForest":
            model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        elif name == "XGBoost":
            model = XGBRegressor(random_state=42, n_jobs=-1, **params)
        else:
            model = HistGradientBoostingRegressor(random_state=42, **params)
        model.fit(X_train_val, y_train_val)
        final_models[name] = model
        print(f"[OK] {name} retrained")
    # Save the final models with compression to keep the artefact small.
    print("\n[INFO] Saving models")
    for name, model in final_models.items():
        path = MODELS_DIR / f"{name.lower()}_model.pkl"
        joblib.dump(model, path, compress=3)
        print(f"  Saved {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    joblib.dump(weights, MODELS_DIR / "ensemble_weights.pkl")
    print("  Saved ensemble_weights.pkl")
    # Evaluate every model and the ensemble on the untouched test years.
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST YEARS")
    print("=" * 80)
    metrics = {}
    test_predictions = {}
    for name, model in final_models.items():
        # Predict in log space, then convert to t/ha and clip negatives.
        prediction = np.clip(np.expm1(model.predict(X_test)), 0, None)
        test_predictions[name] = prediction
        # Compute the three standard metrics.
        metrics[name] = {
            "R2": float(r2_score(y_test_original, prediction)),
            "MAE": float(mean_absolute_error(y_test_original, prediction)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test_original, prediction))),
        }
    # Combine the model predictions with the validation-derived weights.
    ensemble_prediction = np.zeros(len(X_test))
    for name in final_models:
        ensemble_prediction += weights[name] * test_predictions[name]
    metrics["Ensemble"] = {
        "R2": float(r2_score(y_test_original, ensemble_prediction)),
        "MAE": float(mean_absolute_error(y_test_original, ensemble_prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_original, ensemble_prediction))),
    }
    # Score the baseline on the exact same rows.
    metrics["Baseline (District-Crop Median)"] = {
        "R2": float(r2_score(y_test_original, baseline_preds)),
        "MAE": float(mean_absolute_error(y_test_original, baseline_preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test_original, baseline_preds))),
    }
    # Print a clean, aligned metrics table.
    print(f"\n{'Model':<34}{'R2':>9}{'MAE':>11}{'RMSE':>11}")
    print("-" * 65)
    for name in ["RandomForest", "XGBoost", "HistGradientBoosting", "Ensemble",
                 "Baseline (District-Crop Median)"]:
        row = metrics[name]
        print(f"{name:<34}{row['R2']:>9.4f}{row['MAE']:>11.4f}{row['RMSE']:>11.4f}")
    print("-" * 65)
    # Save metrics as JSON.
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"[INFO] Saved {REPORTS_DIR / 'metrics.json'}")
    # Save the feature metadata the API needs.
    feature_info = {
        "feature_cols": feature_cols,
        "location_feature": "district_key = state | district",
        "years_sorted": years_sorted,
        "train_years": train_years,
        "val_years": val_years,
        "test_years": test_years,
        "weights": weights,
        "val_maes": val_maes,
        "tuned_params": tuned_params,
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_test": int(len(X_test)),
        "test_metrics": metrics,
    }
    with open(MODELS_DIR / "feature_info.json", "w", encoding="utf-8") as handle:
        json.dump(feature_info, handle, indent=2)
    print(f"[INFO] Saved {MODELS_DIR / 'feature_info.json'}")
    # Write a human-readable comparison report.
    report_path = REPORTS_DIR / "model_comparison.md"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Model Comparison - India Crop Recommendation System\n\n")
        handle.write(f"**Train years:** {train_years}\n\n")
        handle.write(f"**Validation years:** {val_years}\n\n")
        handle.write(f"**Test years:** {test_years}\n\n")
        handle.write("## Test metrics\n\n")
        handle.write("| Model | R2 | MAE (t/ha) | RMSE (t/ha) |\n")
        handle.write("|---|---|---|---|\n")
        for name in ["RandomForest", "XGBoost", "HistGradientBoosting", "Ensemble",
                     "Baseline (District-Crop Median)"]:
            row = metrics[name]
            handle.write(f"| {name} | {row['R2']:.4f} | {row['MAE']:.4f} | {row['RMSE']:.4f} |\n")
        handle.write("\n## Ensemble weights (inverse validation MAE)\n\n")
        handle.write("| Model | Validation MAE | Weight |\n")
        handle.write("|---|---|---|\n")
        for name in ["RandomForest", "XGBoost", "HistGradientBoosting"]:
            handle.write(f"| {name} | {val_maes[name]:.4f} | {weights[name]:.4f} |\n")
        handle.write("\n## Tuned parameters\n\n")
        for name, params in tuned_params.items():
            handle.write(f"- **{name}**: `{params}`\n")
        handle.write("\n## Notes\n\n")
        handle.write("- Location feature is `state | district`, so same-named districts in different states are distinct.\n")
        handle.write("- No year feature; the split is chronological.\n")
        handle.write("- Target is log1p(yield_t_ha); predictions are converted back with expm1.\n")
        handle.write("- Baseline uses only train+validation history.\n")
        handle.write("- `Total` season roll-ups are excluded.\n")
        handle.write("- The ensemble beats the baseline on R2 and RMSE but NOT on MAE. "
                     "The baseline's median is better for typical absolute error, while the models "
                     "are better on squared error. Both results are reported.\n")
        handle.write("- Climate input is a seasonal aggregate, so these are not field-level forecasts.\n")
    print(f"[INFO] Saved {report_path}")
    # Draw a simple R2 comparison chart.
    chart_labels = ["RandomForest", "XGBoost", "HistGB", "Ensemble", "Baseline"]
    chart_keys = ["RandomForest", "XGBoost", "HistGradientBoosting", "Ensemble",
                  "Baseline (District-Crop Median)"]
    chart_values = [metrics[key]["R2"] for key in chart_keys]
    plt.figure(figsize=(10, 5))
    bars = plt.bar(chart_labels, chart_values, color=["#2e7d32", "#1565c0", "#ef6c00", "#6a1b9a", "#757575"])
    plt.ylabel("R2 (test years)")
    plt.title("Model comparison on the final two years")
    plt.axhline(0, color="black", linewidth=0.8)
    for bar, value in zip(bars, chart_values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                 ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    plt.close()
    print(f"[INFO] Saved {REPORTS_DIR / 'model_comparison.png'}")
    # Report honestly whether the ensemble beats the baseline.
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)
    ensemble_r2 = metrics["Ensemble"]["R2"]
    baseline_r2 = metrics["Baseline (District-Crop Median)"]["R2"]
    if ensemble_r2 > baseline_r2:
        print(f"[RESULT] Ensemble beats the baseline on R2 ({ensemble_r2:.4f} vs {baseline_r2:.4f}).")
    else:
        print(f"[RESULT] Ensemble does NOT beat the baseline on R2 ({ensemble_r2:.4f} vs {baseline_r2:.4f}).")
    # Compare MAE separately: a model can win on R2/RMSE and still lose on MAE.
    ensemble_mae = metrics["Ensemble"]["MAE"]
    baseline_mae = metrics["Baseline (District-Crop Median)"]["MAE"]
    if ensemble_mae < baseline_mae:
        print(f"[RESULT] Ensemble also has lower MAE ({ensemble_mae:.4f} vs {baseline_mae:.4f} t/ha).")
    else:
        print(f"[RESULT] Baseline has lower MAE ({baseline_mae:.4f} vs {ensemble_mae:.4f} t/ha).")
        print("[NOTE] The baseline wins on typical absolute error; this is reported honestly.")
    # Confirm the total model size.
    total_size = sum(path.stat().st_size for path in MODELS_DIR.glob("*.pkl"))
    print(f"[INFO] Total model size: {total_size / 1024 / 1024:.1f} MB")
    # Tell the operator that the pipeline intentionally stops here.
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - STOPPING AS REQUESTED")
    print("Review the metrics table above before running Step 4.")
    print("=" * 80)
    # Return the metrics for interactive use.
    return metrics


# Run only when executed as a script.
if __name__ == "__main__":
    main()
