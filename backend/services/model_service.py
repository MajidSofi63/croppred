#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model service for the India Crop Recommendation System.
Loads the three trained regressors, the label encoders, the ensemble weights,
and the feature metadata once at startup, then serves fast predictions.
"""

# Import json for reading metrics and feature metadata.
import json
# Import joblib for loading serialised models.
import joblib
# Import numpy for array construction and back-transformation.
import numpy as np
# Import Path for filesystem paths.
from pathlib import Path
# Import typing helpers.
from typing import Dict, Any


class ModelService:
    """Load and use the trained yield models and their ensemble."""

    def __init__(self, models_dir: Path):
        """Store the model directory and load every artefact."""
        # Save the directory for later report lookups.
        self.models_dir = Path(models_dir)
        # Populate empty containers before loading.
        self.models: Dict[str, Any] = {}
        self.weights: Dict[str, float] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_info: Dict[str, Any] = {}
        # Load everything immediately so request handling stays fast.
        self.load_models()

    def load_models(self):
        """Load models, encoders, weights, and feature metadata from disk."""
        # Load the three estimators saved by Step 3.
        self.models["RandomForest"] = joblib.load(self.models_dir / "randomforest_model.pkl")
        self.models["XGBoost"] = joblib.load(self.models_dir / "xgboost_model.pkl")
        self.models["HistGradientBoosting"] = joblib.load(
            self.models_dir / "histgradientboosting_model.pkl"
        )
        # Load the label encoders.
        self.encoders["district"] = joblib.load(self.models_dir / "le_district.pkl")
        self.encoders["season"] = joblib.load(self.models_dir / "le_season.pkl")
        self.encoders["crop"] = joblib.load(self.models_dir / "le_crop.pkl")
        # Load the ensemble weights.
        self.weights = joblib.load(self.models_dir / "ensemble_weights.pkl")
        # Load the feature metadata JSON.
        with open(self.models_dir / "feature_info.json", "r", encoding="utf-8") as handle:
            self.feature_info = json.load(handle)
        # Single-row inference must not spawn parallel worker processes.
        # Training used n_jobs=-1, which is disastrous for one-row predictions.
        for model in self.models.values():
            if hasattr(model, "set_params") and "n_jobs" in model.get_params():
                model.set_params(n_jobs=1)
        # Cache the encoder class sets so every request does not rebuild them.
        self.class_sets = {
            name: set(encoder.classes_) for name, encoder in self.encoders.items()
        }
        # Confirm success in the startup log.
        print("[OK] All models, encoders, weights, and metadata loaded")

    def _encode(self, encoder_name: str, value: str):
        """Encode one category, returning None when the value is unseen."""
        # Pull the requested encoder.
        encoder = self.encoders[encoder_name]
        # Use the cached class set instead of rebuilding it for every crop.
        if value not in self.class_sets[encoder_name]:
            return None
        # Return the integer code.
        return int(encoder.transform([value])[0])

    def predict_many(self, district_key: str, season: str, crops: list,
                     temp_mean: float, rain_total: float) -> Dict[str, Dict[str, float]]:
        """
        Predict yield (t/ha) for several crops in ONE batched model call.

        A single-row predict() call per crop is slow because tree models pay
        fixed per-call overhead. Batching all candidate crops cuts a request
        from several seconds to a few milliseconds.
        """
        # Encode the two values that are shared by every candidate crop.
        district_enc = self._encode("district", district_key)
        season_enc = self._encode("season", season)
        # Nothing can be predicted when the location or season is unseen.
        if district_enc is None or season_enc is None:
            return {}
        # Keep only crops the encoder knows, preserving the caller's order.
        valid_crops = []
        rows = []
        for crop in crops:
            crop_enc = self._encode("crop", crop)
            if crop_enc is None:
                continue
            valid_crops.append(crop)
            # Training feature order:
            # district_enc, season_enc, crop_enc, temp_mean, rain_total.
            rows.append([district_enc, season_enc, crop_enc,
                         float(temp_mean), float(rain_total)])
        # Return early when no crop was usable.
        if not rows:
            return {}
        # Build the batch matrix once and predict each model once.
        features = np.asarray(rows, dtype=float)
        per_model = {}
        for name, model in self.models.items():
            # Predict in log space, then convert back to t/ha.
            log_predictions = model.predict(features)
            per_model[name] = np.clip(np.expm1(log_predictions), 0.0, None)
        # Assemble per-crop results plus the weighted ensemble.
        results: Dict[str, Dict[str, float]] = {}
        for index, crop in enumerate(valid_crops):
            values = {
                name: float(predictions[index])
                for name, predictions in per_model.items()
            }
            # Combine with the validation-derived weights.
            ensemble = sum(
                float(self.weights.get(name, 0.0)) * values[name]
                for name in self.models
            )
            # Store the ensemble under its own key.
            values["Ensemble"] = float(max(ensemble, 0.0))
            results[crop] = values
        return results

    def predict(self, district_key: str, season: str, crop: str,
                temp_mean: float, rain_total: float) -> Dict[str, Any]:
        """
        Predict yield (t/ha) for one location-season-crop combination.

        district_key must be "state | district", matching training.
        Returns per-model predictions and the weighted ensemble.
        """
        # Reuse the batched implementation so there is only one code path.
        batch = self.predict_many(
            district_key, season, [crop], temp_mean, rain_total
        )
        # Report which value was unseen.
        if crop not in batch:
            return {
                "predictions": {},
                "success": False,
                "error": f"Unseen categorical value for crop '{crop}'",
            }
        # Return the single result in the original shape.
        return {"predictions": batch[crop], "success": True}

    def get_model_info(self) -> Dict[str, Any]:
        """Return test metrics, weights, validation MAEs, and test years."""
        # Serve the cached copy after the first read; the file never changes
        # while the API is running.
        if getattr(self, "_model_info_cache", None) is not None:
            return self._model_info_cache
        try:
            # The reports directory lives two levels above backend/models.
            metrics_path = self.models_dir.parent.parent / "reports" / "metrics.json"
            with open(metrics_path, "r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            # Return the metadata the API exposes to the sidebar.
            self._model_info_cache = {
                "metrics": metrics,
                "weights": self.weights,
                "val_maes": self.feature_info.get("val_maes", {}),
                "test_years": self.feature_info.get("test_years", []),
            }
            return self._model_info_cache
        except Exception as exc:
            # Missing reports must not break a recommendation.
            return {"error": str(exc), "metrics": {}, "weights": self.weights}

    def get_test_metrics_for_model(self, model_name: str) -> Dict[str, Any]:
        """Return the test metrics for one model name."""
        # Read the shared metrics report.
        info = self.get_model_info()
        # Look the model up and default to an empty dict.
        return info.get("metrics", {}).get(model_name, {})

    def get_model_weight(self, model_name: str) -> float:
        """Return one model's ensemble weight."""
        return float(self.weights.get(model_name, 0.0))
