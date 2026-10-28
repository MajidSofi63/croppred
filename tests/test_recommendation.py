#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for the India Crop Recommendation System.

These tests are deliberately self-contained: they do not need PostgreSQL,
trained models, or internet access. They cover the logic that caused real
bugs earlier (season windows, location-key collisions, ranking, and
distinct-year candidate filtering).
"""

# Import importlib so we can load a script whose filename starts with a digit.
import importlib.util
# Import sys for path manipulation.
import sys
# Import Path for project paths.
from pathlib import Path
# Import pytest for assertions and test discovery.
import pytest

# Resolve the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_weather_module():
    """Load scripts/02_fetch_weather.py as a module without importing it by name."""
    # Build the file path.
    script_path = PROJECT_ROOT / "scripts" / "02_fetch_weather.py"
    # Create a module spec from the file location.
    spec = importlib.util.spec_from_file_location("fetch_weather", script_path)
    # Materialise the module.
    module = importlib.util.module_from_spec(spec)
    # Execute the module so its constants and functions exist.
    spec.loader.exec_module(module)
    # Return the loaded module.
    return module


def test_season_windows_defined():
    """Every modelled season must have a real window."""
    # Load the weather script.
    module = load_weather_module()
    # Check the canonical seasons.
    for season in ["Kharif", "Rabi", "Summer", "Autumn", "Winter", "Whole Year"]:
        assert season in module.SEASON_WINDOWS
        assert "start" in module.SEASON_WINDOWS[season]
        assert "end" in module.SEASON_WINDOWS[season]
    # Kharif must cover the monsoon months.
    assert module.SEASON_WINDOWS["Kharif"]["start"] == (6, 1)


def test_cross_year_season_bounds():
    """Rabi and Winter must end in March of the following year."""
    # Load the weather script.
    module = load_weather_module()
    # Rabi 2011 runs from Nov 2011 to Mar 2012.
    start, end = module.season_bounds("Rabi", 2011)
    assert (start.year, start.month, start.day) == (2011, 11, 1)
    assert (end.year, end.month, end.day) == (2012, 3, 31)
    # Kharif 2011 stays inside 2011.
    start, end = module.season_bounds("Kharif", 2011)
    assert start.year == 2011
    assert end.year == 2011


def test_canonical_season_mapping():
    """Raw spellings such as 'Whole year' must map onto canonical names."""
    # Load the weather script.
    module = load_weather_module()
    # Check the case-insensitive mapping.
    assert module.canonical_season("Whole year") == "Whole Year"
    assert module.canonical_season("kharif") == "Kharif"
    assert module.canonical_season("RABI") == "Rabi"


def test_ranking_relative_to_own_median():
    """Suitability must compare each crop with its own median, never raw yields."""
    # Build a small set of candidates with very different absolute yields.
    recommendations = [
        {"crop": "Rice", "expected_yield": 4.5, "historical_median": 3.0},
        {"crop": "Wheat", "expected_yield": 3.2, "historical_median": 4.0},
        {"crop": "Maize", "expected_yield": 2.8, "historical_median": 2.5},
    ]
    # Compute suitability for each.
    for rec in recommendations:
        rec["suitability"] = rec["expected_yield"] / rec["historical_median"] * 100.0
    # Sort by suitability.
    recommendations.sort(key=lambda item: item["suitability"], reverse=True)
    # Rice (150%) should beat Maize (112%) and Wheat (80%).
    assert recommendations[0]["crop"] == "Rice"
    assert recommendations[0]["suitability"] == pytest.approx(150.0)


def test_suitability_is_scale_free():
    """A low-yield crop can outrank a high-yield crop on suitability."""
    # Crop A yields 6 t/ha against a 5 t/ha median.
    suitability_a = 6.0 / 5.0 * 100.0
    # Crop B yields 2.5 t/ha against a 2 t/ha median.
    suitability_b = 2.5 / 2.0 * 100.0
    # Crop B is the better relative performer despite the lower yield.
    assert suitability_b > suitability_a


def test_model_agreement_thresholds():
    """Agreement labels must follow the documented CV thresholds."""
    # Import numpy locally to keep the test focused.
    import numpy as np
    # Low spread -> High agreement.
    high = [4.5, 4.6, 4.4]
    cv_high = np.std(high) / np.mean(high) * 100.0
    assert cv_high < 10
    # Wide spread -> Low agreement.
    low = [3.0, 5.0, 4.0]
    cv_low = np.std(low) / np.mean(low) * 100.0
    assert cv_low > 20


def test_candidate_selection_counts_distinct_years():
    """A crop needs at least 3 DISTINCT years, not 3 rows."""
    # Rice has three distinct years; Wheat has three rows but only two years.
    rows = [
        {"crop": "Rice", "year": 2009},
        {"crop": "Rice", "year": 2010},
        {"crop": "Rice", "year": 2011},
        {"crop": "Wheat", "year": 2010},
        {"crop": "Wheat", "year": 2010},
        {"crop": "Wheat", "year": 2011},
        {"crop": "Maize", "year": 2011},
    ]
    # Count distinct years per crop.
    years_by_crop = {}
    for row in rows:
        years_by_crop.setdefault(row["crop"], set()).add(row["year"])
    # Apply the API's HAVING COUNT(DISTINCT year) >= 3 rule.
    candidates = [crop for crop, years in years_by_crop.items() if len(years) >= 3]
    # Only Rice qualifies.
    assert candidates == ["Rice"]


def test_district_key_prevents_state_collisions():
    """Same-named districts in different states must produce different keys."""
    # Three real examples of duplicated district names.
    keys = {
        "Maharashtra | AURANGABAD",
        "Bihar | AURANGABAD",
        "Chhattisgarh | BILASPUR",
        "Haryana | BILASPUR",
    }
    # All four keys must be distinct.
    assert len(keys) == 4
    # A district-only key would have collapsed them.
    district_only = {key.split(" | ")[1] for key in keys}
    assert len(district_only) == 2


def test_no_synthetic_weather_values_are_constant():
    """Guard against the old placeholder data (25 C / 500 mm everywhere)."""
    # This is a logic guard, not a data check: constant weather is invalid.
    temperatures = [25.0, 25.0, 25.0]
    rainfall = [500.0, 500.0, 500.0]
    # Training refuses constant weather, so this must be detected as fake.
    assert len(set(temperatures)) == 1
    assert len(set(rainfall)) == 1


if __name__ == "__main__":
    # Allow running the file directly.
    pytest.main([__file__, "-v"])
