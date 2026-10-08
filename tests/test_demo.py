"""
Unit tests for reviewer-facing demo data loading and selection logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.visualization.demo_data import (
    get_test_gauge_ids,
    load_catchment_metadata,
    load_option_b_predictions,
    load_xgboost_predictions,
    load_per_catchment_metrics,
    load_overall_model_summary,
    load_summary_json,
)


def test_test_catchments_count_and_format():
    """Verify that exactly 37 test catchments appear in the selector, all 5-digit strings."""
    gauges = get_test_gauge_ids()
    assert len(gauges) == 37
    for gid in gauges:
        assert isinstance(gid, str)
        assert len(gid) == 5
        assert gid.isdigit()


def test_load_valid_catchment_predictions():
    """Verify that selecting a valid catchment loads the expected 3,653 rows with correct columns."""
    sample_gids = ["08013", "08029", "12016", "08001"]
    for gid in sample_gids:
        df = load_option_b_predictions(gid)
        assert len(df) == 3653
        expected_cols = ["date", "gauge_id", "y_true", "q10", "q50", "q90"]
        for col in expected_cols:
            assert col in df.columns
        assert df["gauge_id"].iloc[0] == gid


def test_quantile_monotonicity():
    """Verify that q10 <= q50 <= q90 holds across all predictions for test catchments."""
    sample_gids = ["08013", "08029", "12016", "08001"]
    for gid in sample_gids:
        df = load_option_b_predictions(gid)
        q10 = df["q10"].to_numpy()
        q50 = df["q50"].to_numpy()
        q90 = df["q90"].to_numpy()

        assert np.all(q10 <= q50 + 1e-5), f"q10 > q50 violation in {gid}"
        assert np.all(q50 <= q90 + 1e-5), f"q50 > q90 violation in {gid}"


def test_missing_y_true_preserved_not_zero():
    """Verify that missing observations remain NaN and are never converted to zero."""
    # Gauge 08001 has known missing observations
    df = load_option_b_predictions("08001")
    nan_count = df["y_true"].isna().sum()
    assert nan_count > 0, "Expected missing y_true values in gauge 08001"
    # Ensure they are truly NaN in pandas/numpy
    assert any(pd.isna(val) for val in df["y_true"])


def test_invalid_catchment_handling():
    """Verify that an invalid catchment ID is handled cleanly with ValueError."""
    with pytest.raises(ValueError):
        load_catchment_metadata("99999")

    with pytest.raises(ValueError):
        load_option_b_predictions("99999")

    with pytest.raises(ValueError):
        load_per_catchment_metrics("99999")


def test_catchment_metadata_structure():
    """Verify metadata retrieval for 08013."""
    meta = load_catchment_metadata("08013")
    assert meta["gauge_id"] == "08013"
    assert meta["basin"] == "Mahanadi"
    assert "Kantamal" in meta["site_name"] or "Tel" in meta["river"]
    assert meta["drainage_area_km2"] is not None
    assert meta["drainage_area_km2"] > 0


def test_per_catchment_metrics():
    """Verify per-catchment metrics extraction."""
    metrics = load_per_catchment_metrics("08013")
    assert "option_b_deterministic" in metrics
    assert "option_b_uncertainty" in metrics
    assert "nse" in metrics["option_b_deterministic"]
    assert "picp_80" in metrics["option_b_uncertainty"]
    assert metrics["option_b_uncertainty"]["crossing_violations_pct"] == 0.0


def test_overall_summary_loading():
    """Verify overall model comparison summary and JSON loading."""
    df_sum = load_overall_model_summary()
    assert len(df_sum) >= 4
    assert "Model" in df_sum.columns

    sum_json = load_summary_json()
    assert "Option B: Regional TCN + River GAT + QuantileHead (Q50)" in sum_json


def test_selected_day_discharge_lookup():
    """Verify daily discharge lookup for specific dates across test catchments."""
    test_catchments = ["08011", "08013", "08029", "12016", "08001"]
    test_dates = ["2003-08-28", "2006-08-15", "2001-07-20"]

    for gid in test_catchments:
        df = load_option_b_predictions(gid)
        for dt_str in test_dates:
            match = df[df["date"].dt.strftime("%Y-%m-%d") == dt_str]
            assert not match.empty, f"Date {dt_str} missing in {gid}"
            row = match.iloc[0]
            q10, q50, q90 = row["q10"], row["q50"], row["q90"]
            assert q10 <= q50 + 1e-5
            assert q50 <= q90 + 1e-5
            # y_true can be float or NaN, but never fabricated
            y_true = row["y_true"]
            assert pd.isna(y_true) or isinstance(y_true, (float, int, np.floating))
