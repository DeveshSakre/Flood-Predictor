"""
Unit tests for XGBoost Baseline model and tabular feature engineering.
Verifies anti-leakage invariants, non-negativity constraints, and serialization.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile

from src.models.xgboost_baseline import XGBoostBaseline, BASE_DYNAMIC_COLS
from src.data.loaders import load_catchment_forcing, load_static_attributes
from src.data.preprocessing import Preprocessor


class TestXGBoostBaseline:
    """Test suite for XGBoost baseline model."""

    @pytest.fixture(scope="class")
    def setup_data(self):
        static_df = load_static_attributes()
        preprocessor = Preprocessor()
        forcing_df = load_catchment_forcing("03001")
        return static_df, preprocessor, forcing_df

    def test_feature_preparation_shape_and_nans(self, setup_data):
        static_df, preprocessor, forcing_df = setup_data
        static_row = static_df.loc["03001"]

        X = XGBoostBaseline.prepare_catchment_tabular_features(
            forcing_df, static_row, preprocessor
        )

        assert len(X) == len(forcing_df), "Row count must match daily timesteps"
        assert X.isna().sum().sum() == 0, "Tabular features must contain zero NaNs"
        # 20 base dynamic + 19 engineered dynamic + 129 static = 168 features
        assert X.shape[1] == 168, f"Expected 168 features, got {X.shape[1]}"

    def test_anti_leakage_invariants(self, setup_data):
        static_df, preprocessor, forcing_df = setup_data
        static_row = static_df.loc["03001"]

        X = XGBoostBaseline.prepare_catchment_tabular_features(
            forcing_df, static_row, preprocessor
        )

        cols_lower = [c.lower() for c in X.columns]
        forbidden = [
            "streamflow", "observed_flow", "q_mean", "runoff_ratio",
            "reservoir_index", "flow_availability", "dis_av_cms", "ord_flow"
        ]
        for f in forbidden:
            assert f not in cols_lower, f"Forbidden leakage feature '{f}' found in features!"

    def test_fit_and_non_negative_predictions(self):
        # Create small synthetic dataset
        rng = np.random.default_rng(42)
        n_samples = 200
        n_features = 10
        feature_names = [f"feat_{i}" for i in range(n_features)]

        X_train = pd.DataFrame(rng.normal(size=(n_samples, n_features)), columns=feature_names)
        # Target with some zeros and positives
        y_train = np.maximum(rng.normal(loc=10.0, scale=15.0, size=n_samples), 0.0)

        model = XGBoostBaseline(config={"n_estimators": 20, "max_depth": 3, "random_state": 42})
        model.fit(X_train, y_train)

        preds = model.predict(X_train)
        assert len(preds) == n_samples
        assert np.all(preds >= 0.0), "All streamflow predictions must be non-negative"

        feat_imp = model.get_feature_importances()
        assert len(feat_imp) == n_features
        assert np.isclose(feat_imp.sum(), 1.0, atol=1e-3)

    def test_save_and_load(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(50, 5)), columns=[f"f_{i}" for i in range(5)])
        y = np.maximum(rng.normal(loc=5.0, scale=2.0, size=50), 0.0)

        model = XGBoostBaseline(config={"n_estimators": 10, "max_depth": 2, "random_state": 42})
        model.fit(X, y)
        preds_orig = model.predict(X)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "xgb_model.joblib"
            model.save(save_path)
            assert save_path.exists()

            loaded_model = XGBoostBaseline()
            loaded_model.load(save_path)
            preds_loaded = loaded_model.predict(X)

            assert np.allclose(preds_orig, preds_loaded)
