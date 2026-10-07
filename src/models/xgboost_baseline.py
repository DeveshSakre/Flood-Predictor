"""
XGBoost Baseline Model for Streamflow Prediction across Catchments.

This baseline predicts next-day streamflow using tabular features
(meteorological forcings + physically motivated antecedent precipitation and
soil-moisture lags/rolling statistics + 129 static physiographic catchment descriptors).

RESPONSIBILITY: Member 5 (XGBoost Baseline & Evaluation)
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
import json
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from src.utils.config import load_config
from src.data.preprocessing import Preprocessor


# 20 approved base dynamic features
BASE_DYNAMIC_COLS = [
    "prcp(mm/day)", "tmax(C)", "tmin(C)", "srad_lw(w/m2)", "srad_sw(w/m2)",
    "wind_u(m/s)", "wind_v(m/s)", "rel_hum(%)", "pet_gleam(mm/day)", "aet_gleam(mm/day)",
    "evap_canopy(mm/day)", "evap_surface(mm/day)", "sm_lvl1(kg/m2)", "sm_lvl2(kg/m2)",
    "sm_lvl3(kg/m2)", "sm_lvl4(kg/m2)", "sin_month", "cos_month", "sin_doy", "cos_doy"
]


class XGBoostBaseline:
    """
    Gradient boosted decision tree baseline for multi-catchment streamflow forecasting.
    Trains on 169 source catchments using concatenated dynamic lags and 129 static
    catchment descriptors without target leakage.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model with hyperparameter configuration.

        Args:
            config: Dictionary of model parameters (n_estimators, max_depth, learning_rate, etc.).
        """
        if config is None:
            # Default to standard robust hyperparameters from baseline_config.yaml
            self.config = {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 6,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 30,
                "eval_metric": "rmse",
                "objective": "reg:squarederror",
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": -1,
            }
        else:
            self.config = config.copy()

        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_names: List[str] = []
        self.best_iteration: Optional[int] = None

    @staticmethod
    def prepare_catchment_tabular_features(
        df_forcing: pd.DataFrame,
        static_series: pd.Series,
        preprocessor: Optional[Preprocessor] = None,
        precip_lags: Optional[List[int]] = None,
        precip_windows: Optional[List[int]] = None,
        sm_lags: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Transform a single catchment's daily meteorological forcing and 129 static attributes
        into a tabular feature matrix without any future temporal or target streamflow leakage.

        Args:
            df_forcing: Raw daily forcing DataFrame for 1 catchment (14,976 days, 22 columns).
            static_series: Preprocessed 129-feature static Series for this catchment.
            preprocessor: Preprocessor instance to scale dynamic features using train-fitted stats.
            precip_lags: Days to lag precipitation (default: [1, 2, 3, 7, 14, 30]).
            precip_windows: Rolling precipitation sum windows in days (default: [3, 7, 14, 30]).
            sm_lags: Days to lag topsoil moisture (default: [1, 3, 7]).

        Returns:
            DataFrame of shape [14976, N_features] with deterministic column names.
        """
        if precip_lags is None:
            precip_lags = [1, 2, 3, 7, 14, 30]
        if precip_windows is None:
            precip_windows = [3, 7, 14, 30]
        if sm_lags is None:
            sm_lags = [1, 3, 7]

        # 1. Base dynamic normalization using train-fitted parameters
        if preprocessor is not None:
            df_norm = preprocessor.transform_dynamic(df_forcing)
        else:
            df_norm = df_forcing.copy()

        # 2. Add cyclical harmonic calendar features (replaces raw year/month indices)
        dt = pd.to_datetime(df_forcing[["year", "month", "day"]])
        df_norm["sin_month"] = np.sin(2.0 * np.pi * (dt.dt.month - 1) / 12.0)
        df_norm["cos_month"] = np.cos(2.0 * np.pi * (dt.dt.month - 1) / 12.0)
        df_norm["sin_doy"] = np.sin(2.0 * np.pi * (dt.dt.dayofyear - 1) / 365.25)
        df_norm["cos_doy"] = np.cos(2.0 * np.pi * (dt.dt.dayofyear - 1) / 365.25)

        # 3. Physically motivated antecedent precipitation lags (backward-looking only)
        # We compute lags and rolling windows on the normalized precipitation
        prcp_col = "prcp(mm/day)"
        for lag in precip_lags:
            df_norm[f"prcp_lag_{lag}"] = df_norm[prcp_col].shift(lag).bfill()

        # 4. Antecedent precipitation rolling sums
        for w in precip_windows:
            df_norm[f"prcp_roll_{w}d_sum"] = (
                df_norm[prcp_col].rolling(window=w, min_periods=1).sum()
            )

        # 5. Soil moisture lags & rolling means (topsoil sm_lvl1 and shallow root sm_lvl2)
        sm1_col = "sm_lvl1(kg/m2)"
        for lag in sm_lags:
            df_norm[f"sm1_lag_{lag}"] = df_norm[sm1_col].shift(lag).bfill()

        sm2_col = "sm_lvl2(kg/m2)"
        for lag in [1, 7]:
            df_norm[f"sm2_lag_{lag}"] = df_norm[sm2_col].shift(lag).bfill()

        df_norm["sm1_roll_7d_mean"] = df_norm[sm1_col].rolling(window=7, min_periods=1).mean()
        df_norm["sm2_roll_7d_mean"] = df_norm[sm2_col].rolling(window=7, min_periods=1).mean()

        # 6. Evaporative demand & temperature memory
        df_norm["pet_roll_7d_mean"] = df_norm["pet_gleam(mm/day)"].rolling(window=7, min_periods=1).mean()
        df_norm["tmax_roll_3d_mean"] = df_norm["tmax(C)"].rolling(window=3, min_periods=1).mean()

        # 7. Collect all dynamic columns in fixed order
        engineered_dynamic_cols = [
            f"prcp_lag_{lag}" for lag in precip_lags
        ] + [
            f"prcp_roll_{w}d_sum" for w in precip_windows
        ] + [
            f"sm1_lag_{lag}" for lag in sm_lags
        ] + [
            f"sm2_lag_{lag}" for lag in [1, 7]
        ] + [
            "sm1_roll_7d_mean", "sm2_roll_7d_mean",
            "pet_roll_7d_mean", "tmax_roll_3d_mean"
        ]

        all_dynamic_cols = BASE_DYNAMIC_COLS + engineered_dynamic_cols
        X_dyn = df_norm[all_dynamic_cols].copy()

        # 8. Broadcast static attributes (129 features) across all rows
        static_df = pd.DataFrame(
            np.tile(static_series.to_numpy(), (len(X_dyn), 1)),
            columns=static_series.index.tolist(),
            index=X_dyn.index,
        )

        # 9. Concatenate dynamic and static features
        X_tabular = pd.concat([X_dyn, static_df], axis=1)

        # 10. Strict Anti-Leakage Assertion
        forbidden_exact_or_substrings = [
            "reservoir_index", "flow_availability", "dis_av_cms", "ord_flow",
            "lstm_pred_streamflow", "observed_flow", "streamflow_observed",
            "q_mean", "runoff_ratio", "slope_fdc", "bfi", "q_10", "q_50", "q_90",
            "gauge_id", "station_name", "basin_name"
        ]
        col_lower_set = [c.lower() for c in X_tabular.columns]
        for kw in forbidden_exact_or_substrings:
            for c_low in col_lower_set:
                assert kw != c_low and not c_low.startswith(kw), f"CRITICAL LEAKAGE: forbidden feature '{c_low}' in XGBoost inputs!"

        return X_tabular

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "XGBoostBaseline":
        """
        Fit XGBoost regressor on training partition with optional validation early stopping.

        Args:
            X_train: Training features DataFrame.
            y_train: Training target streamflow in m^3/s.
            X_val: Optional validation features DataFrame.
            y_val: Optional validation target streamflow in m^3/s.

        Returns:
            Self (fitted model).
        """
        self.feature_names = list(X_train.columns)

        y_tr = np.asarray(y_train).ravel()
        mask_tr = (~np.isnan(y_tr)) & (y_tr >= 0.0)
        X_tr_clean = X_train.loc[mask_tr] if isinstance(X_train, pd.DataFrame) else X_train[mask_tr]
        y_tr_clean = y_tr[mask_tr]

        # Extract config parameters
        params = {
            "n_estimators": int(self.config.get("n_estimators", 300)),
            "learning_rate": float(self.config.get("learning_rate", 0.05)),
            "max_depth": int(self.config.get("max_depth", 6)),
            "subsample": float(self.config.get("subsample", 0.8)),
            "colsample_bytree": float(self.config.get("colsample_bytree", 0.8)),
            "min_child_weight": float(self.config.get("min_child_weight", 1.0)),
            "reg_alpha": float(self.config.get("reg_alpha", 0.0)),
            "reg_lambda": float(self.config.get("reg_lambda", 1.0)),
            "objective": self.config.get("objective", "reg:squarederror"),
            "eval_metric": self.config.get("eval_metric", "rmse"),
            "tree_method": self.config.get("tree_method", "hist"),
            "random_state": int(self.config.get("random_state", 42)),
            "n_jobs": int(self.config.get("n_jobs", -1)),
        }

        early_stopping = int(self.config.get("early_stopping_rounds", 30))

        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = early_stopping
            self.model = xgb.XGBRegressor(**params)

            y_v = np.asarray(y_val).ravel()
            mask_v = (~np.isnan(y_v)) & (y_v >= 0.0)
            X_v_clean = X_val.loc[mask_v] if isinstance(X_val, pd.DataFrame) else X_val[mask_v]
            y_v_clean = y_v[mask_v]

            eval_set = [(X_v_clean, y_v_clean)]
            self.model.fit(
                X_tr_clean,
                y_tr_clean,
                eval_set=eval_set,
                verbose=False,
            )
            self.best_iteration = getattr(self.model, "best_iteration", None)
        else:
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_tr_clean, y_tr_clean, verbose=False)
            self.best_iteration = None

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Generate deterministic streamflow predictions in physical units (m^3/s).
        Enforces physical non-negativity constraint: streamflow >= 0.0 m^3/s.

        Args:
            X: Input tabular feature matrix.

        Returns:
            1D numpy array of predicted streamflow.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call `fit()` first.")

        raw_preds = self.model.predict(X)
        # Physically realistic constraint: river discharge cannot be negative
        clipped_preds = np.maximum(raw_preds, 0.0)
        return clipped_preds

    def get_feature_importances(self) -> pd.Series:
        """
        Extract feature importances sorted in descending order.

        Returns:
            pd.Series indexed by feature name.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")

        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize trained model artifact, configuration, and feature names to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "model": self.model,
            "config": self.config,
            "feature_names": self.feature_names,
            "best_iteration": self.best_iteration,
        }
        joblib.dump(artifact, p)

    def load(self, filepath: Union[str, Path]) -> "XGBoostBaseline":
        """Load trained model artifact from disk."""
        artifact = joblib.load(filepath)
        if isinstance(artifact, dict) and "model" in artifact:
            self.model = artifact["model"]
            self.config = artifact["config"]
            self.feature_names = artifact["feature_names"]
            self.best_iteration = artifact.get("best_iteration")
        else:
            self.model = artifact
        return self

