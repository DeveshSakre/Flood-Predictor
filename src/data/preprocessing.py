"""
Preprocessing module for dynamic, static, and graph node features.
Loads train-fitted scalers and parameters to prevent data leakage.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

from src.utils.config import get_repo_root


class Preprocessor:
    """
    Manages transformation and normalization of dynamic and static hydrologic features
    using training-set fitted parameters.
    """

    def __init__(self, processed_data_dir: Optional[Union[str, Path]] = None):
        """
        Initialize preprocessor by loading fitted scalers and metadata.

        Args:
            processed_data_dir: Path to directory containing processed artifacts and scalers.
        """
        if processed_data_dir is None:
            self.processed_dir = get_repo_root() / "data" / "processed"
        else:
            self.processed_dir = Path(processed_data_dir)

        self.scalers_dir = self.processed_dir / "scalers"
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all train-fitted parameters from disk."""
        # Dynamic scaler params (per-feature mean and std fitted on train partition)
        dyn_params_file = self.scalers_dir / "dynamic_scaler_params.json"
        self.dynamic_params = {}
        if dyn_params_file.exists():
            with open(dyn_params_file, "r") as f:
                raw_dyn = json.load(f)
                if "features" in raw_dyn and "means" in raw_dyn and "stds" in raw_dyn:
                    for feat, mean, std in zip(raw_dyn["features"], raw_dyn["means"], raw_dyn["stds"]):
                        self.dynamic_params[feat] = {"mean": float(mean), "std": float(std)}
                else:
                    self.dynamic_params = raw_dyn

        # Static scaler joblib & params
        static_scaler_file = self.scalers_dir / "static_scaler.joblib"
        if static_scaler_file.exists():
            self.static_scaler = joblib.load(static_scaler_file)
        else:
            self.static_scaler = None

        static_params_file = self.scalers_dir / "static_scaler_params.json"
        if static_params_file.exists():
            with open(static_params_file, "r") as f:
                self.static_params: Dict[str, Any] = json.load(f)
        else:
            self.static_params = {}

        # Static imputation values
        impute_file = self.scalers_dir / "static_imputation_values.json"
        if impute_file.exists():
            with open(impute_file, "r") as f:
                self.static_imputation_values: Dict[str, float] = json.load(f)
        else:
            self.static_imputation_values = {}

        # Categorical encoding metadata
        cat_file = self.scalers_dir / "categorical_encoding_meta.json"
        if cat_file.exists():
            with open(cat_file, "r") as f:
                self.categorical_meta: Dict[str, Any] = json.load(f)
        else:
            self.categorical_meta = {}

        # Graph node scaler
        graph_scaler_file = self.scalers_dir / "graph_node_scaler.joblib"
        if graph_scaler_file.exists():
            self.graph_node_scaler = joblib.load(graph_scaler_file)
        else:
            self.graph_node_scaler = None

    def transform_dynamic(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize dynamic meteorological/streamflow features using train-fitted mean and std.

        Args:
            df: DataFrame containing dynamic features.

        Returns:
            Normalized copy of the DataFrame.
        """
        df_out = df.copy()
        for col, stats in self.dynamic_params.items():
            if col in df_out.columns:
                mean = stats["mean"]
                std = stats["std"]
                if std > 1e-8:
                    df_out[col] = (df_out[col] - mean) / std
                else:
                    df_out[col] = df_out[col] - mean
        return df_out

    def encode_categorical_static(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical static attributes using train-fitted metadata.

        Args:
            df: DataFrame containing raw categorical columns ('dom_land_cover', 'hsg_major', 'geol_class_1st').

        Returns:
            DataFrame augmented with one-hot encoded indicator columns.
        """
        df_out = df.copy()
        if not self.categorical_meta:
            return df_out

        if "dom_land_cover" in df_out.columns and "dom_land_cover_categories" in self.categorical_meta:
            for c in self.categorical_meta["dom_land_cover_categories"]:
                df_out[f"dom_lc_{c}"] = (df_out["dom_land_cover"] == c).astype(float)

        if "hsg_major" in df_out.columns and "hsg_major_categories" in self.categorical_meta:
            for c in self.categorical_meta["hsg_major_categories"]:
                df_out[f"hsg_{c}"] = (df_out["hsg_major"] == c).astype(float)

        if "geol_class_1st" in df_out.columns and "geol_class_1st_categories" in self.categorical_meta:
            for c in self.categorical_meta["geol_class_1st_categories"]:
                df_out[f"geol_{c}"] = (df_out["geol_class_1st"] == c).astype(float)

        return df_out

    def transform_static(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Impute missing static values and scale using train-fitted StandardScaler.

        Args:
            df: DataFrame containing static catchment attributes.

        Returns:
            Scaled copy of static features with exact 129-feature ordering.
        """
        df_out = df.copy()

        # Apply train-fitted medians for imputation
        for col, val in self.static_imputation_values.items():
            if col in df_out.columns:
                df_out[col] = df_out[col].fillna(val)

        # Apply train-fitted scaler to numeric and encoded columns
        if self.static_scaler is not None and "features" in self.static_params:
            cols = self.static_params["features"]
            matched_cols = [c for c in cols if c in df_out.columns]
            if len(matched_cols) == len(cols):
                scaled_mat = self.static_scaler.transform(df_out[cols].to_numpy())
                # Preserve exact 129-feature ordering from scaler metadata
                scaled_df = pd.DataFrame(scaled_mat, columns=cols, index=df_out.index)
                for c in df_out.columns:
                    if c not in cols:
                        scaled_df[c] = df_out[c]
                df_out = scaled_df

        return df_out

    def inverse_transform_feature(self, norm_values: Union[np.ndarray, pd.Series], feature_name: str) -> np.ndarray:
        """
        Inverse-transform normalized feature predictions back to original physical units.

        Args:
            norm_values: Normalized values.
            feature_name: Name of dynamic feature (e.g., 'prcp(mm/day)').

        Returns:
            Values in original units.
        """
        if feature_name not in self.dynamic_params:
            raise KeyError(f"'{feature_name}' not found in dynamic scaler parameters.")

        mean = self.dynamic_params[feature_name]["mean"]
        std = self.dynamic_params[feature_name]["std"]
        return np.asarray(norm_values) * std + mean
