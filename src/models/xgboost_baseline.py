"""
XGBoost Baseline Model for Streamflow Prediction across Catchments.

This baseline predicts next-day streamflow using tabular features (lagged dynamic variables + static attributes).
RESPONSIBILITY: Member 5 (XGBoost Baseline & Evaluation)
STATUS: Initial baseline interface. Scheduled for full training in Step 4.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import joblib


class XGBoostBaseline:
    """
    Gradient boosted tree baseline for multi-catchment streamflow forecasting.
    Trains on 169 source catchments using concatenated dynamic lags and static catchment descriptors.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize model with hyperparameter configuration.

        Args:
            config: Dictionary of model parameters (n_estimators, max_depth, learning_rate, etc.).
        """
        self.config = config or {
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
        }
        self.model = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> "XGBoostBaseline":
        """
        Fit XGBoost regressor on training partition with optional validation early stopping.

        TODO (Member 5):
        Implement in Step 4 when training begins.
        """
        raise NotImplementedError(
            "Model training has not started. Step 4 will train this baseline model. "
            "Assigned to Member 5."
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate deterministic streamflow predictions (m^3/s).
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call `fit()` first.")
        return self.model.predict(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize trained model artifact to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        joblib.dump(self.model, filepath)

    def load(self, filepath: Union[str, Path]) -> "XGBoostBaseline":
        """Load trained model artifact from disk."""
        self.model = joblib.load(filepath)
        return self
