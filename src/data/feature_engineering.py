"""
Feature engineering module for hydrologic and meteorological series.
Provides time-lagged, rolling-window, and cyclical seasonal features.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def add_cyclical_calendar_features(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Generate cyclical sine/cosine transformations for day of year to capture annual seasonality.

    Args:
        df: Input DataFrame containing date column.
        date_column: Name of datetime column.

    Returns:
        DataFrame with sin_doy and cos_doy columns.
    """
    df_out = df.copy()
    dt = pd.to_datetime(df_out[date_column])
    doy = dt.dt.dayofyear
    df_out["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    df_out["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    return df_out


def add_antecedent_precipitation_indices(
    df: pd.DataFrame,
    precip_col: str = "total_precipitation_hourly_mean",
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Compute rolling antecedent precipitation sums and exponential decay indices (API)
    to represent catchment moisture state without target leakage.

    Args:
        df: Input time-sorted DataFrame.
        precip_col: Column name representing precipitation flux.
        windows: List of lag windows in days (default: [3, 7, 14, 30]).

    Returns:
        DataFrame augmented with rolling precipitation accumulation features.
    """
    if windows is None:
        windows = [3, 7, 14, 30]

    df_out = df.copy()
    if precip_col not in df_out.columns:
        return df_out

    for w in windows:
        col_name = f"{precip_col}_roll_{w}d_sum"
        df_out[col_name] = df_out[precip_col].rolling(window=w, min_periods=1).sum()

    return df_out


def create_sequence_windows(
    dynamic_features: np.ndarray,
    target: np.ndarray,
    seq_length: int = 180,
    forecast_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice time series into sliding windows of length `seq_length` for sequential neural architectures (TCN / LSTM).

    Args:
        dynamic_features: 2D array of shape [T, F_dyn].
        target: 1D or 2D array of target variable of length T.
        seq_length: Lookback window length in time steps.
        forecast_horizon: Prediction step ahead (default 1 for next-day flood forecasting).

    Returns:
        X: 3D array of shape [N_samples, seq_length, F_dyn]
        y: Target array corresponding to horizon step.
    """
    num_timesteps = len(dynamic_features)
    x_list = []
    y_list = []

    for i in range(seq_length, num_timesteps - forecast_horizon + 1):
        x_list.append(dynamic_features[i - seq_length : i])
        y_list.append(target[i + forecast_horizon - 1])

    if not x_list:
        return np.empty((0, seq_length, dynamic_features.shape[1])), np.empty((0,))

    return np.array(x_list), np.array(y_list)
