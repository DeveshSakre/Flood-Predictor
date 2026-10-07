"""
Hydrologic model evaluation metrics.
Includes Nash-Sutcliffe Efficiency (NSE), Kling-Gupta Efficiency (KGE), RMSE, MAE, PBIAS, and Pearson r.
"""

from typing import Dict, Any, Union
import numpy as np


def _clean_arrays(obs: np.ndarray, sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Filter out NaN values and ensure 1D shape."""
    obs = np.asarray(obs).ravel()
    sim = np.asarray(sim).ravel()
    mask = ~np.isnan(obs) & ~np.isnan(sim)
    return obs[mask], sim[mask]


def nash_sutcliffe_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Compute Nash-Sutcliffe Efficiency (NSE).

    NSE = 1 - (sum((obs - sim)^2) / sum((obs - mean(obs))^2))
    Optimal value: 1.0. Values < 0 mean model is worse than the mean observed flow.
    """
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) < 2:
        return float("nan")

    denominator = np.sum((obs_clean - np.mean(obs_clean)) ** 2)
    if denominator < 1e-12:
        return float("nan")

    numerator = np.sum((obs_clean - sim_clean) ** 2)
    return float(1.0 - (numerator / denominator))


def pearson_correlation(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Compute Pearson correlation coefficient r.
    Optimal value: 1.0.
    """
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) < 2:
        return float("nan")

    std_obs = float(np.std(obs_clean, ddof=1))
    std_sim = float(np.std(sim_clean, ddof=1))

    if std_obs < 1e-12 or std_sim < 1e-12:
        return float("nan")

    corr_mat = np.corrcoef(obs_clean, sim_clean)
    r = float(corr_mat[0, 1])
    return r if not np.isnan(r) else float("nan")


def kling_gupta_efficiency(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Compute Kling-Gupta Efficiency (KGE, Gupta et al., 2009).

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    where:
        r: Pearson correlation coefficient
        alpha: std(sim) / std(obs) (variability ratio)
        beta: mean(sim) / mean(obs) (bias ratio)
    Optimal value: 1.0.
    """
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) < 2:
        return float("nan")

    std_obs = float(np.std(obs_clean, ddof=1))
    std_sim = float(np.std(sim_clean, ddof=1))
    mean_obs = float(np.mean(obs_clean))
    mean_sim = float(np.mean(sim_clean))

    if std_obs < 1e-12 or mean_obs < 1e-12:
        return float("nan")

    # Pearson correlation r
    corr_mat = np.corrcoef(obs_clean, sim_clean)
    r = float(corr_mat[0, 1])
    if np.isnan(r):
        return float("nan")

    alpha = std_sim / std_obs
    beta = mean_sim / mean_obs

    ed = np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    return float(1.0 - ed)


def root_mean_squared_error(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE)."""
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((obs_clean - sim_clean) ** 2)))


def mean_absolute_error(obs: np.ndarray, sim: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) == 0:
        return float("nan")
    return float(np.mean(np.abs(obs_clean - sim_clean)))


def percent_bias(obs: np.ndarray, sim: np.ndarray) -> float:
    """
    Compute Percent Bias (PBIAS).

    PBIAS = 100 * sum(sim - obs) / sum(obs)
    Optimal value: 0.0%. Positive = overestimation, Negative = underestimation.
    """
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) == 0 or np.sum(obs_clean) < 1e-12:
        return float("nan")
    return float(100.0 * np.sum(sim_clean - obs_clean) / np.sum(obs_clean))


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """Compute standard suite of hydrologic verification metrics."""
    return {
        "NSE": nash_sutcliffe_efficiency(obs, sim),
        "KGE": kling_gupta_efficiency(obs, sim),
        "RMSE": root_mean_squared_error(obs, sim),
        "MAE": mean_absolute_error(obs, sim),
        "PBIAS": percent_bias(obs, sim),
        "Pearson_r": pearson_correlation(obs, sim),
    }
