"""
Uncertainty calibration, interval diagnostics, and reliability metrics.

Computes:
- PICP: Prediction Interval Coverage Probability (nominal: 80% for P10-P90)
- ACE: Average Coverage Error (|PICP - nominal|)
- MPIW: Mean Prediction Interval Width (sharpness metric)
- Quantile coverage & absolute calibration error for individual quantiles
- Quantile crossing rate & violation diagnostics
- Winkler Score: Proper scoring rule evaluating interval sharpness and coverage penalties
- Comprehensive uncertainty evaluation suite
"""

from __future__ import annotations

from typing import Dict, Sequence, Union
import numpy as np


def _prepare_arrays(*arrays: Union[np.ndarray, Sequence[float]]) -> list[np.ndarray]:
    """
    Convert inputs to 1D float numpy arrays and mask out rows containing NaN/inf.
    """
    arr_list = [np.asarray(a, dtype=np.float64).ravel() for a in arrays]

    if len({len(a) for a in arr_list}) != 1:
        raise ValueError(f"All input arrays must have the same length, got {[len(a) for a in arr_list]}")

    mask = np.ones(len(arr_list[0]), dtype=bool)
    for arr in arr_list:
        mask &= np.isfinite(arr)

    return [arr[mask] for arr in arr_list]


def prediction_interval_coverage_probability(
    y_true: Union[np.ndarray, Sequence[float]],
    lower_bound: Union[np.ndarray, Sequence[float]],
    upper_bound: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Calculate Prediction Interval Coverage Probability (PICP).

    Fraction of true observations falling between lower and upper predicted quantiles.
    For an 80% interval (P10 to P90), nominal coverage is 0.80.

    Args:
        y_true: Ground truth observations.
        lower_bound: Predicted lower quantile (e.g. P10).
        upper_bound: Predicted upper quantile (e.g. P90).

    Returns:
        PICP as float in [0.0, 1.0].
    """
    y_t, lower, upper = _prepare_arrays(y_true, lower_bound, upper_bound)
    if len(y_t) == 0:
        return 0.0

    covered = (y_t >= lower) & (y_t <= upper)
    return float(np.mean(covered))


def mean_prediction_interval_width(
    lower_bound: Union[np.ndarray, Sequence[float]],
    upper_bound: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Calculate Mean Prediction Interval Width (MPIW).

    Measures sharpness of probabilistic intervals:
        MPIW = mean(upper_bound - lower_bound)

    Args:
        lower_bound: Predicted lower quantile (e.g. P10).
        upper_bound: Predicted upper quantile (e.g. P90).

    Returns:
        Mean width in physical streamflow units (m^3/s).
    """
    lower, upper = _prepare_arrays(lower_bound, upper_bound)
    if len(lower) == 0:
        return 0.0

    return float(np.mean(upper - lower))


def quantile_coverage(
    y_true: Union[np.ndarray, Sequence[float]],
    predicted_quantile: Union[np.ndarray, Sequence[float]],
) -> float:
    """
    Calculate empirical coverage of a predicted quantile.

    Fraction of true observations that fall below the predicted quantile:
        Coverage(tau) = mean(y_true <= q_tau)

    For example:
        - P10 should cover ~10% of observations
        - P50 should cover ~50%
        - P90 should cover ~90%
    """
    y_t, predicted = _prepare_arrays(y_true, predicted_quantile)
    if len(y_t) == 0:
        return 0.0

    return float(np.mean(y_t <= predicted))


def quantile_calibration_error(
    y_true: Union[np.ndarray, Sequence[float]],
    predicted_quantile: Union[np.ndarray, Sequence[float]],
    nominal_quantile: float,
) -> float:
    """
    Calculate absolute calibration error for one quantile:
        Error = | Coverage(tau) - nominal_tau |
    """
    observed_coverage = quantile_coverage(y_true, predicted_quantile)
    return float(abs(observed_coverage - nominal_quantile))


def check_crossing_violations(
    lower_bound: Union[np.ndarray, Sequence[float]],
    median: Union[np.ndarray, Sequence[float]],
    upper_bound: Union[np.ndarray, Sequence[float]],
) -> Dict[str, float]:
    """
    Check for quantile crossing violations (e.g., lower > median, median > upper).

    Returns:
        Dictionary with percentage crossing violation rates.
    """
    lower, med, upper = _prepare_arrays(lower_bound, median, upper_bound)
    n = len(lower)

    if n == 0:
        return {
            "lower_gt_median_pct": 0.0,
            "median_gt_upper_pct": 0.0,
            "lower_gt_upper_pct": 0.0,
            "any_crossing_pct": 0.0,
        }

    cross_lower_med = np.sum(lower > med)
    cross_med_upper = np.sum(med > upper)
    cross_lower_upper = np.sum(lower > upper)
    any_crossing = (lower > med) | (med > upper) | (lower > upper)

    return {
        "lower_gt_median_pct": float(cross_lower_med / n * 100.0),
        "median_gt_upper_pct": float(cross_med_upper / n * 100.0),
        "lower_gt_upper_pct": float(cross_lower_upper / n * 100.0),
        "any_crossing_pct": float(np.mean(any_crossing) * 100.0),
    }


def winkler_score(
    y_true: Union[np.ndarray, Sequence[float]],
    lower_bound: Union[np.ndarray, Sequence[float]],
    upper_bound: Union[np.ndarray, Sequence[float]],
    alpha: float = 0.20,
) -> float:
    """
    Calculate Winkler Score for a (1 - alpha) prediction interval.

    Proper scoring rule penalizing interval width and severe coverage misses:
        W = (U - L) + (2 / alpha) * (L - y) * I(y < L) + (2 / alpha) * (y - U) * I(y > U)

    For P10-P90, alpha = 0.20, so the penalty factor is 2 / 0.20 = 10.
    Lower scores indicate superior probabilistic forecasting performance.
    """
    y_t, lower, upper = _prepare_arrays(y_true, lower_bound, upper_bound)
    if len(y_t) == 0:
        return 0.0

    width = upper - lower
    penalty_lower = np.maximum(0.0, lower - y_t) * (2.0 / alpha)
    penalty_upper = np.maximum(0.0, y_t - upper) * (2.0 / alpha)

    score = width + penalty_lower + penalty_upper
    return float(np.mean(score))


def evaluate_uncertainty_forecasts(
    y_true: Union[np.ndarray, Sequence[float]],
    p10: Union[np.ndarray, Sequence[float]],
    p50: Union[np.ndarray, Sequence[float]],
    p90: Union[np.ndarray, Sequence[float]],
) -> Dict[str, float]:
    """
    Compute comprehensive uncertainty quantification and calibration report.

    Args:
        y_true: Ground truth observations.
        p10: Predicted 10th percentile.
        p50: Predicted median (50th percentile).
        p90: Predicted 90th percentile.

    Returns:
        Dictionary containing PICP, MPIW, ACE, Winkler score, empirical coverages,
        calibration errors, and crossing statistics.
    """
    y_t, l_10, m_50, u_90 = _prepare_arrays(y_true, p10, p50, p90)
    if len(y_t) == 0:
        return {}

    # Coverage metrics
    picp_80 = prediction_interval_coverage_probability(y_t, l_10, u_90)
    ace_80 = abs(picp_80 - 0.80)
    mpiw_80 = mean_prediction_interval_width(l_10, u_90)

    # Individual quantile empirical coverages
    cov_10 = quantile_coverage(y_t, l_10)
    cov_50 = quantile_coverage(y_t, m_50)
    cov_90 = quantile_coverage(y_t, u_90)

    cal_err_10 = abs(cov_10 - 0.10)
    cal_err_50 = abs(cov_50 - 0.50)
    cal_err_90 = abs(cov_90 - 0.90)
    mean_cal_err = (cal_err_10 + cal_err_50 + cal_err_90) / 3.0

    # Proper score and crossing rate
    w_score = winkler_score(y_t, l_10, u_90, alpha=0.20)
    crossing_stats = check_crossing_violations(l_10, m_50, u_90)

    # Residual pinball loss
    from src.uncertainty.quantile_loss import pinball_loss
    loss_10 = pinball_loss(y_t, l_10, tau=0.10)
    loss_50 = pinball_loss(y_t, m_50, tau=0.50)
    loss_90 = pinball_loss(y_t, u_90, tau=0.90)
    composite_loss = (loss_10 + loss_50 + loss_90) / 3.0

    return {
        "picp_80": picp_80,
        "ace_80": ace_80,
        "mpiw": mpiw_80,
        "coverage_p10": cov_10,
        "coverage_p50": cov_50,
        "coverage_p90": cov_90,
        "calibration_error_p10": cal_err_10,
        "calibration_error_p50": cal_err_50,
        "calibration_error_p90": cal_err_90,
        "mean_calibration_error": mean_cal_err,
        "winkler_score_80": w_score,
        "pinball_loss_p10": loss_10,
        "pinball_loss_p50": loss_50,
        "pinball_loss_p90": loss_90,
        "composite_pinball_loss": composite_loss,
        "crossing_violations_pct": crossing_stats["any_crossing_pct"],
        "n_samples": float(len(y_t)),
    }