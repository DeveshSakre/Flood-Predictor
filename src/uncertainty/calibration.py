"""
Uncertainty calibration and interval diagnostics.
Computes PICP (Coverage Probability), MPIW (Interval Width), and crossing rate.
"""

from typing import Tuple, Dict, Any
import numpy as np


def prediction_interval_coverage_probability(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Calculate Prediction Interval Coverage Probability (PICP).

    Fraction of true observations falling between lower and upper predicted quantiles (e.g. P10 and P90).
    Ideal value for 80% interval (P10 to P90) is 0.80.

    Args:
        y_true: Ground truth streamflow values.
        lower_bound: Predicted lower quantile (e.g. P10).
        upper_bound: Predicted upper quantile (e.g. P90).

    Returns:
        PICP as a float in [0.0, 1.0].
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower_bound)
    upper = np.asarray(upper_bound)

    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def mean_prediction_interval_width(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Calculate Mean Prediction Interval Width (MPIW).

    Measures the sharpness / sharpness penalty of probabilistic intervals.
    MPIW = mean(upper_bound - lower_bound).

    Args:
        lower_bound: Predicted lower quantile.
        upper_bound: Predicted upper quantile.

    Returns:
        Mean width in streamflow physical units (m^3/s).
    """
    lower = np.asarray(lower_bound)
    upper = np.asarray(upper_bound)
    return float(np.mean(upper - lower))


def check_crossing_violations(
    lower_bound: np.ndarray,
    median: np.ndarray,
    upper_bound: np.ndarray,
) -> Dict[str, float]:
    """
    Check for quantile crossing violations (e.g., lower > median, median > upper).

    Returns:
        Dictionary with crossing counts and percentages.
    """
    lower = np.asarray(lower_bound)
    med = np.asarray(median)
    upper = np.asarray(upper_bound)
    n = len(lower)

    cross_lower_med = np.sum(lower > med)
    cross_med_upper = np.sum(med > upper)
    cross_lower_upper = np.sum(lower > upper)

    return {
        "lower_gt_median_pct": float(cross_lower_med / n * 100) if n > 0 else 0.0,
        "median_gt_upper_pct": float(cross_med_upper / n * 100) if n > 0 else 0.0,
        "lower_gt_upper_pct": float(cross_lower_upper / n * 100) if n > 0 else 0.0,
    }
