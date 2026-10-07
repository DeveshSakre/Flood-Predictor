"""
Event and flood peak metrics.
Evaluates model fidelity during high-flow flood events.
"""

from typing import Dict, Any, Tuple
import numpy as np

from src.evaluation.metrics import _clean_arrays, nash_sutcliffe_efficiency


def peak_flow_error(obs: np.ndarray, sim: np.ndarray) -> Dict[str, float]:
    """
    Compute peak magnitude error between maximum observed and maximum simulated flow.

    Returns:
        Dict with obs_peak, sim_peak, absolute_diff, and relative_error_pct.
    """
    obs_clean, sim_clean = _clean_arrays(obs, sim)

    if len(obs_clean) == 0:
        return {"obs_peak": float("nan"), "sim_peak": float("nan"), "diff_m3s": float("nan"), "rel_error_pct": float("nan")}

    obs_max = float(np.max(obs_clean))
    sim_max = float(np.max(sim_clean))
    diff = sim_max - obs_max
    rel_error = (diff / obs_max * 100.0) if obs_max > 1e-12 else float("nan")

    return {
        "obs_peak": obs_max,
        "sim_peak": sim_max,
        "diff_m3s": diff,
        "rel_error_pct": rel_error,
    }


def peak_timing_error(obs: np.ndarray, sim: np.ndarray) -> int:
    """
    Compute timing offset (in days/steps) between observed and simulated peak.
    Positive value: simulated peak arrived later (delay).
    Negative value: simulated peak arrived earlier (premature).
    """
    obs = np.asarray(obs).ravel()
    sim = np.asarray(sim).ravel()
    mask = ~np.isnan(obs) & ~np.isnan(sim)

    if not np.any(mask):
        return 0

    idx_obs = int(np.argmax(np.where(mask, obs, -np.inf)))
    idx_sim = int(np.argmax(np.where(mask, sim, -np.inf)))

    return idx_sim - idx_obs


def high_flow_nse(obs: np.ndarray, sim: np.ndarray, percentile: float = 90.0) -> float:
    """
    Compute Nash-Sutcliffe Efficiency specifically on high-flow flood days (above 90th percentile).
    """
    obs_clean, sim_clean = _clean_arrays(obs, sim)
    if len(obs_clean) < 5:
        return float("nan")

    threshold = float(np.percentile(obs_clean, percentile))
    mask = obs_clean >= threshold
    if np.sum(mask) < 2:
        return float("nan")

    return nash_sutcliffe_efficiency(obs_clean[mask], sim_clean[mask])
