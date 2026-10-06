"""
HydroRIVERS Asia interface module.
Provides reference paths and geometry extraction utilities for river reaches.

STATUS: Base utility functions defined. Full reach topology matching completed in Step 3.
RESPONSIBILITY: Member 1 (Data) & Member 3 (Spatial Graph)
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from src.utils.config import load_data_config, get_repo_root


def get_hydrorivers_path() -> Path:
    """Resolve local path to HydroRIVERS FileGeodatabase (.gdb) from data_config.yaml."""
    cfg = load_data_config()
    raw_path = Path(cfg["paths"]["hydrorivers_dir"])
    if not raw_path.is_absolute():
        return get_repo_root() / raw_path
    return raw_path


def load_reach_metadata(reach_id: Optional[int] = None) -> pd.DataFrame:
    """
    Extract river reach characteristics (NEXT_DOWN, LENGTH_KM, DIST_MAIN, UPA, ORD_STRA).

    TODO (Member 3):
    Extend with spatial index queries for arbitrary ungauged point coordinates.
    Currently, pre-extracted reach connections are serialized in data/processed/graph/edges_*.csv.
    """
    raise NotImplementedError(
        "Direct .gdb spatial querying is handled in the offline audit pipeline. "
        "For model training and graph analysis, use pre-extracted topology in "
        "data/processed/graph/ via `src.spatial.graph_utils.load_partition_graph`."
    )
