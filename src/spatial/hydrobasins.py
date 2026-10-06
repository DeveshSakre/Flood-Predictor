"""
HydroBASINS Asia interface module.
Provides reference paths and Pfafstetter hydrologic watershed boundary extraction.

STATUS: Base utility functions defined. Watershed aggregation completed in Step 2.
RESPONSIBILITY: Member 1 (Data) & Member 3 (Spatial Graph)
"""

from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

from src.utils.config import load_data_config, get_repo_root


def get_hydrobasins_path() -> Path:
    """Resolve local path to HydroBASINS Level 04-12 shapefiles from data_config.yaml."""
    cfg = load_data_config()
    raw_path = Path(cfg["paths"]["hydrobasins_dir"])
    if not raw_path.is_absolute():
        return get_repo_root() / raw_path
    return raw_path


def load_basin_polygon(hybas_id: int) -> Any:
    """
    Load polygon boundary geometry for a specific HydroBASINS sub-basin unit.

    TODO (Member 1 / Member 3):
    Implement shapefile/geopandas loader for custom catchment polygon boundaries.
    For catchment-level static features and graph connections, use
    data/processed/candidate_*_catchments.csv and data/processed/graph/.
    """
    raise NotImplementedError(
        "Direct shapefile geometry loading is managed during spatial preprocessing. "
        "For tabular modeling, catchment group mapping is available in "
        "data/processed/catchment_groups.csv."
    )
