"""Data loaders for split metadata, canonical mappings, forcing time series, and static attributes."""
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def _resolve_path(rel_path: str) -> Path:
    """Resolve a relative path against the repository root."""
    p = Path(rel_path)
    if p.exists():
        return p
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root / rel_path
    if candidate.exists():
        return candidate
    return p


def load_split_metadata(partition: str = "train", data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load candidate catchment metadata for a given experimental partition.
    
    Args:
        partition: One of 'train' (source), 'val', 'test' (unseen held-out), or 'all'.
        data_dir: Path to the processed data directory.
        
    Returns:
        DataFrame containing catchment metadata with standardized 5-digit string gauge_id.
    """
    base = _resolve_path(data_dir)
    file_map = {
        "train": base / "candidate_source_catchments.csv",
        "val": base / "candidate_validation_catchments.csv",
        "test": base / "candidate_unseen_test_catchments.csv",
        "all": base / "catchment_groups.csv"
    }
    
    if partition not in file_map:
        raise ValueError(f"Partition must be one of {list(file_map.keys())}, got: {partition}")
        
    csv_file = file_map[partition]
    if not csv_file.exists():
        raise FileNotFoundError(f"Split file not found: {csv_file}")
        
    df = pd.read_csv(csv_file, dtype={"gauge_id": str})
    df["gauge_id"] = df["gauge_id"].str.zfill(5)
    return df


def load_split_catchment_ids(data_dir: str = "data/processed") -> Dict[str, List[str]]:
    """
    Load dictionary of gauge IDs for all experimental partitions:
    - 'train': 169 source catchments
    - 'val': 36 validation catchments
    - 'test': 37 held-out test catchments
    - 'ungauged_demo': 5 zero-flow demonstration catchments
    """
    train_df = load_split_metadata("train", data_dir)
    val_df = load_split_metadata("val", data_dir)
    test_df = load_split_metadata("test", data_dir)
    all_df = load_split_metadata("all", data_dir)

    train_ids = train_df["gauge_id"].tolist()
    val_ids = val_df["gauge_id"].tolist()
    test_ids = test_df["gauge_id"].tolist()

    # Ungauged demo catchments have status 'OPERATIONAL_UNGAUGED_DEPLOYMENT'
    demo_mask = all_df["target_status"] == "OPERATIONAL_UNGAUGED_DEPLOYMENT"
    demo_df = all_df[demo_mask]
    demo_ids = demo_df["gauge_id"].tolist()

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "ungauged_demo": demo_ids,
    }


def load_canonical_id_mapping(data_dir: str = "data/processed") -> pd.DataFrame:
    """
    Load the canonical 472-catchment ID mapping table.
    """
    mapping_file = _resolve_path(f"{data_dir}/id_mapping.csv")
    if not mapping_file.exists():
        raise FileNotFoundError(f"ID mapping file not found: {mapping_file}")
    df = pd.read_csv(mapping_file, dtype={"standard_gauge_id": str, "raw_shapefile_id": str})
    df["gauge_id"] = df["standard_gauge_id"].str.zfill(5)
    return df


def load_id_mapping(data_dir: str = "data/processed") -> pd.DataFrame:
    """Alias for load_canonical_id_mapping."""
    return load_canonical_id_mapping(data_dir)


def load_feature_catalog(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load the 159-feature catalog."""
    catalog_file = _resolve_path(f"{data_dir}/final_feature_list.csv")
    if not catalog_file.exists():
        raise FileNotFoundError(f"Feature catalog not found: {catalog_file}")
    return pd.read_csv(catalog_file)


def load_catchment_forcing(
    gauge_id: str,
    raw_dir: str = "data/raw",
    fallback_dir: str = "../Datasets"
) -> pd.DataFrame:
    """
    Load daily forcing time series for a single catchment.
    
    Args:
        gauge_id: 5-character string or integer gauge ID.
        raw_dir: Primary path to raw datasets.
        fallback_dir: Fallback path to raw datasets.
        
    Returns:
        DataFrame with daily meteorological records.
    """
    gid_str = str(gauge_id).zfill(5)
    primary_path = _resolve_path(f"{raw_dir}/CAMELS_IND_All_Catchments/catchment_mean_forcings/{gid_str}.csv")
    fallback_path = _resolve_path(f"{fallback_dir}/CAMELS_IND_All_Catchments/catchment_mean_forcings/{gid_str}.csv")
    
    if primary_path.exists():
        return pd.read_csv(primary_path)
    elif fallback_path.exists():
        return pd.read_csv(fallback_path)
    else:
        raise FileNotFoundError(
            f"Daily forcing file for catchment {gid_str} not found at {primary_path} or {fallback_path}. "
            "Please ensure raw datasets are placed in data/raw/ or configured appropriately."
        )
