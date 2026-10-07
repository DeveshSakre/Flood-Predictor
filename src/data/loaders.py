"""Data loaders for split metadata, canonical mappings, forcing time series, and static attributes."""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
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


def load_static_attributes(
    data_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    fallback_dir: str = "../Datasets",
    catchment_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load, encode, impute, and scale static catchment attributes for RegionalTCN.

    Adheres strictly to the anti-leakage specification:
    - Loads only the 6 approved physical attribute tables (topo, clim, land, soil, geol, anth).
    - Excludes camels_ind_hydro.csv (all 73 streamflow signatures).
    - Excludes flow_availability, reservoir_index, DIS_AV_CMS, ORD_FLOW, lstm_pred_streamflow.
    - Applies existing training-only categorical encoding metadata.
    - Applies existing training-only imputation medians / domain zeros.
    - Applies existing training-fitted StandardScaler.
    - Returns a DataFrame indexed by 5-digit gauge_id with exactly 129 columns in deterministic order.

    Args:
        data_dir: Path to processed directory containing scalers.
        raw_dir: Primary path to raw datasets.
        fallback_dir: Fallback path to raw datasets.
        catchment_ids: Optional list of 5-digit gauge IDs to filter.

    Returns:
        DataFrame of shape [N, 129] indexed by standardized 5-digit gauge_id.
    """
    from src.data.preprocessing import Preprocessor

    # 1. Resolve attributes directory
    attr_candidates = [
        _resolve_path(f"{data_dir}/../raw/CAMELS_IND_All_Catchments/attributes_csv"),
        _resolve_path(f"{raw_dir}/CAMELS_IND_All_Catchments/attributes_csv"),
        _resolve_path(f"{fallback_dir}/CAMELS_IND_All_Catchments/attributes_csv"),
        _resolve_path("Datasets/CAMELS_IND_All_Catchments/attributes_csv"),
        _resolve_path("../Datasets/CAMELS_IND_All_Catchments/attributes_csv"),
    ]
    attr_dir = None
    for cand in attr_candidates:
        if cand.exists():
            attr_dir = cand
            break

    if attr_dir is None:
        raise FileNotFoundError(
            f"CAMELS-IND attribute directory not found in candidates: {attr_candidates}. "
            "Please ensure raw datasets are placed in data/raw/ or Datasets/."
        )

    # 2. Load the six approved CAMELS-IND attribute tables (never camels_ind_hydro or camels_ind_name)
    approved_tables = ["anth", "clim", "geol", "land", "soil", "topo"]
    static_dfs: Dict[str, pd.DataFrame] = {}
    for table_name in approved_tables:
        csv_path = attr_dir / f"camels_ind_{table_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Approved attribute table not found: {csv_path}")
        df_tbl = pd.read_csv(csv_path, dtype={"gauge_id": str})
        df_tbl["gauge_id"] = df_tbl["gauge_id"].str.zfill(5)
        static_dfs[table_name] = df_tbl

    # 3. Merge tables by canonical 5-digit gauge_id
    merged = static_dfs["topo"][["gauge_id"]].copy()
    for table_name in approved_tables:
        merged = merged.merge(static_dfs[table_name], on="gauge_id")

    # 4. Initialize preprocessor with train-fitted artifacts (no refitting)
    preprocessor = Preprocessor(processed_data_dir=data_dir)
    expected_features = preprocessor.static_params.get("features", [])
    if len(expected_features) != 129:
        raise ValueError(
            f"Expected 129 static features in scaler params, got {len(expected_features)}"
        )

    # 5. One-hot encode categorical features using train-fitted categories
    encoded_df = preprocessor.encode_categorical_static(merged)
    encoded_df.index = merged["gauge_id"]

    # 6. Apply train-fitted imputation and StandardScaler
    scaled_df = preprocessor.transform_static(encoded_df)

    # 7. Extract exactly the 129 columns in deterministic order
    static_matrix = scaled_df[expected_features].copy()
    static_matrix.index = merged["gauge_id"]
    static_matrix.index.name = "gauge_id"

    # 8. Strict anti-leakage verification checks
    forbidden = [
        "reservoir_index",
        "flow_availability",
        "DIS_AV_CMS",
        "ORD_FLOW",
        "lstm_pred_streamflow",
    ]
    for feat in forbidden:
        if feat in static_matrix.columns:
            raise ValueError(f"CRITICAL: Forbidden feature '{feat}' leaked into static attributes!")

    # Verify no streamflow signatures
    hydro_sample_signatures = ["q_mean", "runoff_ratio", "slope_fdc", "bfi", "q_10", "q_50", "q_90"]
    for sig in hydro_sample_signatures:
        if sig in static_matrix.columns:
            raise ValueError(f"CRITICAL: Streamflow signature '{sig}' leaked into static attributes!")

    # Verify column count and absence of NaN/Inf
    if static_matrix.shape[1] != 129:
        raise ValueError(f"Expected exactly 129 columns, got {static_matrix.shape[1]}")
    if static_matrix.isna().any().any():
        raise ValueError("Static matrix contains NaN values after imputation and scaling.")
    if np.isinf(static_matrix.to_numpy()).any():
        raise ValueError("Static matrix contains Inf values after scaling.")

    # 9. Optional filter by catchment_ids
    if catchment_ids is not None:
        standardized_ids = [str(gid).zfill(5) for gid in catchment_ids]
        missing_ids = set(standardized_ids) - set(static_matrix.index)
        if missing_ids:
            raise KeyError(f"Catchment IDs not found in static attributes: {missing_ids}")
        static_matrix = static_matrix.loc[standardized_ids]

    return static_matrix
