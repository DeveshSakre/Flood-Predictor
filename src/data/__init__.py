"""Data loading, preprocessing, and feature engineering modules."""
from src.data.loaders import (
    load_split_metadata,
    load_split_catchment_ids,
    load_id_mapping,
    load_canonical_id_mapping,
    load_feature_catalog,
    load_catchment_forcing,
)
from src.data.preprocessing import Preprocessor
from src.data.feature_engineering import (
    add_cyclical_calendar_features,
    add_antecedent_precipitation_indices,
    create_sequence_windows,
)

__all__ = [
    "load_split_metadata",
    "load_split_catchment_ids",
    "load_id_mapping",
    "load_canonical_id_mapping",
    "load_feature_catalog",
    "load_catchment_forcing",
    "Preprocessor",
    "add_cyclical_calendar_features",
    "add_antecedent_precipitation_indices",
    "create_sequence_windows",
]
