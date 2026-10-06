# Data Architecture and Local Setup Guide

This directory manages the data pipeline for **"ML-Based Flood Prediction for Ungauged Rivers"**.

> [!IMPORTANT]
> **Raw Data Privacy & Size Rule**:  
> Raw dataset files are **NOT** tracked by Git and must **NEVER** be committed to GitHub. Each team member must download/store raw datasets locally in their own local environment and configure paths via `configs/data_config.yaml`.

---

## 1. Expected Local Directory Layout

To reproduce experiments, team members should place the raw datasets into `data/raw/` (or configure an external directory such as `../Datasets` in `configs/data_config.yaml`):

```
ml-flood-prediction-ungauged-rivers/
└── data/
    ├── raw/                                           # NOT tracked in Git (local storage only)
    │   ├── .gitkeep
    │   │
    │   ├── CAMELS_IND_All_Catchments/                 # Extracted CAMELS-IND dataset
    │   │   ├── attributes_csv/                        # camels_ind_{anth,clim,geol,hydro,land,name,soil,topo}.csv
    │   │   ├── catchment_mean_forcings/               # 03001.csv, ..., 17025.csv (472 daily files)
    │   │   ├── shapefiles_catchment/                  # merged/all_catchments.shp and basin shapefiles
    │   │   └── streamflow_timeseries/                 # streamflow_observed.csv, lstm_pred_streamflow.csv
    │   │
    │   ├── HydroRIVERS_v10_as.gdb/                    # Extracted HydroRIVERS Asia geodatabase
    │   │   └── HydroRIVERS_v10_as.gdb/                # Layer: HydroRIVERS_v10_as (1,428,959 reaches)
    │   │
    │   └── hybas_as_lev01-12_v1c/                     # HydroBASINS Asia shapefiles
    │       ├── hybas_as_lev01_v1c.shp
    │       └── ... up to hybas_as_lev12_v1c.shp
    │
    └── processed/                                     # TRACKED in Git (Metadata, Scalers & Graphs)
        ├── README.md                                  # Detailed documentation of processed files
        ├── final_feature_list.csv                     # 159 approved model-ready features
        ├── id_mapping.csv                             # Master 5-digit canonical ID cross-reference table
        ├── catchment_groups.csv                       # Status of all 472 catchments
        ├── candidate_source_catchments.csv            # 169 training/source catchments
        ├── candidate_validation_catchments.csv        # 36 validation catchments
        ├── candidate_unseen_test_catchments.csv       # 37 held-out test catchments
        ├── scalers/                                   # Train-fitted scalers and imputation parameters
        └── graph/                                     # Directed river graph nodes, edges & adj matrices
```

---

## 2. Dataset Descriptions & Provenance

### 2.1 CAMELS-IND (Version 2.2, March 2025)
- **Publication**: Mangukiya et al. (2025), *CAMELS-IND: hydrometeorological time series and catchment attributes for catchments in Peninsular India*, Earth System Science Data (ESSD), 17, 461–2025. [https://doi.org/10.5194/essd-17-461-2025](https://doi.org/10.5194/essd-17-461-2025).
- **Contents**: 472 catchments across 15 peninsular river basins; continuous daily meteorology and soil moisture (1980–2020, 14,976 days); 218 static physiographic attributes; observed daily CWC streamflow.

### 2.2 HydroRIVERS Asia (v1.0)
- **Reference**: Lehner, B., Grill, G. (2013), *Global river hydrography and network routing*, Hydrological Processes, 27(15), 2171–2186.
- **Contents**: 1,428,959 river reach segments across Asia with topological connectivity (`NEXT_DOWN`), reach lengths, upstream drainage areas (`UPLAND_SKM`), and Strahler stream orders.

### 2.3 HydroBASINS Asia (v1c)
- **Reference**: WWF HydroSHEDS project (Lehner & Grill, 2013).
- **Contents**: Nested hierarchical sub-basin polygons (Levels 01 to 12) structured via topological Pfafstetter codes.

---

## 3. Immutability of Raw Data

> [!CAUTION]
> **Strict Data Integrity Policy**:  
> Team members must **never** modify, edit, overwrite, rename, or delete raw dataset files. All preprocessing and feature transformations are executed in memory or read from pre-computed metadata in `data/processed/`.

---

## 4. Configurable Paths via `configs/data_config.yaml`

Paths to local datasets are decoupled from hardcoded machine strings. To set your local path, update `configs/data_config.yaml`:
```yaml
paths:
  raw_data_dir: "data/raw"             # Or relative "../Datasets"
  camels_dir: "data/raw/CAMELS_IND_All_Catchments"
  hydrorivers_gdb: "data/raw/HydroRIVERS_v10_as.gdb/HydroRIVERS_v10_as.gdb"
  hydrobasins_dir: "data/raw/hybas_as_lev01-12_v1c"
  processed_dir: "data/processed"
```
