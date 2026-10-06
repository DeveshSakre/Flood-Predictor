# Processed Data, Metadata & Graph Artifacts

This directory contains the model-ready metadata, canonical identifiers, train-fitted scalers, and directed river network graphs generated in Steps 1–3.

All files in this directory are lightweight and tracked by Git so that team members can immediately load pre-computed, leakage-safe splits without recomputing expensive graph traversals or geospatial intersections.

---

## 1. Catchment Split Metadata

| File Name | Catchment Count | Description |
| :--- | :---: | :--- |
| [`candidate_source_catchments.csv`](candidate_source_catchments.csv) | **169** | Training/Source catchments across 10 hydrologically isolated basins (Godavari, Krishna, WFRS, EFRS, WFRN, Brahmani-Baitarani, Pennar, Mahi, EFRN, Subernarekha). |
| [`candidate_validation_catchments.csv`](candidate_validation_catchments.csv) | **36** | Validation/Hyperparameter tuning catchments across 3 isolated basins (Cauvery, Tapi, Sabarmati). |
| [`candidate_unseen_test_catchments.csv`](candidate_unseen_test_catchments.csv) | **37** | Held-Out Test catchments across 2 isolated basins (Mahanadi, Narmada). Streamflow is strictly hidden during inference. |
| [`catchment_groups.csv`](catchment_groups.csv) | **472** | Complete master catalogue classifying all 472 CAMELS-IND catchments by status (Train, Val, Test, Ungauged Demo, Nested Excluded). |
| [`id_mapping.csv`](id_mapping.csv) | **472** | Canonical 5-character zero-padded string ID mapping cross-referenced with HydroRIVERS reaches and HydroBASINS polygons. |
| [`final_feature_list.csv`](final_feature_list.csv) | **159** | Registry of all approved input features (20 dynamic, 129 static, 10 graph features). |

---

## 2. Train-Fitted Scalers & Parameters (`scalers/`)

All scalers were fitted **strictly on the 169 training catchments** to ensure zero data leakage:

- `dynamic_scaler_params.json`: Global mean ($\mu$) and standard deviation ($\sigma$) for the 16 physical daily forcings calculated across $2,530,944$ daily observations.
- `static_scaler.joblib`: Serialized Scikit-Learn `StandardScaler` for the 129 encoded static catchment features.
- `static_scaler_params.json`: Human-readable means and standard deviations for all static features.
- `static_imputation_values.json`: Medians and domain-zero rules for missing static values.
- `categorical_encoding_meta.json`: Verified category vocabularies for land cover, hydrologic soil groups, and lithology.
- `graph_node_scaler.joblib`: Serialized `StandardScaler` for HydroRIVERS reach scale features.
- `graph_node_scaler_params.json`: Feature means and standard deviations for graph nodes.

---

## 3. Directed River Network Graphs (`graph/`)

Directed river graphs constructed from HydroRIVERS Asia for each partition:

- **Train Graph**:
  - `nodes_train.csv`: 169 nodes with reach attributes (`UPLAND_SKM`, `CATCH_SKM`, `DIST_DN_KM`, `DIST_UP_KM`, `ORD_STRA`, `ORD_CLAS`).
  - `edges_train.csv`: 119 directed downhill edges with routing distances and area ratios.
  - `adj_matrix_train.npz`: Compressed $(169 \times 169)$ directed adjacency matrix.
- **Validation Graph**:
  - `nodes_val.csv`: 36 nodes.
  - `edges_val.csv`: 33 directed downhill edges.
  - `adj_matrix_val.npz`: Compressed $(36 \times 36)$ directed adjacency matrix.
- **Held-Out Test Graph**:
  - `nodes_test.csv`: 37 nodes.
  - `edges_test.csv`: 34 directed downhill edges.
  - `adj_matrix_test.npz`: Compressed $(37 \times 37)$ directed adjacency matrix.

> [!NOTE]
> All graphs are verified Directed Acyclic Graphs ($\text{DAG} = \text{True}$) with **zero cross-partition edges**.
