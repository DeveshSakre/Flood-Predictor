# River-Network GAT Spatial Encoder — Member 2 Report

> [!IMPORTANT]
> **This component is NOT the final flood forecasting model.**
> The supplied data package contains no daily meteorological forcing or observed streamflow timeseries data. This report describes only the **River-Network GAT Spatial Encoder** — a static graph embedding layer that will be combined with the TCN temporal component and a quantile regression fusion head in the final integrated system.

---

## 1. Purpose

This report documents the implementation of **Member 2: River-Network GAT** for the capstone project *"ML-Based Flood Prediction for Ungauged Rivers"* using the framework:

```
Daily meteorological forcing sequence
        ↓
Regional TCN
        ↓
Temporal embedding
        ↓
             ┌────────────────────┐
River graph →│  River-Network GAT │  ← THIS COMPONENT
             └────────────────────┘
                    ↓
             Spatial embedding [N, 64]
                    ↓
        TCN + GAT feature fusion   (future Member 3/4)
                    ↓
           Quantile Regression      (future Member 3/4)
                    ↓
               P10 / P50 / P90
```

---

## 2. Scope of Member 2 Component

This implementation covers:
- Loading and validating the pre-built river network graphs (Train/Val/Test)
- Loading the existing node scaler without refitting
- Fitting a separate edge scaler on training edges only
- Implementing a directed edge-aware GATv2Conv spatial encoder
- Running forward passes across all three graph splits
- Saving embeddings and checkpoint
- Automated tests validating all structural and numerical requirements
- Documenting the integration interface for the TCN fusion component

---

## 3. Available Data

All data consumed is read-only from `extracted_data/data/processed/`:

| File | Description |
|------|-------------|
| `graph/nodes_{train,val,test}.csv` | Node attributes (6 spatial features per catchment) |
| `graph/edges_{train,val,test}.csv` | Edge attributes (4 routing features) |
| `graph/adj_matrix_{train,val,test}.npz` | Sparse adjacency matrices |
| `scalers/graph_node_scaler.joblib` | Pre-fitted node StandardScaler (do not refit) |
| `scalers/graph_node_scaler_params.json` | Node scaler parameter documentation |
| `final_feature_list.csv` | Full feature list with type/scaling/description |

---

## 4. Missing Temporal Data

The following are **absent** from the supplied package and are **not implemented here**:

- Daily meteorological forcings (`prcp`, `tmax`, `tmin`, `pet`, `srad`, etc.)
- Observed streamflow timeseries (`streamflow_observed.csv`)
- LSTM benchmark predictions

These are required for the TCN component (Member 1) and final evaluation.

---

## 5. Graph Statistics

| Split | Nodes | Edges | Basins |
|-------|-------|-------|--------|
| Train | 169 | 119 | Godavari, Krishna, Subernarekha, Brahmani-Baitarani, Pennar, Mahi, WFRN, WFRS, EFRN, EFRS |
| Validation | 36 | 33 | Verified separate from train |
| Test | 37 | 34 | Verified separate from train and val |

All splits verified to have zero node ID overlap.

---

## 6. Node Features

Exactly **6** node features are used per catchment node, sourced from HydroRIVERS:

| Feature | Description | Scaling |
|---------|-------------|---------|
| `UPLAND_SKM` | Upstream drainage area (km²) | Pre-fitted `graph_node_scaler.joblib` |
| `CATCH_SKM` | Local catchment area (km²) | Pre-fitted `graph_node_scaler.joblib` |
| `DIST_DN_KM` | Distance downstream to ocean (km) | Pre-fitted `graph_node_scaler.joblib` |
| `DIST_UP_KM` | Distance upstream to headwater (km) | Pre-fitted `graph_node_scaler.joblib` |
| `ORD_STRA` | Strahler stream order | Pre-fitted `graph_node_scaler.joblib` |
| `ORD_CLAS` | Classical stream order | Pre-fitted `graph_node_scaler.joblib` |

> [!IMPORTANT]
> The pre-existing `graph_node_scaler.joblib` was **loaded** without refitting. No additional catchment static attributes (e.g. hydrology signatures, CAMELS) were added to the node features to prevent data leakage.

---

## 7. Edge Features

Exactly **4** directed edge features are used, encoding river routing topology:

| Feature | Description | Scaling |
|---------|-------------|---------|
| `routing_distance_km` | HydroRIVERS routing path distance (km) | New `edge_scaler.joblib` (fit on train) |
| `steps` | Number of river segment hops | New `edge_scaler.joblib` (fit on train) |
| `drainage_area_ratio` | Upstream/downstream drainage area ratio | New `edge_scaler.joblib` (fit on train) |
| `stream_order_delta` | Strahler order difference across edge | New `edge_scaler.joblib` (fit on train) |

**Edge Scaler Decision**: A new `StandardScaler` was fitted **exclusively on train-set edges** and saved to `models/gat/edge_scaler.joblib`. The same fitted scaler is applied (transform-only) to val and test edges. This preserves anti-leakage standards.

---

## 8. GAT Architecture

### GATv2Conv-based Spatial Encoder

```
node_features [N, 6]
      ↓
Linear Projection [N, 64]       (no bias)
      ↓
GATv2Conv Layer 1               (in=64, out=64, heads=4, concat=True)
  edge_attr used ✓              → [N, 256]
      ↓
ELU activation
      ↓
Dropout (p=0.2)
      ↓
GATv2Conv Final Layer           (in=256, out=64, heads=4, concat=False)
  edge_attr used ✓              → [N, 64]  (averaged over heads)
      ↓
spatial_embedding [N, 64]
```

**Baseline (MLP)**: A 3-layer MLP (`6 → 64 → 256 → 64`) using ELU activation runs in parallel as a non-graph baseline for representation comparison.

---

## 9. Directed Graph Handling

- The directed river network topology is **preserved** end-to-end.
- Edges are one-directional (upstream catchment → downstream catchment).
- The graph is **not converted** to undirected form at any stage.
- `GATv2Conv` respects directed edges natively via its asymmetric attention mechanism.

---

## 10. Self-Loop Handling

> [!NOTE]
> `GATv2Conv` adds **self-loops by default** (`add_self_loops=True`).
> When self-loops are added, each node gets a synthetic edge to itself.
> The `fill_value=0.0` parameter is used so self-loop edges receive a **zero vector** for edge attributes (as no meaningful routing information exists for a node attending to itself).
> This is documented behavior, not a silent modification.

---

## 11. Edge-Aware Attention

`GATv2Conv` in PyTorch Geometric supports `edge_dim` directly. All four edge attributes are passed as `edge_attr` on every forward pass. The attention coefficients are computed jointly over `(W_i * x_i || W_j * x_j || W_e * edge_attr)`, making attention fully edge-aware.

---

## 12. Spatial Embedding Dimension

Default: **64** (`spatial_embedding_dim` in `GATConfig`).

This is **configurable**. The TCN fusion member can request a different dimension (e.g. 128, 256) by changing `GATConfig.spatial_embedding_dim` before instantiating the model.

---

## 13. Forward-Pass Interface (TCN Integration Contract)

```python
from src.models.gat_config import GATConfig
from src.models.river_gat import RiverGAT

config = GATConfig(spatial_embedding_dim=64)
model = RiverGAT(config)

# Load encoder checkpoint
import torch
model.load_state_dict(torch.load("models/gat/river_gat_encoder.pt"))
model.eval()

with torch.no_grad():
    spatial_embedding = model(
        node_features,  # [N, 6]  — scaled by graph_node_scaler
        edge_index,     # [2, E]  — from edges_{split}.csv
        edge_attr       # [E, 4]  — scaled by edge_scaler
    )
    # Output: [N, 64]
```

**Future fusion**:

```
temporal_embedding : [B, N, D_tcn]    (from TCN — Member 1)
spatial_embedding  : [N, D_gat]       (from this encoder)

# broadcast spatial to batch dimension:
spatial_broadcast  : [1, N, D_gat]  →  [B, N, D_gat]

# fused_features   : [B, N, D_tcn + D_gat]
# → Quantile Regression head (Members 3/4)
```

> [!IMPORTANT]
> The fusion layer and quantile regression head are **not implemented here**. This component only produces the static spatial embedding.

---

## 14. Validation Methodology

Because no streamflow or forcing data is available in this package, supervised flood prediction evaluation was not performed.

### What WAS done (spatial encoder validation only):

| Method | Status |
|--------|--------|
| Graph structural integrity verification | ✅ All 3 splits |
| Node count verification (169/36/37) | ✅ Exact match |
| Edge count verification (119/33/34) | ✅ Exact match |
| Node feature dimension = 6 | ✅ All splits |
| Edge feature dimension = 4 | ✅ All splits |
| NaN/Inf in node features | ✅ None detected |
| NaN/Inf in edge attributes | ✅ None detected |
| NaN/Inf in output embeddings | ✅ None detected |
| Train/Val/Test ID overlap | ✅ Zero overlap |
| Node scaler loaded (not refitted) | ✅ Verified |
| Edge scaler fitted on train only | ✅ Verified |
| Forward pass output shape [N, 64] | ✅ All splits |
| Checkpoint save/load | ✅ Verified |

### Spatial Reconstruction Objective (representation sanity, NOT flood prediction):

A brief 5-epoch reconstruction pre-training was run where the spatial embedding is decoded back to the original 6-dimensional node features via a temporary linear projection. This was used **purely** to verify that gradients flow through the GATv2 layers correctly. It is labeled:

> **"Spatial GAT encoder gradient verification — NOT flood forecasting evaluation"**

---

## 15. Tests

**Test suite: `tests/test_gat.py`**

| Test | Result |
|------|--------|
| `test_graph_loading_and_counts` | ✅ PASSED |
| `test_data_integrity` | ✅ PASSED |
| `test_no_overlap` | ✅ PASSED |
| `test_scaler_saved` | ✅ PASSED |
| `test_forward_pass` | ✅ PASSED |

**Run**: `5 passed in 3.25s`

---

## 16. Attention Analysis

`GATv2Conv` exposes attention weights via `return_attention_weights=True`. This is implemented in `RiverGAT.forward()`. Weights can be extracted at inference time for interpretability analysis.

> [!NOTE]
> At this stage, attention weights are not analyzed qualitatively. This is appropriate because:
> 1. The encoder has not been trained on the actual supervised flood prediction objective.
> 2. Attention weights from reconstruction pre-training may not reflect final flood-relevant patterns.
> 3. Attention weights indicate routing priority, not causality.

---

## 17. Limitations

- **No temporal signal**: This encoder is purely spatial. It produces a static graph embedding for each catchment based on river network topology only.
- **No flood prediction**: Cannot produce P10/P50/P90 estimates independently.
- **Small graph**: Train graph has only 169 nodes — overfitting risk increases in long training runs.
- **Node scaler version mismatch**: Pre-built `graph_node_scaler.joblib` was saved with `sklearn==1.3.2`; current environment uses `1.5.2`. A version warning is raised but transform output is numerically valid.
- **Self-loops with zero edge attrs**: Self-loops receive `fill_value=0.0` edge attributes. This is a reasonable default but means the model cannot distinguish self-loop edges by attribute.

---

## 18. Files Created

```
c:\Users\hp\Downloads\capstone\
├── .venv\                                    # Python virtual environment (PyG isolated)
├── src\
│   ├── __init__.py
│   ├── data\
│   │   ├── __init__.py
│   │   └── graph_dataset.py                 # Dataset loader (read-only on raw data)
│   ├── models\
│   │   ├── __init__.py
│   │   ├── gat_config.py                    # GATConfig dataclass
│   │   └── river_gat.py                     # RiverGAT + BaselineMLP
│   └── evaluation\
│       ├── __init__.py
│       └── evaluate_gat.py                  # Forward pass runner + checkpoint saver
├── tests\
│   └── test_gat.py                          # 5 automated tests
├── models\
│   └── gat\
│       ├── river_gat_encoder.pt             # Trained model checkpoint
│       └── edge_scaler.joblib               # Edge feature scaler (train-fitted only)
├── results\
│   └── gat\
│       ├── train_embeddings.pt              # Spatial embeddings [169, 64]
│       ├── val_embeddings.pt                # Spatial embeddings [36, 64]
│       ├── test_embeddings.pt               # Spatial embeddings [37, 64]
│       ├── evaluation_summary.json          # Forward-pass audit
│       └── final_test_run.log               # pytest output
└── reports\
    └── river_gat_report.md                  # This document
```

**Original package** (`extracted_data/data/`) was **not modified**.

---

## 19. Reproducibility Information

| Item | Value |
|------|-------|
| Python | 3.13.3 |
| PyTorch | 2.11.0+cpu |
| PyTorch Geometric | 2.8.0.post1 |
| scikit-learn (venv) | 1.5.2 |
| Device | CPU |
| Random seed | 42 |
| Node feature dimension | 6 |
| Edge feature dimension | 4 |
| Hidden dimension | 64 |
| Attention heads | 4 |
| Number of GAT layers | 2 |
| Dropout | 0.2 |
| Spatial embedding dimension | 64 |
| Self-loops | Yes (`fill_value=0.0` for edge attrs) |
| Directed graph | Yes (preserved) |
| Edge scaling | StandardScaler fitted on train edges only |

---

## 20. TCN Integration Contract (Final)

```
GAT Input:
  node_features : Tensor[N, 6]     (scaled by graph_node_scaler.joblib)
  edge_index    : Tensor[2, E]     (directed, from edges_{split}.csv)
  edge_attr     : Tensor[E, 4]     (scaled by edge_scaler.joblib)

GAT Output:
  graph_embedding : Tensor[N, D_gat]   (default D_gat = 64, configurable)

Future TCN Output (placeholder):
  temporal_embedding : Tensor[B, N, D_tcn]

Future Fusion Layer (NOT implemented here):
  fused = concat(temporal_embedding, spatial_embedding.unsqueeze(0).expand(B, -1, -1))
  # → Tensor[B, N, D_tcn + D_gat]
  # → Quantile Regression Head → P10, P50, P90
```

**Model loading interface**:
```python
model = RiverGAT(GATConfig(spatial_embedding_dim=64))
model.load_state_dict(torch.load("models/gat/river_gat_encoder.pt"))
model.eval()
embedding = model(node_features, edge_index, edge_attr)  # [N, 64]
```
