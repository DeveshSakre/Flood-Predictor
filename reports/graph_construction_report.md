# River Network Graph Construction & Topological Diagnostics Report

**Project**: ML-Based Flood Prediction for Ungauged Rivers  
**Planned Architecture**: Regional Temporal Convolutional Network (TCN) + River-Network Graph Attention Network (GAT) + Quantile Regression  
**Datasets**: CAMELS-IND, HydroRIVERS Asia, HydroBASINS Asia  
**Phase**: Step 3 — River Network Graph Construction & Topological Verification  
**Date**: October 2026  
**Status**: Completed (Graph files serialized, DAG verified, zero cross-partition leakage confirmed)

---

## Executive Summary

This report establishes the directed river network graph representation required for the **River-Network Graph Attention Network (GAT)**. In hydrology, surface runoff and channel flow follow a strict gravitational hierarchy: headwaters drain into intermediate tributaries, which converge into major river trunks. A standard Graph Neural Network must model this advective, directional routing without cycle formation and without leaking hydraulic signals across experimental partitions.

Using the Asian reach network of HydroRIVERS (1,428,959 reaches), directed graphs were constructed for each experimental partition:
- **Train Graph**: 169 nodes, 119 directed edges across 10 independent river basins.
- **Validation Graph**: 36 nodes, 33 directed edges across 3 independent river basins.
- **Held-Out Test Graph**: 37 nodes, 34 directed edges across 2 independent river basins (Mahanadi and Narmada).
- **Zero Cross-Partition Leakage**: **Exactly 0 cross-partition directed edges** exist between Train, Validation, and Test, guaranteeing mathematically complete topological isolation.
- **Directed Acyclic Graph (DAG) Verification**: All three partition graphs are verified to be strictly acyclic ($\text{DAG} = \text{True}$), directly reflecting downhill gravity flow.

---

## 1. Graph Construction Methodology (Part 7)

### 1.1 Node Definition & Representation
Each node $v_i \in \mathcal{V}$ corresponds to a CAMELS-IND catchment delineated at a gauged river section and mapped to its intersecting HydroRIVERS reach $\text{HYRIV\_ID}$.
- **Node Ordering**: Nodes within each partition are sorted canonically by their standardized 5-character gauge ID and assigned integer indices $i \in \{0, \dots, N_{\text{split}}-1\}$.
- **Node Feature Vector $\mathbf{x}_{v_i} \in \mathbb{R}^6$**:
  Derived from HydroRIVERS physical reach and contributing basin scales:
  1. `UPLAND_SKM`: Total upstream contributing drainage area ($\text{km}^2$). Governs cumulative macro-scale flood volume potential.
  2. `CATCH_SKM`: Local reach sub-catchment area ($\text{km}^2$). Governs immediate lateral overland inflow into the channel reach.
  3. `DIST_DN_KM`: Longitudinal flow distance from the reach outlet to the ocean sink ($\text{km}$). Establishes downstream network position.
  4. `DIST_UP_KM`: Longitudinal flow distance from the reach outlet to the furthest headwater divide ($\text{km}$). Establishes upstream network maturity.
  5. `ORD_STRA`: Strahler stream order ($1$ to $9$). Measures hierarchical branching complexity.
  6. `ORD_CLAS`: Classical stream order ($1$ = main stem river from sink to source). Distinguishes backbone trunks from lateral tributaries.

> [!NOTE]
> All continuous node features were normalized using a `StandardScaler` fitted **strictly on the 169 training nodes**. Parameters are saved in [`data/processed/scalers/graph_node_scaler_params.json`](../data/processed/scalers/graph_node_scaler_params.json).

### 1.2 Directed Edge Definition & Hydraulic Message Passing
Water flows advectively downhill. Therefore, all graph edges are directed from **upstream catchments to downstream catchments**:
$$e = (u, v) \quad \text{where } u \text{ drains into } v$$
- **Edge Formation Rule**: A directed edge is established from node $u$ to node $v$ if and only if reach $u$ connects to reach $v$ along the downstream flow path in HydroRIVERS (`NEXT_DOWN`) **and no other gauged node in the partition lies between them**. This captures the immediate parent-child tributary structure.
- **Edge Attributes Vector $\mathbf{e}_{(u, v)} \in \mathbb{R}^4$**:
  1. `routing_distance_km`: Cumulative physical river channel length ($\sum \text{LENGTH\_KM}$) along the HydroRIVERS path between $u$ and $v$ ($\text{km}$). Serves as the continuous edge distance for flood wave travel time.
  2. `steps`: Number of intermediate 15-arc-second HydroRIVERS reach segments traversed between $u$ and $v$.
  3. `drainage_area_ratio`: Ratio of upstream area to downstream area $\frac{\text{UPLAND\_SKM}(u)}{\text{UPLAND\_SKM}(v)}$. Represents the fractional tributary contribution to the downstream junction.
  4. `stream_order_delta`: $\text{ORD\_STRA}(v) - \text{ORD\_STRA}(u)$. Measures stream order increase along the reach.

### 1.3 Strict Exclusion of Modeled Discharge
`DIS_AV_CMS` and `ORD_FLOW` were **strictly excluded** from node and edge features. Using global WaterGAP simulation priors would inject non-physical model biases into our ungauged GAT attention heads.

---

## 2. Comprehensive Graph Diagnostics (Part 8)

Graph structural metrics were computed programmatically for each partition:

```
                            GRAPH TOPOLOGY SUMMARY
═══════════════════════════════════════════════════════════════════════════
Metric                        Train Graph      Validation Graph    Test Graph
───────────────────────────────────────────────────────────────────────────
Total Catchment Nodes (N)        169                 36                37
Total Directed Edges (E)         119                 33                34
Graph Density                  0.00419             0.02619           0.02553
Isolated Nodes (in=0, out=0)      36                  0                 1
In-Degree Range [Min, Max]      [0, 5]              [0, 6]            [0, 5]
Mean In-Degree                   0.704               0.917             0.919
Out-Degree Range [Min, Max]     [0, 1]              [0, 1]            [0, 1]
Mean Out-Degree                  0.704               0.917             0.919
Weakly Connected Components       50                  3                 3
Strongly Connected Components    169                 36                37
Directed Acyclic Graph (DAG)     TRUE                TRUE              TRUE
Cross-Partition Edges              0                  0                 0
═══════════════════════════════════════════════════════════════════════════
```

### 2.1 Detailed Topology Analysis

#### A. Train Graph (169 Nodes, 119 Directed Edges)
- **Out-Degree Constraint**: The maximum out-degree is strictly $1$ across all 169 nodes ($\text{out-degree} \in \{0, 1\}$). Because surface water flows down a single drainage channel in a dendritic network, every upstream node connects to at most one immediate downstream station. Terminal basin outlet stations have $\text{out-degree} = 0$.
- **In-Degree Distribution**: In-degree ranges from $0$ (headwater stations with no upstream gauged stations in the dataset) up to $5$ (major confluence junctions receiving flow from up to 5 gauged upstream sub-tributaries).
- **Weakly Connected Components (50 Components)**:
  - 1 massive dendritic tree for Godavari (50 gauges).
  - 1 massive dendritic tree for Krishna (41 gauges).
  - Several intermediate trees for Brahmani-Baitarani (6 gauges), Mahi (6 gauges), Pennar (7 gauges).
  - 36 isolated headwater/coastal catchments with zero gauged neighbors in the partition.

#### B. Validation Graph (36 Nodes, 33 Directed Edges)
- **Zero Isolated Nodes**: Every single validation node participates in directed river routing.
- **Components**: Exactly 3 weakly connected components, corresponding to the three validation river systems: Cauvery ($21$ nodes, $20$ edges), Tapi ($10$ nodes, $9$ edges), and Sabarmati ($5$ nodes, $4$ edges).
- **Maximum In-Degree**: Station `05008` in Cauvery has an in-degree of $6$, acting as the primary central convergence hub of the Cauvery network.

#### C. Held-Out Test Graph (37 Nodes, 34 Directed Edges)
- **Components**: Exactly 3 weakly connected components:
  1. *Mahanadi Trunk Tree*: 19 interconnected nodes ($18$ directed edges) tracing the Seonath, Tel, and mainstem Mahanadi down to the delta.
  2. *Narmada Trunk Tree*: 17 interconnected nodes ($16$ directed edges) tracing the steep rift-valley descent of the Narmada.
  3. *Isolated Headwater Reach*: 1 independent headwater station in the upper Narmada divide.
- **Maximum In-Degree**: Station `08038` in the lower Mahanadi receives flow from $5$ upstream tributary stations.

---

## 3. Programmatic Proof of Zero Train $\rightarrow$ Test Edge Leakage

To verify that message-passing in the future GAT will never leak hydraulic representations across experimental partitions, reach paths were exhaustively checked across all pairs:
$$\forall u \in \mathcal{V}_{\text{train}}, \quad \forall v \in \mathcal{V}_{\text{test}}, \quad (u, v) \notin \mathcal{E} \quad \text{and} \quad (v, u) \notin \mathcal{E}$$

```python
# Programmatic Verification Result:
Cross-Partition Directed Edges: 0
Train -> Test Edges:            0
Train -> Val Edges:             0
Val -> Test Edges:              0
Topological Isolation Status:   100% VERIFIED PASSED
```

Because Mahanadi and Narmada share zero river reaches with Godavari, Krishna, Cauvery, or Tapi, **no hydraulic information, flood wave signal, or gradient can propagate across partition boundaries during graph neural network training or inference**.

---

## 4. Graph Serialization & File Structure (Part 11)

All graph structures are saved in [`data/processed/graph/`](../data/processed/graph/):

```
data/processed/graph/
├── nodes_train.csv             # 169 nodes: node_idx, gauge_id, river_basin, lat, lon, reach scales
├── edges_train.csv             # 119 edges: source_idx, target_idx, routing_distance_km, steps, area_ratio
├── adj_matrix_train.npz        # Compressed (169 x 169) directed adjacency matrix
│
├── nodes_val.csv               # 36 nodes: node_idx, gauge_id, river_basin, lat, lon, reach scales
├── edges_val.csv               # 33 edges: source_idx, target_idx, routing_distance_km, steps, area_ratio
├── adj_matrix_val.npz          # Compressed (36 x 36) directed adjacency matrix
│
├── nodes_test.csv              # 37 nodes: node_idx, gauge_id, river_basin, lat, lon, reach scales
├── edges_test.csv              # 34 edges: source_idx, target_idx, routing_distance_km, steps, area_ratio
└── adj_matrix_test.npz         # Compressed (37 x 37) directed adjacency matrix
```

### 4.1 Integration into the Future River-Network GAT
During model execution:
1. **Temporal Encoding (TCN)**: Each catchment $v$ independently maps its daily forcing sequence $(\mathbf{X}_{\text{dyn}})_{v, 1:T}$ into a latent catchment-state vector $\mathbf{h}_v \in \mathbb{R}^{d}$.
2. **Topological Message Passing (GAT)**: The River-Network GAT updates node representation $\mathbf{h}_v$ by aggregating attention-weighted signals from its immediate upstream tributaries $\mathcal{N}_{\text{up}}(v)$:
   $$\mathbf{z}_v = \mathbf{W}_0 \mathbf{h}_v + \sum_{u \in \mathcal{N}_{\text{up}}(v)} \alpha_{u, v} \mathbf{W}_1 \left[ \mathbf{h}_u \,\|\, \mathbf{e}_{(u, v)} \right]$$
   where the attention weight $\alpha_{u, v}$ dynamically measures tributary importance conditioned on routing distance and area ratio.
3. **Quantile Head**: The enriched representation $\mathbf{z}_v$ is fed to the Quantile Regression head to output predicted flood quantiles ($\tau \in \{0.1, 0.5, 0.9\}$) without target streamflow leakage.
