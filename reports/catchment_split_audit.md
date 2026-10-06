# Catchment Split Design and Leakage Audit Report

**Project**: ML-Based Flood Prediction for Ungauged Rivers  
**Planned Framework**: Regional Temporal Convolutional Network (TCN) + River-Network Graph Attention Network (GAT) + Quantile Regression  
**Datasets**: CAMELS-IND, HydroRIVERS Asia, HydroBASINS Asia  
**Phase**: Step 2 — Catchment-Level Train / Validation / Test Split Design  
**Date**: October 2026  
**Status**: Completed (Candidate splits generated; zero raw data altered; zero ML models trained)

---

## Executive Summary

Designing an evaluation protocol for an **ungauged river flood prediction model** is fundamentally different from standard machine learning cross-validation. In a true ungauged setting, target catchments possess **no historical streamflow records**. Consequently:
1. An operational model must generate flood hydrographs and quantile predictions using **exclusively** meteorology, land-surface forcings, static catchment attributes, and topological river network connectivity.
2. If random catchment splitting is used, upstream and downstream gauges along the same river system inevitably bleed into both training and test partitions. Because river flow is advective and continuous, downstream discharge is directly driven by upstream discharge, allowing neural networks to trivially interpolate streamflow rather than learning genuine ungauged rainfall-runoff physics.
3. The 159 catchments with 0% observed streamflow cannot serve as the primary quantitative test set because **no ground-truth discharge exists** to evaluate performance (NSE, KGE, RMSE, Quantile Loss).

To solve this, we designed a scientifically rigorous **Dual-Track Evaluation Setup**:
- **Track 1 (Quantitative Pseudo-Ungauged Benchmark)**: 37 hydrologically isolated catchments across two major independent basins (**Mahanadi** and **Narmada**) are held out as the **Unseen Test Catchments**. Their observed streamflow is strictly hidden during training and inference and utilized solely for post-prediction verification.
- **Track 2 (Operational Forward Deployment)**: 5 verified, non-nested zero-flow catchments are retained for zero-gauge operational simulation and spatial flood hazard mapping.
- **Candidate Split Balance**: **169 Source/Train Catchments (69.8%)**, **36 Validation Catchments (14.9%)**, and **37 Unseen Test Catchments (15.3%)**, achieving **strictly zero upstream-downstream, zero river-basin, and zero HydroBASINS overlap**.

---

## 1. Group Structure & Network Topology Analysis (Task 1)

Using the matched HydroRIVERS Asia network and HydroBASINS Asia hierarchy across the 242 streamflow-sufficient catchments, the network topology and regional structure were mapped.

### 1.1 Multi-Scale Catchment Groupings
The 242 catchments cluster hierarchically at three distinct geographic and hydrologic scales:

1. **CWC River Basins (15 Macro-Basins)**:
   - Godavari (50), Krishna (41), WFRS (25), Cauvery (21), Mahanadi (19), Narmada (18), EFRS (12), Tapi (10), WFRN (10), Brahmani-Baitarani (8), Pennar (7), Mahi (7), Sabarmati (5), EFRN (5), Subernarekha (4).
2. **HydroBASINS Continental Drainage Systems (`MAIN_BAS`)**:
   - 32 unique continental macro-basins. The largest correspond directly to major river systems: Godavari (`4060027780`, 50 gauges), Krishna (`4060027940`, 41 gauges), Cauvery (`4060028760`, 21 gauges), Mahanadi (`4060027100`, 19 gauges), and Narmada (`4060031610`, 18 gauges).
3. **HydroBASINS Regional Sub-Basins (Level 06 `PFAF_ID_L06`)**:
   - 119 distinct sub-basin clusters across the 242 catchments, providing intermediate sub-regional hydrologic boundaries.

### 1.2 Upstream-Downstream Connectivity in HydroRIVERS
Tracing downstream reach adjacency (`NEXT_DOWN`) through the 1.43-million reach HydroRIVERS network revealed:
- **56 Disjoint Connected Components**:
  - **37 Completely Isolated Catchments**: Catchments located on independent coastal rivers or headwater tributaries with zero upstream or downstream gauged neighbors in the dataset.
  - **19 Multi-Gauge Connected River Network Trees (205 Catchments)**:
    - *Godavari Tree*: 50 interconnected gauges along the Godavari mainstem, Wainganga, Wardha, Penganga, and Indravati tributaries.
    - *Krishna Tree*: 41 interconnected gauges along the Krishna, Bhima, and Tungabhadra rivers.
    - *Cauvery Tree*: 21 interconnected gauges along the Cauvery trunk and tributaries.
    - *Mahanadi Tree*: 19 interconnected gauges along the Mahanadi mainstem, Seonath, and Tel rivers.
    - *Narmada Tree*: 17 interconnected gauges along the Narmada mainstem and major tributaries (plus 1 independent headwater gauge).
    - *Tapi Tree*: 10 interconnected gauges along the Tapi and Purna rivers.
    - *Brahmani-Baitarani*: 2 separate sub-basin trees (6 and 2 gauges).
    - *Pennar*: 1 connected tree (7 gauges).
    - *Mahi*: 2 trees (6 and 1 gauges).
    - *Sabarmati*: 1 tree (5 gauges).
    - *WFRS / WFRN*: 30 independent coastal river reaches (components of 1 to 4 gauges).

> [!IMPORTANT]
> **Hydrological Rule**: To avoid catastrophic advective leakage, any multi-gauge river tree must be kept intact within a single partition. An entire connected tree must belong wholly to Train, wholly to Validation, or wholly to Test.

---

## 2. Investigation of the 159 Zero-Discharge Catchments (Task 2)

All 159 CAMELS-IND catchments with 0% observed streamflow were systematically audited for spatial location, data completeness, dam regulation, and spatial overlap with gauged basins.

### 2.1 Data Completeness of Zero-Flow Catchments
- **Dynamic Forcings**: **100% complete**. All 159 catchments possess complete daily forcing files (14,976 days from 1980 to 2020) across all 19 meteorological and soil moisture variables.
- **Static Attributes**: **100% complete** across topography, climate indices, and land cover. Standard soil and geology attributes match the gauged dataset.
- **Observed Streamflow**: **0.00% availability**.

### 2.2 Spatial Overlap with Gauged Catchments
Spatial intersection analysis of catchment polygons revealed a critical structural property:
- **Nested Within Gauged Basins**: **146 out of 159 zero-flow catchments** are geographically nested inside larger gauged catchments already in CAMELS-IND (e.g. upstream sub-catchments of the Godavari, Krishna, Narmada, and Mahanadi basins).
- **Containing Gauged Basins**: **54 zero-flow catchments** delineate downstream confluence points that physically enclose smaller upstream gauged catchments.
- **Completely Isolated Zero-Flow Catchments**: **0 catchments**.

### 2.3 Formal Classification of Zero-Flow Catchments

Every one of the 159 zero-flow catchments was classified into one of three operational categories:

| Classification | Count | Criteria | Hydrological & Modeling Rationale |
| :--- | :--- | :--- | :--- |
| **C. EXCLUDE_FROM_UNGAUGED_TEST** | **145** | $>85\%$ spatial overlap with gauged catchment, area extreme ($>100,000\text{ km}^2$ or $<100\text{ km}^2$), or heavy dam regulation ($>50$ large dams). | **Severe Spatial Leakage**: Runoff from these catchments flows directly into gauged downstream stations. Training on the downstream gauge while testing on the upstream nested sub-basin constitutes severe spatial overlap. Furthermore, no streamflow exists to evaluate accuracy. |
| **B. POSSIBLE_UNGAUGED_TARGET** | **9** | Moderate dam regulation ($1$ to $50$ dams) or moderate nesting ($50\%$ to $85\%$ overlap). | Can be retained for regional water balance checks or basin-wide flow aggregation demonstrations. |
| **A. GOOD_UNGAUGED_TARGET** | **5** | Clean natural basins ($0$ dams), well-defined boundaries ($100$ to $2,000\text{ km}^2$), low overlap with gauged gauges, complete forcings and static attributes. | **Deployment Candidates**: Ideal for demonstration of real-world ungauged flood forecasting and spatial hydrograph generation. |

#### The 5 Verified `GOOD_UNGAUGED_TARGET` Catchments:
1. `05020` — Cauvery Basin (Site: *Mukkombu*, Area: $1,118.6\text{ km}^2$, 0 dams, clean drainage).
2. `15029` — WFRS Basin (Site: *Neeleswaram*, Area: $193.0\text{ km}^2$, 0 dams, coastal drainage).
3. `17004` — EFRS Basin (Site: *Gundlakamma*, Area: $299.5\text{ km}^2$, 0 dams, eastern coastal drainage).
4. `17010` — EFRS Basin (Site: *Kallada*, Area: $936.2\text{ km}^2$, 0 dams, clean coastal tributary).
5. `17020` — EFRS Basin (Site: *Maniyar*, Area: $786.2\text{ km}^2$, 0 dams, clean natural basin).

> [!WARNING]
> **Definitive Evaluation Limitation**: Because these 5 catchments have **0% observed streamflow**, quantitative metrics (NSE, KGE, RMSE, peak timing error, quantile loss against observations) **CANNOT** be computed for them. They must NOT be used as the primary test benchmark.

---

## 3. Investigation of the 242 Gauged Source Catchments (Task 3)

The 242 catchments with $\ge 30\%$ observed discharge were audited for temporal completeness, decadal coverage, threshold trade-offs, and spatial duplicates.

### 3.1 Observation Availability & Threshold Trade-Offs

| Flow Availability Threshold | Number of Catchments | % of Gauged Dataset | Trade-Off Analysis |
| :---: | :---: | :---: | :--- |
| **$\ge 30\%$ (Current)** | **242** | **100.0%** | **Maximum sample size and spatial diversity** across 15 river basins. Catchments between 30% and 50% average ~14 years of valid daily discharge across 1980–2020. |
| **$\ge 40\%$** | **224** | **92.6%** | Drops 18 catchments with sporadic records. |
| **$\ge 50\%$** | **205** | **84.7%** | **Optimal Quality/Quantity Balance**. Every catchment has at least 20.5 years of daily discharge. Eliminates records with excessive missingness while preserving 85% of catchments. |
| **$\ge 60\%$** | **186** | **76.9%** | Drops 56 catchments; minor loss of regional representation. |
| **$\ge 70\%$** | **158** | **65.3%** | High data quality, but drops 35% of catchments; loses critical small-basin diversity in Subernarekha, Mahi, and Sabarmati. |
| **$\ge 80\%$** | **130** | **53.7%** | Severe reduction; nearly half the catchments discarded. |
| **$\ge 90\%$** | **90** | **37.2%** | Drastic data loss; inadequate spatial coverage for regional deep learning. |

### 3.2 Decadal Temporal Coverage
Gauged observations are distributed across 1980–2020:
- **1980–1989**: 144 to 163 active gauges per year.
- **1990–1999**: 201 to 216 active gauges per year.
- **2000–2009 (Peak Coverage)**: **234 to 239 active gauges per year** (98% of all gauges recording concurrently).
- **2010–2019**: 198 to 225 active gauges per year.
- **2020**: 64 active gauges (many stations had delays reporting in India-WRIS).

### 3.3 Spatial Near-Duplicates & Confluence Stations
Auditing station coordinates identified **11 pairs of gauges located within 10 km of each other**:
- `4021` & `4063` (Krishna Basin, $3.65\text{ km}$ apart, areas $5,462\text{ km}^2$ vs $1,890\text{ km}^2$).
- `4026` & `4056` (Krishna Basin, $3.27\text{ km}$ apart, areas $15,190\text{ km}^2$ vs $2,425\text{ km}^2$).
- `5004` & `5008` (Cauvery Basin, $5.50\text{ km}$ apart, tributary $3,386\text{ km}^2$ vs mainstem $53,233\text{ km}^2$).
- `12029` & `12035` (Narmada Basin, $7.82\text{ km}$ apart, tributary $3,919\text{ km}^2$ vs mainstem $72,809\text{ km}^2$).

> [!CAUTION]
> In each pair, one station gauges a tributary immediately before it enters the mainstem gauged by the other station. If one station were placed in Train and the other in Test, the model would receive catastrophic spatial proximity leakage. Grouping by entire river basin eliminates this risk completely.

---

## 4. Evaluation of Split Strategies (Task 4)

We systematically compared four distinct splitting strategies against the requirements of an ungauged flood prediction model:

| Strategy | Description | Advantages | Disadvantages | Leakage Risk | Represents Project Goal? |
| :--- | :--- | :--- | :--- | :---: | :---: |
| **Strategy A: Random Catchment Split** | Random 70/15/15 assignment of individual catchments. | Simple; uniform distribution of catchment sizes and flow. | Slices connected river networks; splits tributaries from mainstems; tests trivial spatial interpolation. | **FATAL** | **NO** |
| **Strategy B: HydroBASINS Macro-Basin Split** | Entire river basins assigned exclusively to Train, Val, or Test. | Strictly zero spatial leakage; simulates regional transfer to completely unseen geographic basins. | Regional climatological shifts between Western Ghats, Deccan, and Eastern coast. | **ZERO** | **YES** |
| **Strategy C: River-Network-Aware Group Split** | Topological connected components (trees) isolated across splits. | Preserves graph structure within each basin; prevents advective flow leakage across splits. | Requires complex graph partitioning; can result in uneven partition sizes. | **ZERO** | **YES** |
| **Strategy D: Train on Gauged, Test on 159 Zero-Flow** | Train on 242 gauged catchments; test on 159 zero-discharge catchments. | Mimics deployment on completely ungauged basins. | **NO GROUND TRUTH**: Zero observed streamflow means NSE, KGE, RMSE, and Quantile Loss cannot be calculated. | High (nested overlap) | **PARTIALLY (Deployment only)** |

---

## 5. Recommended Primary Experiment: The "Pseudo-Ungauged" Setup (Task 5)

To reconcile the necessity of rigorous quantitative evaluation with the reality of an ungauged setting, we recommend a **Scientifically Defensible Dual-Track Experiment**:

```
                         242 GAUGED CATCHMENTS
                         (Flow Data >= 30%)
                                 │
           ┌─────────────────────┼─────────────────────┐
           ▼                     ▼                     ▼
      SOURCE/TRAIN           VALIDATION           HELD-OUT TEST
     (169 Catchments)      (36 Catchments)       (37 Catchments)
     - Godavari (50)       - Cauvery (21)        - Mahanadi (19)
     - Krishna (41)        - Tapi (10)           - Narmada (18)
     - WFRS (25)           - Sabarmati (5)       
     - EFRS (12)                                 
     - WFRN (10)                                 
     - Others (31)                               
           │                     │                     │
           ▼                     ▼                     ▼
     Train Regional        Tune Hyper-           STRICTLY UNGAUGED:
     TCN + GAT             parameters            - Hide streamflow!
     Architecture                                - Predict using ONLY
                                                   met/soil + static +
                                                   river network GAT
                                                       │
                                                       ▼
                                                 EVALUATION:
                                                 Compare predictions
                                                 against held-out
                                                 ground-truth streamflow
                                                 (NSE, KGE, Quantile Loss)

                                 │
                                 ▼
                     OPERATIONAL FORWARD DEPLOYMENT
                     (5 Clean Zero-Flow Catchments)
                     - Run trained model forward
                     - Generate daily hydrographs & flood risk maps
                     - Explicitly report: No ground-truth verification possible
```

### Why Mahanadi and Narmada Form the Optimal Unseen Test Set
1. **Zero Spatial / Upstream-Downstream Overlap**: Mahanadi drains east into the Bay of Bengal; Narmada drains west into the Arabian Sea. Neither shares any tributary, river reach, or basin boundary with the training or validation basins.
2. **Exceptionally High Observation Quality**: The 37 test catchments exhibit a **median flow availability of 94.3%** (mean 86.1%), providing near-unbroken daily ground truth to evaluate peak flood quantiles and hydrograph shape.
3. **Hydroclimatic Diversity**: Mahanadi represents high-monsoon cyclonic rainfall in Eastern India; Narmada represents central continental terrain and steep rift-valley gradients.
4. **Scale Diversity**: Test catchments range from small upland tributaries ($787\text{ km}^2$) to major river trunk stations ($124,450\text{ km}^2$), testing scalability across stream orders 3 through 7.

---

## 6. Candidate Split Metadata (Task 6)

The candidate split metadata files were generated and saved in [`data/processed/`](../data/processed/):

1. **[`data/processed/candidate_source_catchments.csv`](../data/processed/candidate_source_catchments.csv)** (169 Rows):
   - Training catchments from 10 hydrologically isolated basins: Godavari, Krishna, WFRS, EFRS, WFRN, Brahmani-Baitarani, Pennar, Mahi, EFRN, Subernarekha.
   - Mean area: $19,558\text{ km}^2$ (median $4,234\text{ km}^2$, min $125\text{ km}^2$, max $307,800\text{ km}^2$).
   - Mean flow availability: $75.4\%$.
2. **[`data/processed/candidate_validation_catchments.csv`](../data/processed/candidate_validation_catchments.csv)** (36 Rows):
   - Tuning catchments from 3 isolated basins: Cauvery, Tapi, Sabarmati.
   - Mean area: $15,555\text{ km}^2$ (median $5,506\text{ km}^2$, min $362\text{ km}^2$, max $66,243\text{ km}^2$).
   - Mean flow availability: $67.0\%$.
3. **[`data/processed/candidate_unseen_test_catchments.csv`](../data/processed/candidate_unseen_test_catchments.csv)** (37 Rows):
   - Held-out test catchments from 2 isolated basins: Mahanadi and Narmada.
   - Mean area: $19,742\text{ km}^2$ (median $4,650\text{ km}^2$, min $787\text{ km}^2$, max $124,450\text{ km}^2$).
   - Mean flow availability: $86.1\%$ (median $94.3\%$).
4. **[`data/processed/catchment_groups.csv`](../data/processed/catchment_groups.csv)** (472 Rows):
   - Complete inventory of all 472 CAMELS-IND catchments classified by `target_status`:
     - `SOURCE_GAUGED_TRAIN`: 169
     - `HELD_OUT_GAUGED_VAL`: 36
     - `HELD_OUT_GAUGED_TEST`: 37
     - `EXCLUDED_UNGAUGED_NESTED`: 145 (zero-flow catchments nested inside gauged basins)
     - `EXCLUDED_LOW_OBSERVATION`: 71 (partial flow availability $0 < \text{flow} < 30\%$)
     - `POSSIBLE_UNGAUGED_DEPLOYMENT`: 9 (zero-flow with moderate regulation)
     - `OPERATIONAL_UNGAUGED_DEPLOYMENT`: 5 (clean zero-flow catchments for deployment demo)

---

## 7. Formal Leakage Verification Audit (Task 7)

Every candidate partition underwent programmatic verification against seven potential leakage vectors:

| Leakage Vector | Audit Check Description | Result | Status |
| :--- | :--- | :---: | :---: |
| **1. Upstream/Downstream Overlap** | Downstream graph traversal along HydroRIVERS reach network between Train, Val, and Test. | **0 cross-split flow paths** | **PASSED** |
| **2. HydroBASINS Overlap** | Unique continental macro-basin ID (`MAIN_BAS`) set intersection. | **0 shared `MAIN_BAS`** | **PASSED** |
| **3. Major River Basin Overlap** | CWC River Basin boundary set intersection. | **0 shared river basins** | **PASSED** |
| **4. Duplicate / Spatial Proximity** | Station coordinate distance $< 10\text{ km}$ across partitions. | **0 cross-split pairs $< 10\text{ km}$** | **PASSED** |
| **5. Feature Streamflow Leakage** | Verification that `camels_ind_hydro` and `reservoir_index` are excluded. | **Strictly excluded** | **PASSED** |
| **6. Indirect Discharge Proxies** | Verification that HydroRIVERS `DIS_AV_CMS` and `ORD_FLOW` are excluded. | **Strictly excluded** | **PASSED** |
| **7. Baseline Leakage** | Verification that `lstm_pred_streamflow` is not an input feature. | **Preserved as benchmark only** | **PASSED** |

---

## 8. Final Summary & Recommended Next Steps

### 8.1 Core Quantitative Answers
1. **Suitable source/training candidates**: **169 catchments** across 10 major basins.
2. **Suitable validation candidates**: **36 catchments** across 3 basins (Cauvery, Tapi, Sabarmati).
3. **Suitable as truly ungauged targets**: **5 clean catchments** (out of 159 zero-flow catchments). The remaining 154 are excluded due to severe spatial nesting (146), macro-scale extremes, or heavy dam regulation.
4. **Ungauged catchments with enough information for our proposed model**: **All 5 clean ungauged catchments** have 100% complete daily forcings (1980–2020), complete static attributes, and matched river network graphs. However, they have **zero observed streamflow**, making quantitative evaluation impossible.
5. **Leakage risks identified**:
   - Random splitting creates catastrophic advective flow leakage between upstream and downstream gauges.
   - Tributary/mainstem spatial near-duplicates (11 pairs within 10 km).
   - Upstream nested zero-flow catchments draining into downstream gauged stations.
   - Streamflow contamination in `camels_ind_hydro`, `reservoir_index`, and `DIS_AV_CMS`.
6. **Recommended Split Strategy**: **Strategy B/C Hybrid (River-Network-Aware Regional Basin Split)**.
7. **Scientific Justification**: It guarantees **zero spatial and hydraulic leakage** while preserving complete river network graphs within each basin for the GAT. It creates a legitimate "pseudo-ungauged" test set on 37 catchments with unbroken ground truth records, allowing rigorous evaluation of extreme flood quantiles and peak timings.

### 8.2 Recommended Next Step (STEP 3)

> [!IMPORTANT]
> **Recommended Step 3**:  
> **"Construct the graph topology matrices from HydroRIVERS and feature preprocessing pipelines (standardizing ID formatting, imputing subsoil bulk density, and generating cyclical calendar encodings) for the candidate training, validation, and test catchments."**
