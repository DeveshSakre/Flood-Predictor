# ML-Based Flood Prediction for Ungauged Rivers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](requirements.txt)
[![Status: Under Development](https://img.shields.io/badge/Status-Under%20Development-orange.svg)](#current-project-status)

**Framework Architecture**: Regional Temporal Convolutional Network (TCN) + River-Network Graph Attention Network (GAT) + Quantile Regression  
**Study Region**: Peninsular India (472 CAMELS-IND Catchments)  
**Primary Datasets**: CAMELS-IND, HydroRIVERS Asia, HydroBASINS Asia  

---

## 1. Project Objective

The overarching goal of this project is to develop a deep-learning hydrological modeling framework capable of predicting daily streamflow and extreme flood hazard quantiles ($P_{10}, P_{50}, P_{90}$) in **genuinely ungauged river basins**. 

In an operational ungauged setting, target rivers possess **no historical streamflow gauges**. Therefore, our model must map directly from continuous meteorological forcing, static physiographic characteristics, and upstream river-network graph topology to river discharge without ever using historical target streamflow as an input feature.

---

## 2. Problem Statement

Traditional conceptual hydrological models require extensive calibration against years of observed streamflow at the target river section, rendering them ineffective or unreliable in ungauged regions. While deep neural networks (e.g. LSTMs) have demonstrated high accuracy in gauged basins, conventional machine learning setups suffer from severe failure modes when applied to ungauged rivers:
1. **Advective River Network Leakage**: Random catchment-level train/test splits place upstream and downstream gauges along the same river system into both training and evaluation sets. Because downstream streamflow is physically driven by upstream runoff, models learn trivial spatial interpolation rather than genuine rainfall-runoff physics.
2. **Ignoring Fluvial Topology**: Standard temporal models treat each catchment as an isolated point in space, failing to route tributary discharge downstream along the river network.
3. **Deterministic Limitations**: Floods are extreme events accompanied by high uncertainty. Deterministic point forecasts fail to convey critical prediction bounds during severe monsoonal extremes.

Our planned framework addresses these gaps by coupling **Temporal Convolutional Networks (TCN)** for multi-scale rainfall-runoff dynamics, **Graph Attention Networks (GAT)** for topological river routing, and **Quantile Regression** for calibrated uncertainty quantification.

---

## 3. Datasets

1. **CAMELS-IND (v2.2, March 2025)**:
   - 472 catchments across Peninsular India spanning 15 major river basins.
   - 41 years of continuous daily hydrometeorological forcing (1980–2020, 14,976 timesteps) including IMD rainfall/temperature, IMDAA reanalysis radiation/humidity/wind/soil moisture, and GLEAM evapotranspiration.
   - 218 static catchment attributes spanning topography, climate normals, land cover, soil, geology, and dams.
   - Daily observed discharge compiled from Central Water Commission (CWC) gauging stations.
2. **HydroRIVERS Asia (v1.0)**:
   - 1,428,959 river reaches providing explicit directed downhill routing (`NEXT_DOWN`), reach lengths, upstream contributing drainage areas (`UPLAND_SKM`), and Strahler stream orders (`ORD_STRA`).
3. **HydroBASINS Asia (v1c)**:
   - 12 hierarchical polygon layers structured via Pfafstetter topological codes, enabling macro-basin grouping and regional boundaries.

---

## 4. Current Experimental Setup & Leakage-Safe Splits

To ensure scientifically defensible evaluation, we established a **River-Network-Aware Regional Basin Split** that guarantees **strictly zero upstream-downstream, zero river-basin, and zero HydroBASINS overlap** between training, validation, and test sets:

```
                         242 GAUGED CATCHMENTS (>=30% Flow)
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
     TCN + GAT             parameters            - Streamflow HIDDEN!
     Architecture                                - Conditioned ONLY on
                                                   met/soil + static +
                                                   river network graph
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

### Key Partition Statistics
- **Source/Train (169 catchments)**: 10 isolated river basins. Mean area: $19,558\text{ km}^2$, mean flow availability: $75.4\%$.
- **Validation (36 catchments)**: 3 isolated basins (Cauvery, Tapi, Sabarmati). Mean area: $15,555\text{ km}^2$, mean flow availability: $67.0\%$.
- **Held-Out Test (37 catchments)**: 2 completely independent basins (Mahanadi and Narmada). Mean area: $19,742\text{ km}^2$, **median flow availability: $94.3\%$**.
- **Operational Ungauged Demo (5 catchments)**: 5 verified, non-nested zero-flow catchments (`05020`, `15029`, `17004`, `17010`, `17020`) for real-world forward simulation.

---

## 5. Model Development Plan

> [!NOTE]
> The full neural architecture is **UNDER ACTIVE DEVELOPMENT**. We are following a phased, milestone-based implementation:

```
Phase 1: Baselines (Next Step)      Phase 2: Regional TCN        Phase 3: River GAT + Quantile
┌───────────────────────────┐       ┌───────────────────────┐    ┌───────────────────────────┐
│ • Linear / Ridge          │  ───> │ • Dilated Causal Conv │───>│ • Topological Message Pass │
│ • XGBoost (Global Static) │       │ • Multi-scale Receptive│   │ • Attention Routing       │
│ • Standard LSTM Baseline  │       │ • Temporal Embedding  │    │ • Pinball Loss (P10/50/90)│
└───────────────────────────┘       └───────────────────────┘    └───────────────────────────┘
```

---

## 6. Repository Structure

```
ml-flood-prediction-ungauged-rivers/
│
├── README.md                                          # Project overview and setup
├── LICENSE                                            # MIT License
├── .gitignore                                         # Git exclusion rules
├── requirements.txt                                   # Pip dependencies
├── environment.yml                                    # Conda environment definition
│
├── data/
│   ├── README.md                                      # Raw data placement and paths
│   ├── raw/                                           # Local raw data (NOT tracked in Git)
│   │   └── .gitkeep
│   └── processed/                                     # Model-ready metadata & graphs (TRACKED)
│       ├── README.md
│       ├── final_feature_list.csv                     # 159 approved model features
│       ├── id_mapping.csv                             # Master 5-digit canonical ID cross-reference
│       ├── catchment_groups.csv                       # Status of all 472 catchments
│       ├── candidate_source_catchments.csv            # 169 training catchments
│       ├── candidate_validation_catchments.csv        # 36 validation catchments
│       ├── candidate_unseen_test_catchments.csv       # 37 held-out test catchments
│       ├── scalers/                                   # Train-only scalers & imputation rules
│       └── graph/                                     # Directed river graphs (nodes, edges, adj)
│
├── src/                                               # Core modular source code
│   ├── data/                                          # Data loaders, preprocessing & lag features
│   ├── spatial/                                       # River network graph tools & HydroSHEDS utils
│   ├── models/                                        # Model architectures (Baselines, TCN, GAT)
│   ├── uncertainty/                                   # Pinball loss & quantile regression
│   ├── evaluation/                                    # Hydrological metrics (NSE, KGE, Peak Timing)
│   └── utils/                                         # Config parsers, logging & seeds
│
├── configs/                                           # Declarative YAML experiment configs
│   ├── data_config.yaml
│   ├── baseline_config.yaml
│   ├── tcn_config.yaml
│   ├── gat_config.yaml
│   └── quantile_config.yaml
│
├── experiments/                                       # Experiment scripts & runner logs
│   ├── README.md
│   ├── baseline/
│   ├── regional_tcn/
│   ├── regional_tcn_gat/
│   └── quantile/
│
├── notebooks/                                         # Exploratory and audit notebooks
├── results/                                           # Metric outputs, benchmark tables, plots
├── models/                                            # Model checkpoints (NOT tracked in Git)
├── reports/                                           # Formal milestone audit reports
├── docs/                                              # In-depth architectural & team docs
└── tests/                                             # Unit and integration tests
```

---

## 7. Installation & Setup

### 7.1 Clone and Environment Setup
```bash
# 1. Clone the repository
git clone https://github.com/<your-org>/ml-flood-prediction-ungauged-rivers.git
cd ml-flood-prediction-ungauged-rivers

# 2. Create virtual environment using Conda
conda env create -f environment.yml
conda activate ml-flood-ungauged

# Alternatively using Pip
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 7.2 Local Data Configuration
1. Place local raw datasets into `data/raw/` as documented in [`data/README.md`](data/README.md).
2. Configure dataset paths in [`configs/data_config.yaml`](configs/data_config.yaml).
3. Run test suite to verify data and graph integrity:
   ```bash
   pytest tests/
   ```

---

## 8. Team Collaboration Workflow

For our 5-person team, all development follows feature branches branching off `main`:
- **Member 1 (Data & Preprocessing)**: `member1-data`
- **Member 2 (TCN & Regional TCN)**: `member2-tcn`
- **Member 3 (River-Network GAT)**: `member3-gat`
- **Member 4 (Quantile Regression & Uncertainty)**: `member4-uncertainty`
- **Member 5 (Baselines, Evaluation & Integration)**: `member5-evaluation`

Refer to [`docs/team_workflow.md`](docs/team_workflow.md) for branch management, PR review, and commit standards.

---

## 9. Current Project Status

- [x] **Step 1: Data Audit & Feature Classification** (Completed — 269 variables audited, anti-leakage rules codified).
- [x] **Step 2: Leakage-Safe Catchment Split Design** (Completed — 169 Train / 36 Val / 37 Test / 5 Ungauged Demo).
- [x] **Step 3: Model-Ready Preprocessing & Graph Construction** (Completed — Train-only scalers fitted, 3 DAGs constructed).
- [ ] **Step 4: Baseline Models Implementation** (Next Step — XGBoost and standard LSTM benchmarks).
- [ ] **Step 5: Regional TCN Implementation** (Under Development).
- [ ] **Step 6: River-Network GAT Integration** (Under Development).
- [ ] **Step 7: Quantile Regression & Uncertainty Evaluation** (Under Development).

---

## 10. Data Privacy & Licensing
- Code released under the [MIT License](LICENSE).
- CAMELS-IND dataset is open for academic and research purposes; users must cite Mangukiya et al. (2025).
- HydroRIVERS and HydroBASINS datasets are provided by World Wildlife Fund (WWF) HydroSHEDS (Lehner & Grill, 2013).
