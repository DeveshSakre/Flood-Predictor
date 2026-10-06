# Feature Preprocessing & Model-Ready Data Preparation Report

**Project**: ML-Based Flood Prediction for Ungauged Rivers  
**Planned Architecture**: Regional Temporal Convolutional Network (TCN) + River-Network Graph Attention Network (GAT) + Quantile Regression  
**Datasets**: CAMELS-IND, HydroRIVERS Asia, HydroBASINS Asia  
**Phase**: Step 3 — Feature Preprocessing, Scaling & ID Standardization  
**Date**: October 2026  
**Status**: Completed (Zero ML models trained, zero raw datasets altered, scalers fitted strictly on training catchments)

---

## Executive Summary

This report establishes the model-ready feature selection, imputation, normalization, and temporal engineering pipelines for our flood prediction framework. Following the strict anti-leakage principles of the ungauged catchment setting:
1. **Zero Streamflow Feature Contamination**: All streamflow signatures, historical runoff statistics, discharge-scaled infrastructure indices, and macro-hydrological model priors were permanently excluded from input features.
2. **Train-Only Preprocessing**: All scaling parameters ($\mu, \sigma$), imputation statistics (medians), and categorical encoding vocabularies were fitted **strictly on the 169 source/training catchments** (representing 2,530,944 daily time steps) and serialized to [`data/processed/scalers/`](../data/processed/scalers/). Validation, test, and ungauged catchments were transformed purely using these pre-fitted parameters.
3. **Harmonic Cyclical Temporal Encodings**: Non-linear seasonal dynamics (monsoon onset, peak precipitation, and dry season) are captured via cyclical $\sin$ and $\cos$ transformations of month and day-of-year, avoiding discontinuous boundary effects.
4. **Standardized Identity Protocol**: All catchment IDs across forcing files, static tables, streamflow matrices, shapefiles, HydroRIVERS, and HydroBASINS were unified under a canonical 5-character zero-padded string format (e.g. `'03001'`) via a master mapping table without modifying raw filenames.

---

## 1. Final Approved Feature List (Part 1)

Every feature was selected from [`reports/feature_inventory.csv`](./feature_inventory.csv) and classified into functional model input roles. The complete specification is saved in [`data/processed/final_feature_list.csv`](../data/processed/final_feature_list.csv) (159 total model-ready features).

```
Model Input Features: 159 Total Features
├── DYNAMIC FEATURES (20):
│   ├── Physical Meteorological & Soil Moisture Forcings:  16 Features
│   └── Cyclical Harmonic Calendar Encodings:               4 Features
├── STATIC CATCHMENT FEATURES (129):
│   ├── Topography & Geometry (DEM-delineated):             10 Features
│   ├── Geographic Coordinates (Outlet Spatial Anchors):     2 Features
│   ├── Long-Term Climate Normals (1980–2020):              38 Features
│   ├── Land Cover & Dynamic Vegetation (ESA/MODIS):        15 Features (12 numeric + 3 one-hot)
│   ├── Soil Physical & Hydraulic Properties (SoilGrids):   29 Features (26 numeric + 3 one-hot)
│   ├── Subsurface Geology & Lithology (GLiM/GLHYMPS):      12 Features (5 numeric + 7 one-hot)
│   └── Anthropogenic Infrastructure & Population:          23 Features
└── RIVER GRAPH FEATURES (10):
    ├── Graph Node Features (HydroRIVERS Reach Scales):      6 Features
    └── Graph Edge Features (Hydraulic Routing Attributes):  4 Features
```

### 1.1 Strict Exclusion Verification
The following variables were programmatically verified as **permanently excluded** from input features:
- `pet(mm/day)`: Excluded due to the 1980 data gap (366 missing days). Replaced by `pet_gleam(mm/day)` (100% complete).
- `tavg(C)` & scalar `wind(m/s)`: Excluded due to exact mathematical collinearity with `tmax`/`tmin` and `wind_u`/`wind_v`.
- All 73 hydrological signatures in `camels_ind_hydro.csv` (`q_mean`, `runoff_ratio`, `slope_fdc`, `bfi`, `q_10`–`q_90`, etc.): Excluded due to fatal target streamflow leakage.
- `reservoir_index`: Excluded because it divides reservoir capacity by observed multi-year annual discharge.
- `flow_availability`: Excluded because it discloses target observation coverage.
- `lstm_pred_streamflow`: Excluded from input features; retained strictly as an external baseline benchmark.
- HydroRIVERS `DIS_AV_CMS` & `ORD_FLOW`: Excluded to avoid injecting global hydrological simulation biases into our ungauged physical model.

---

## 2. Feature Quality, Variance & Outlier Audit (Part 2)

Statistical profiling of all selected features across the 169 training catchments revealed the following properties:

### 2.1 Dynamic Feature Summary (Train Set: 2,530,944 Daily Timesteps)

| Feature Name | Physical Role | Units | Raw Missing | Train Mean ($\mu$) | Train Std ($\sigma$) | Min | Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `prcp(mm/day)` | Daily areal precipitation | mm/day | 0.00% | 2.871 | 9.614 | 0.000 | 487.32 |
| `tmax(C)` | Maximum daily 2m temperature | °C | 0.00% | 32.842 | 4.312 | 12.150 | 47.85 |
| `tmin(C)` | Minimum daily 2m temperature | °C | 0.00% | 21.054 | 4.886 | 2.450 | 33.20 |
| `srad_lw(w/m2)` | Downward longwave radiation | $\text{W/m}^2$ | 0.00% | 382.416 | 32.185 | 224.100 | 468.90 |
| `srad_sw(w/m2)` | Downward shortwave radiation | $\text{W/m}^2$ | 0.00% | 234.872 | 58.741 | 14.200 | 362.40 |
| `wind_u(m/s)` | 10m Zonal wind component | m/s | 0.00% | 0.428 | 2.145 | -11.850 | 14.21 |
| `wind_v(m/s)` | 10m Meridional wind component | m/s | 0.00% | 0.812 | 1.874 | -9.620 | 12.45 |
| `rel_hum(%)` | 2m Relative humidity | % | 0.00% | 58.624 | 19.842 | 8.420 | 99.85 |
| `pet_gleam(mm/day)` | Potential evapotranspiration | mm/day | 0.00% | 4.128 | 1.412 | 0.210 | 11.45 |
| `aet_gleam(mm/day)` | Actual evapotranspiration | mm/day | 0.00% | 1.942 | 1.185 | 0.000 | 7.82 |
| `evap_canopy(mm/day)` | Canopy interception evaporation | mm/day | 0.00% | 0.315 | 0.742 | 0.000 | 12.15 |
| `evap_surface(mm/day)` | Ground surface evaporation | mm/day | 0.00% | 0.628 | 0.584 | 0.000 | 6.42 |
| `sm_lvl1(kg/m2)` | Topsoil moisture (0–0.1 m) | $\text{kg/m}^2$ | 0.00% | 13.842 | 7.915 | 0.850 | 45.62 |
| `sm_lvl2(kg/m2)` | Shallow root moisture (0.1–0.35 m)| $\text{kg/m}^2$ | 0.00% | 58.741 | 24.182 | 4.210 | 132.40 |
| `sm_lvl3(kg/m2)` | Lower root moisture (0.35–1.0 m) | $\text{kg/m}^2$ | 0.00% | 162.154 | 54.312 | 15.400 | 345.10 |
| `sm_lvl4(kg/m2)` | Deep vadose moisture (1.0–3.0 m) | $\text{kg/m}^2$ | 0.00% | 498.621 | 142.815 | 42.100 | 985.40 |

### 2.2 Static Feature Distribution Audit (Train Set: 169 Catchments)
- **Zero Variance Check**: Exactly **0 static features** have zero variance in the training set.
- **Near-Zero Variance Check**:
  - `carb_rocks_frac`: 3 unique values ($0.0$, $0.167$, $0.333$) reflecting low carbonate geology in the Deccan basalt region. Retained as a valid geological proxy.
  - `soil_awsc_max`: 3 discrete capacity bins ($50, 100, 150\text{ mm/m}$). Retained.
- **Extreme Scale Differences**: Catchment areas span four orders of magnitude ($125\text{ km}^2$ to $307,800\text{ km}^2$), dam storage spans $0$ to $364,768 \times 10^3\text{ m}^3$, and elevation spans $12\text{ m}$ to $1,840\text{ m}$. StandardScaler normalizes all continuous inputs into zero mean and unit variance.

---

## 3. Train-Only Preprocessing & Imputation Strategy (Parts 3 & 4)

### 3.1 Strict Train-Only Normalization Policy
To prevent data leakage into validation, test, and ungauged partitions:
$$\mathbf{z}_{\text{val}} = \frac{\mathbf{x}_{\text{val}} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}, \quad \mathbf{z}_{\text{test}} = \frac{\mathbf{x}_{\text{test}} - \boldsymbol{\mu}_{\text{train}}}{\boldsymbol{\sigma}_{\text{train}}}$$
- Validation and test data were never pooled to calculate means, standard deviations, or quantiles.
- All fitted transformation parameters are saved in [`data/processed/scalers/`](../data/processed/scalers/):
  - [`dynamic_scaler_params.json`](../data/processed/scalers/dynamic_scaler_params.json): Mean and std for all 16 physical forcings over $2,530,944$ training days.
  - [`static_scaler.joblib`](../data/processed/scalers/static_scaler.joblib) & [`static_scaler_params.json`](../data/processed/scalers/static_scaler_params.json): Fitted Scikit-Learn StandardScaler for the 129 encoded static attributes.
  - [`graph_node_scaler.joblib`](../data/processed/scalers/graph_node_scaler.joblib) & [`graph_node_scaler_params.json`](../data/processed/scalers/graph_node_scaler_params.json): Fitted StandardScaler for the 6 HydroRIVERS reach node features.

### 3.2 Missing Value Imputation Decisions

| Feature Group | Missing Count in Train | Missing % | Imputation Strategy | Hydrological Justification |
| :--- | :---: | :---: | :--- | :--- |
| `bulkdens_sub_mean` | 0 in Train (1 in Test: `12031`) | $0.0\%$ (Train) | **Train-Set Median ($1.340\text{ g/cm}^3$)** | Invariant subsoil physical density; median of training catchments provides an unbiased, leakage-free prior. |
| Dam Timelines (`first_dam_year`, `last_dam_year`) | 22 in Train | $13.02\%$ | **Impute to $0.0$** | Missing values occur exclusively in catchments with zero dams. A value of $0.0$ acts as a clean physical indicator of dam absence. |
| Dam Operational Fractions (`irrigation_frac`, `hydroelec_frac`, etc.) | 21 in Train | $12.43\%$ | **Impute to $0.0$** | Missing values represent dam-free basins. Zero dams implies zero fractional operational allocation. |
| Secondary Geology (`geol_class_2nd_frac`) | 99 in Train | $58.58\%$ | **Impute to $0.0$** | Missing value denotes $100\%$ coverage by the primary lithology; secondary fraction is strictly zero. |

The complete imputation rules are serialized in [`data/processed/scalers/static_imputation_values.json`](../data/processed/scalers/static_imputation_values.json).

---

## 4. Temporal Features & Cyclical Encodings (Part 5)

Standard integer calendar indicators (e.g. Month $1$ to $12$, Day-of-Year $1$ to $365$) introduce artificial numerical discontinuities between December 31 and January 1. Furthermore, using raw `year` (e.g. $1980, \dots, 2020$) encourages neural networks to learn spurious linear time trends that fail to generalize to future ungauged periods.

### 4.1 Harmonic Cyclical Encodings
We implemented continuous 2D harmonic projections for annual seasonality:

$$\sin_{\text{month}} = \sin\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right), \quad \cos_{\text{month}} = \cos\left(\frac{2\pi \cdot (\text{month} - 1)}{12}\right)$$

$$\sin_{\text{doy}} = \sin\left(\frac{2\pi \cdot (\text{doy} - 1)}{\text{days\_in\_year}}\right), \quad \cos_{\text{doy}} = \cos\left(\frac{2\pi \cdot (\text{doy} - 1)}{\text{days\_in\_year}}\right)$$

### 4.2 Physical Rationale
1. **Monsoon Phase Mapping**: Captures the progression of the Indian Southwest Monsoon (June onset, July–August peak, September withdrawal) and Northeast Monsoon (October–November in Southern India) as smooth orbits on the unit circle.
2. **Boundary Continuity**: December 31 ($(\cos \approx 1, \sin \approx 0)$) smoothly connects to January 1 without numerical distance distortion.
3. **No Target Leakage**: Derived purely from calendar indices.

---

## 5. Catchment ID Standardization Protocol (Part 6)

To resolve inconsistencies between 5-digit zero-padded filenames, integer CSV tables, and shapefile DBFs without renaming raw dataset files, a master mapping table was constructed and saved in [`data/processed/id_mapping.csv`](../data/processed/id_mapping.csv) (472 catchments).

### 5.1 ID Cross-Reference Architecture

| Component | Raw Representation | Standard Internal Format | Verification Status |
| :--- | :--- | :--- | :---: |
| **CAMELS Attributes** | `int64` (e.g. `3001`) | `str(gid).zfill(5)` $\rightarrow$ `'03001'` | Verified 472/472 |
| **CAMELS Forcings** | `str` (e.g. `'03001.csv'`) | `str(gid).zfill(5)` $\rightarrow$ `'03001'` | Verified 472/472 |
| **CAMELS Streamflow** | `str` (e.g. `'3001'`) | `str(col).zfill(5)` $\rightarrow$ `'03001'` | Verified 472/472 |
| **CAMELS Shapefiles** | `str` (e.g. `'03001'`) | `str(gid).zfill(5)` $\rightarrow$ `'03001'` | Verified 472/472 |
| **HydroRIVERS Reach** | `int64` (e.g. `40061234`) | Foreign Key `HYRIV_ID` | Verified 242/242 |
| **HydroBASINS Polygon** | `int64` (e.g. `4060027780`)| Foreign Key `MAIN_BAS` / `PFAF_ID_L06` | Verified 242/242 |

Zero catchments were unmatched across required dataset components.

---

## 6. Model-Ready Input Structures for Regional TCN & GAT (Parts 9 & 10)

### 6.1 Tensor Dimension Planning (Future TCN)
For each partition, temporal dynamic features will be formatted as 3D arrays:
$$\mathbf{X}_{\text{dyn}} \in \mathbb{R}^{N_{\text{catchments}} \times T_{\text{days}} \times 20}$$
- $N_{\text{train}} = 169$, $N_{\text{val}} = 36$, $N_{\text{test}} = 37$, $N_{\text{ungauged}} = 5$.
- $T_{\text{days}} = 14,976$ (1980-01-01 to 2020-12-31).
- Feature dimension $D_{\text{dyn}} = 20$ (16 normalized physical forcings + 4 cyclical calendar features).

Static catchment features are stored in invariant 2D matrices:
$$\mathbf{X}_{\text{static}} \in \mathbb{R}^{N_{\text{catchments}} \times 129}$$
- Stored separately from the temporal tensor and broadcast or concatenated into the TCN dense head alongside the GAT topological embeddings.

### 6.2 Streamflow Target Protocol (Part 10)
Streamflow observations are strictly isolated from input tensors:
- **Train Partition (169 catchments)**: Observed daily streamflow ($m^3/s$) is loaded as the training label $\mathbf{Y}_{\text{train}} \in \mathbb{R}^{N_{\text{train}} \times T_{\text{days}}}$. Daily timesteps where observed streamflow is NaN are masked out of the quantile loss calculation via a boolean validity mask.
- **Validation Partition (36 catchments)**: Observed streamflow is used exclusively to compute validation loss and early-stopping criteria.
- **Held-Out Test Partition (37 catchments)**: Streamflow is completely withheld during model execution. The model predicts flood quantiles conditioned solely on $(\mathbf{X}_{\text{dyn}}, \mathbf{X}_{\text{static}}, \mathcal{G}_{\text{river}})$. Predicted discharge is compared against ground truth observations **only during post-prediction benchmark evaluation**.
- **Operational Ungauged Demonstration (5 catchments)**: Streamflow is permanently non-existent ($0\%$ availability). Predictions are generated for forward flood hazard mapping without computing quantitative loss.

---

## 7. Programmatic Validation Audit (Part 12)

All 9 programmatic validation checks were executed via [`scratch/run_validation_checks.py`](../scratch/run_validation_checks.py):

| Check # | Validation Requirement | Verification Mechanism | Result |
| :---: | :--- | :--- | :---: |
| **1** | No input feature is streamflow-derived | Scanned 159 features for discharge/hydro keywords | **PASSED** |
| **2** | Test observed streamflow absent from inputs | Confirmed streamflow omitted from feature catalogue | **PASSED** |
| **3** | Ungauged streamflow absent from inputs | Confirmed zero-flow catchments have no flow inputs | **PASSED** |
| **4** | Scalers fitted strictly on training set | Verified sample count matches $169 \times 14,976 = 2,530,944$ | **PASSED** |
| **5** | Zero train $\rightarrow$ test graph leakage | Verified edge endpoints belong strictly to partition nodes | **PASSED** |
| **6** | Complete inputs for all test catchments | Verified 14,976 days of forcings and static attributes | **PASSED** |
| **7** | Valid target streamflow for training set | Verified all 169 training gauges have $\ge 30\%$ observations | **PASSED** |
| **8** | Identity mapping consistency | Verified 5-digit padding across all 472 catchments | **PASSED** |
| **9** | Raw dataset immutability | Verified zero raw data files modified or moved | **PASSED** |
