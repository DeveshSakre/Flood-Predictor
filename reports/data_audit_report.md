# Data Audit and Feature Classification Report

**Project**: ML-Based Flood Prediction for Ungauged Rivers  
**Planned Architecture**: Regional Temporal Convolutional Network (TCN) + River-Network Graph Attention Network (GAT) + Quantile Regression  
**Datasets Audited**: 
1. CAMELS-IND (All Catchments & Streamflow Sufficient Subset, v2.2)
2. HydroRIVERS Asia (v1.0 File Geodatabase)
3. HydroBASINS Asia (v1c Shapefiles, Levels 01 to 12)  
**Date of Audit**: October 2026  
**Status**: Comprehensive Data Audit & Feature Classification Complete (No raw data altered, no model trained, no splits created)

---

## Executive Summary

This report delivers a rigorous, multi-faceted data audit and formal feature classification for our machine learning flood forecasting framework across ungauged basins in Peninsular India. In an **ungauged catchment setting**, an operational model must predict streamflow and extreme flood quantiles at target river sections where **no historical streamflow observations exist**. Consequently, incorporating any feature directly or indirectly derived from target streamflow constitutes catastrophic target leakage, inflating apparent model skill during cross-validation while rendering the model entirely non-functional when deployed on true ungauged rivers.

Every candidate variable across 8 static attribute catalogues, 472 daily forcing time series files, 2 streamflow time series matrices, the 1.43-million reach HydroRIVERS Asia network, and the 12-level HydroBASINS Asia hierarchy was systematically examined, verified against official publication metadata, and classified into one of seven distinct functional categories.

---

## 1. High-Level Dataset Inventory & Summary Metrics

| Metric / Dimension | CAMELS-IND (All Catchments) | CAMELS-IND (Streamflow Sufficient) | HydroRIVERS Asia | HydroBASINS Asia (Lev01-12) |
| :--- | :--- | :--- | :--- | :--- |
| **Total Spatial Entities** | 472 Catchments | 242 Catchments ($\ge 30\%$ flow data) | 1,428,959 River Reaches | 1 to 160,508 Sub-basins (12 levels) |
| **Spatial Extent** | Peninsular India (15 basins) | Peninsular India (15 basins) | Continental Asia | Continental Asia |
| **Spatial Geometry Type** | Polygon & Point (WGS 84) | Polygon & Point (WGS 84) | MultiLineString (WGS 84) | Polygon (WGS 84) |
| **Coordinate Reference System** | EPSG:4326 | EPSG:4326 | EPSG:4326 | EPSG:4326 |
| **Temporal Coverage** | 1980-01-01 to 2020-12-31 | 1980-01-01 to 2020-12-31 | Static (1971–2000 Climatology) | Static (Topological Hierarchy) |
| **Temporal Length** | 14,976 Days (41 continuous years) | 14,976 Days (41 continuous years) | Static | Static |
| **Temporal Resolution** | Daily | Daily | Long-term Climatology | Static |
| **Variables / Fields** | 22 Dynamic + 218 Static + 2 Flow | 22 Dynamic + 218 Static + 2 Flow | 15 Attributes | 13 Attributes |
| **Total Data Records** | 7,068,672 Daily Catchment Records | 3,624,192 Daily Catchment Records | 1,428,959 Reach Records | 586,050 Polygons across 12 levels |
| **Primary Identifier** | `gauge_id` (CWC gauge code) | `gauge_id` (CWC gauge code) | `HYRIV_ID` (8 digits) | `HYBAS_ID` (10 digits) |

---

## 2. Ungauged Prediction Paradigm & Strict Anti-Leakage Rules

### 2.1 The Ungauged Prediction Scenario
In our operational paradigm:
1. The model is trained on a set of source/gauged catchments where both meteorology/catchment characteristics and historical CWC streamflow observations are available.
2. When evaluated or deployed on an unseen target catchment (the ungauged river), the model has access **only** to:
   - Continuous daily meteorology and land-surface forcings (precipitation, temperature, radiation, humidity, wind, soil moisture, evapotranspiration).
   - Invariant physiographic attributes (topography, terrain slope, soil hydraulic properties, geology, land cover fractions).
   - Upstream anthropogenic infrastructure (dam counts, reservoir storage volume from national registries).
   - Topological river network structure and basin context (HydroRIVERS graph edges, reach lengths, upstream area, Strahler order, HydroBASINS hierarchy).
3. The model **must never** utilize historical streamflow, streamflow statistics, or variables scaled by observed discharge from the target catchment.

### 2.2 Identification of Streamflow-Derived Leakage Features

Through our exhaustive audit of the 218 static attributes and documentation, **75 variables** were identified as carrying fatal streamflow-derived leakage risk:

#### A. CAMELS-IND Hydrological Signatures (`camels_ind_hydro.csv`) — 73 Variables
All 73 non-ID attributes in `camels_ind_hydro.csv` are calculated directly from CWC historical daily streamflow records. These include:
- Central Tendency & Runoff Ratios: `q_mean`, `runoff_ratio`, `annual_q`, `mean_anum_flow`.
- Flow Duration Curve (FDC) & Regime Signatures: `streamflow_elas`, `slope_fdc`, `bfi` (baseflow index), `q_cv`, `gini_flow`, `cen_time`.
- Flow Percentiles: `q_10`, `q_25`, `q_50`, `q_75`, `q_90`, `q_zero`.
- Low-Flow and High-Flow Extremes: `q_low_days`, `freq_q_low`, `q_high_days`, `freq_q_high`, `annual_max_1day`, `annual_max_3day`, `annual_max_7day`, `annual_max_30day`, `annual_max_90day`, `annual_min_7day`.
- Timing & Seasonality: `month_1day_max`, `month_1day_min`, `doy_min_flow`, `doy_max_flow`, `doy_min_flow_7`, `doy_max_flow_7`.
- Monthly Mean and CV Flows: `mean_jan_flow` through `mean_dec_flow` (12 vars) and `cv_jan_flow` through `cv_dec_flow` (12 vars).
- Seasonal Flow Signatures: `mean_swmn_flow`, `mean_atmn_flow`, `mean_wint_flow`, `mean_sumr_flow`, `q_mean_swmn`, `q_5_swmn`, `q_25_swmn`, `q_50_swmn`, `q_75_swmn`, `q_95_swmn`.
- Hydrograph Dynamics & Alteration: `rise_rate_mean`, `rise_rate_median`, `rise_days`, `fall_rate_mean`, `fall_rate_median`, `fall_days`, `num_hyd_alt`.

> [!CAUTION]
> **Fatal Leakage Alert**: Every single one of these 73 variables requires observed target streamflow to be computed. For catchments lacking streamflow data in CAMELS-IND, these attributes are 51.7% to 70.1% null (244 to 331 missing values). Using them as model inputs is **STRICTLY PROHIBITED**.

#### B. The `reservoir_index` Attribute (`camels_ind_anth.csv`)
- **Audit Investigation**: `reservoir_index` is missing in 244 out of 472 catchments (exactly matching the 244 catchments that lack `q_mean`).
- **Definitive Provenance**: The official CAMELS-IND documentation (*ESSD Table 3*) explicitly defines `reservoir_index` as:
  $$\text{reservoir\_index} = \frac{\text{Total Reservoir Storage Volume } (\text{m}^3)}{\text{Multi-year Mean Annual Streamflow } (\text{m}^3/\text{yr})}$$
- **Classification**: **LEAKAGE_RISK / EXCLUDE_INITIAL**. Because the denominator is observed multi-year discharge, this attribute cannot be computed for a genuinely ungauged catchment. While raw reservoir storage (`res_store_sum` in $10^3\text{ m}^3$ or `total_storage` in $\text{m}^3$) is purely infrastructure-derived and safe to use, `reservoir_index` is contaminated by streamflow.

#### C. The `flow_availability` Attribute (`camels_ind_name.csv`)
- Represents the percentage of valid streamflow observation days from 1980 to 2020.
- Classified as **LEAKAGE_RISK / METADATA**. It discloses the temporal completeness of the target gauge's record.

#### D. Existing Baseline Candidate: `lstm_pred_streamflow.csv`
- A regionally trained LSTM model prediction from Mangukiya et al. (2023) provided for all 472 catchments (14,976 days, 0 nulls).
- Classified as **EXCLUDE_INITIAL / BENCHMARK**. It represents a pre-existing machine learning model prediction. Feeding it into our Regional TCN + GAT would turn our framework into a meta-model/stacking ensemble rather than an independent flood predictor. It will be preserved exclusively as an external benchmark to evaluate performance gains.

#### E. HydroRIVERS Discharge Attributes: `DIS_AV_CMS` & `ORD_FLOW`
- **Audit Investigation**: `DIS_AV_CMS` provides long-term average discharge in $\text{m}^3/\text{s}$ for each river reach. `ORD_FLOW` provides an integer classification (1 to 10) based on logarithmic thresholds of `DIS_AV_CMS`.
- **Definitive Provenance**: The HydroRIVERS Technical Documentation (*Section 2.2*) states that `DIS_AV_CMS` was simulated using the global WaterGAP 2.2 hydrological water balance model (1971–2000 runoff downscaled to 15 arc-seconds) and calibrated against GRDC stations globally.
- **Ungauged Setting Assessment**: While `DIS_AV_CMS` is derived from an external global model rather than local target streamflow observations, using it in an ungauged river setting carries substantial risks:
  1. *Model bias propagation*: Systematic biases of the WaterGAP macro-scale hydrologic model would be injected directly into our neural network.
  2. *Loss of physical realism*: The model could learn to simply rescale WaterGAP predictions rather than learning the physical rainfall-runoff relationship from terrain, soil, land cover, and reanalysis forcing.
- **Classification**: **EXCLUDE_INITIAL**. `DIS_AV_CMS` and `ORD_FLOW` should be excluded from initial model features. Topological stream orders (`ORD_STRA` and `ORD_CLAS`) and upstream area (`UPLAND_SKM`) represent pure physiographic/topological scaling and are safe.

---

## 3. Dynamic Features Audit (Daily Forcing Data)

The 472 daily forcing files in `catchment_mean_forcings/` cover **14,976 continuous days** (1980-01-01 to 2020-12-31, exactly 41 years). Every file contains exactly 14,976 rows and 22 columns (total 7,068,672 catchment-day observations).

| Variable Name | Description | Units | Source | Missing % | Classification | Candidate Input | Reason & Analytical Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `prcp(mm/day)` | Daily areal precipitation | mm/day | IMD | **0.00%** | DYNAMIC_INPUT | **YES** | Primary hydrological driver of overland and channel runoff. |
| `tmax(C)` | Daily maximum temperature | °C | IMD | **0.00%** | DYNAMIC_INPUT | **YES** | Controls thermal extremes and diurnal evaporation demand. |
| `tmin(C)` | Daily minimum temperature | °C | IMD | **0.00%** | DYNAMIC_INPUT | **YES** | Controls nocturnal cooling and dew-point potential. |
| `tavg(C)` | Daily averaged temperature | °C | Derived | **0.00%** | EXCLUDE_INITIAL | **NO** | Exact collinearity: defined as $(t_{max} + t_{min})/2$. Redundant. |
| `srad_lw(w/m2)` | Downward longwave radiation | $\text{W/m}^2$ | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Atmospheric radiative cooling/cloud cover indicator. |
| `srad_sw(w/m2)` | Downward shortwave radiation | $\text{W/m}^2$ | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Incoming solar radiation driving surface energy flux and ET. |
| `wind_u(m/s)` | Zonal (East-West) 10m wind | m/s | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Vector component of aerodynamic moisture transport. |
| `wind_v(m/s)` | Meridional (North-South) 10m wind | m/s | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Vector component of aerodynamic moisture transport. |
| `wind(m/s)` | Scalar 10m wind speed | m/s | Derived | **0.00%** | EXCLUDE_INITIAL | **NO** | Algebraic redundancy: $\sqrt{u^2 + v^2}$. Omit to reduce collinearity. |
| `rel_hum(%)` | Near-surface 2m relative humidity | % | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Atmospheric saturation deficit and vapor pressure proxy. |
| `pet(mm/day)` | Potential evapotranspiration | mm/day | Singer | **2.44%** | EXCLUDE_INITIAL | **NO** | **Data gap**: Missing entire year 1980 (366 days) in all 472 files. |
| `pet_gleam(mm/day)` | GLEAM potential evapotranspiration | mm/day | GLEAM | **0.00%** | DYNAMIC_INPUT | **YES** | Complete 1980–2020 record; replaces `pet(mm/day)` without gaps. |
| `aet_gleam(mm/day)` | GLEAM actual evapotranspiration | mm/day | GLEAM | **0.00%** | DYNAMIC_INPUT | **YES** | Actual land surface moisture loss; complete 1980–2020 record. |
| `evap_canopy(mm/day)` | Canopy interception evaporation | mm/day | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Forest/canopy rainfall interception loss (corrected unit in v2.2). |
| `evap_surface(mm/day)` | Soil surface evaporation | mm/day | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Bare soil/ground surface evaporation rate (corrected unit in v2.2). |
| `sm_lvl1(kg/m2)` | Soil moisture layer 1 (0–0.1 m) | $\text{kg/m}^2$ | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Surface topsoil saturation; critical for fast infiltration excess runoff. |
| `sm_lvl2(kg/m2)` | Soil moisture layer 2 (0.1–0.35 m) | $\text{kg/m}^2$ | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Shallow root zone moisture; governs subsurface infiltration. |
| `sm_lvl3(kg/m2)` | Soil moisture layer 3 (0.35–1.0 m) | $\text{kg/m}^2$ | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Intermediate root zone moisture; governs interflow potential. |
| `sm_lvl4(kg/m2)` | Soil moisture layer 4 (1.0–3.0 m) | $\text{kg/m}^2$ | IMDAA | **0.00%** | DYNAMIC_INPUT | **YES** | Deep vadose zone soil moisture; controls baseflow recharge. |
| `year` | Calendar year | year | Index | **0.00%** | METADATA | **NO** | Temporal index (1980–2020). |
| `month` | Calendar month | month | Index | **0.00%** | METADATA | **NO** | Temporal index (1–12); transformable to harmonic sin/cos encodings. |
| `day` | Calendar day | day | Index | **0.00%** | METADATA | **NO** | Temporal index (1–31). |

### 3.1 Potential Rolling and Antecedent Features for Future Feature Engineering
Although no feature transformations have been created yet, our audit confirms that the following variables possess the requisite quality and zero-missingness to construct lag/rolling features later:
- **Antecedent Precipitation Index (API)**: 3-day, 7-day, 14-day, and 30-day exponentially weighted cumulative rainfall from `prcp(mm/day)`.
- **Soil Moisture Deficit / Change ($\Delta \text{SM}$)**: Multi-layer moisture gradients from `sm_lvl1` to `sm_lvl4`.
- **Cumulative Evaporative Deficit**: $(P - \text{PET}_{\text{gleam}})$ over preceding 15 and 30 days.

---

## 4. Static Catchment Features Audit

The 8 static attribute catalogues in `attributes_csv/` encompass **218 attributes** across 472 catchments. Every attribute was inspected for completeness, variability, collinearity, and physical meaning.

### 4.1 Topography Attributes (`camels_ind_topo.csv` — 17 attributes)
Derived from MERIT DEM (Yamazaki et al., 2017) and official CWC station documentation:
- **Elevation**: `elev_mean`, `elev_median`, `elev_min`, `elev_max`, `gauge_elevation` (m). Complete (0% missing). Safe static inputs describing catchment hypsometry and orographic potential.
- **Slope**: `slope_mean`, `slope_median`, `slope_max` (degrees), `dpsbar` (mean drainage slope, degrees). Complete (0% missing). Safe static inputs governing gravitational flow velocity and runoff response times.
- **`slope_min`**: 470 of 472 catchments have `slope_min = 0.0`. Classified as **EXCLUDE_INITIAL** due to near-zero variance.
- **Drainage Area**: `ghi_area` ($\text{km}^2$, DEM-delineated) vs `cwc_area` ($\text{km}^2$, official CWC reporting). Both are complete and exhibit an almost perfect correlation ($r = 0.9996$). We recommend `ghi_area` as the primary **STATIC_INPUT** because it strictly corresponds to the DEM boundary polygons, classifying `cwc_area` as **EXCLUDE_INITIAL (REDUNDANT)**.
- **Geographic Coordinates**: `ghi_lat`, `ghi_lon` vs `cwc_lat`, `cwc_lon`. Correlation $> 0.99999$. Use `ghi_lat`/`ghi_lon` as spatial positional embeddings for the GAT; classify CWC coordinates as **METADATA**.

### 4.2 Climate Indices (`camels_ind_clim.csv` — 43 attributes)
Derived purely from meteorological forcing records over 1980–2020:
- **Precipitation Indices**: `p_mean`, `p_max`, `p_mean_anum`, `p_monthly_variability`, `p_annual_variability`, `p_unif`, `high_prec_freq`, `high_prec_dur`, `max_high_prec_dur`, `low_prec_freq`, `low_prec_dur`, `max_low_prec_dur`, `asynchronicity`. All 0% missing. Safe **STATIC_INPUT**.
- **`low_prec_timing`**: 100% of all 472 catchments have the identical value `"winter"`. Classified as **EXCLUDE_INITIAL** (Zero variance).
- **`high_prec_timing`**: 457 catchments `"monsoon"`, 15 catchments `"post-monsoon"`. Classified as **EXCLUDE_INITIAL** (Extreme class imbalance, 96.8% identical).
- **Thermal & Evaporative Signatures**: `tmin_mean`, `tmax_mean`, `pet_mean`, `pet_min`, `pet_max`, `pet_mean_anum`, `pet_gleam_mean`, `aet_gleam_mean`, `evap_canopy_mean`, `evap_canopy_max`, `evap_canopy_anum`, `evap_surface_mean`, `evap_surface_min`, `evap_surface_max`, `evap_surface_anum`, `aridity_p_pet`, `aridity_pet_aet`, `ai_mean`, `rel_hum_mean`, `srad_lw_mean`, `srad_sw_mean`, `wind_mean`. All 0% missing. Safe **STATIC_INPUT**.
- **`evap_canopy_min`**: 100% of all 472 catchments have value `0.0`. Classified as **EXCLUDE_INITIAL** (Zero variance).
- **Soil Moisture Climatology**: `sm_lvl1_mean`, `sm_lvl2_mean`, `sm_lvl3_mean`, `sm_lvl4_mean`. Complete (0% missing). Safe **STATIC_INPUT**.

### 4.3 Land Cover Attributes (`camels_ind_land.csv` — 14 attributes)
Derived from ESA WorldCover (10m) and MODIS MCD15A2H LAI products:
- **Land Cover Fractions**: `water_frac`, `trees_frac`, `flooded_veg_frac`, `crops_frac`, `built_area_frac`, `bare_frac`, `range_frac`, `dom_land_cover_frac`. Complete (0% missing). Safe **STATIC_INPUT** determining hydraulic roughness, interception capacity, and infiltration.
- **`dom_land_cover`**: Dominant land cover category (categorical: crops, trees, etc.). Safe **STATIC_INPUT** (one-hot or embedding).
- **Leaf Area Index (LAI)**: `lai_mean`, `lai_min`, `lai_max`, `lai_diff`. Complete (0% missing). Safe **STATIC_INPUT** reflecting seasonal vegetation dynamics and transpiration potential.

### 4.4 Soil Attributes (`camels_ind_soil.csv` — 29 attributes)
Derived from global SoilGrids250m (Hengl et al., 2017) and Pelletier et al. (2016):
- **Soil Depth & Hydraulic Conductivity**: `soil_depth` (m), `soil_conductivity_top` ($\text{cm/day}$), `soil_conductivity_sub` ($\text{cm/day}$). Complete (0% missing). Safe **STATIC_INPUT**.
- **Available Water Capacity**: `soil_awc_top`, `soil_awc_sub`, `soil_awsc_min`, `soil_awsc_max`, `soil_awsc_major` ($\text{mm/m}$). Complete (0% missing). Safe **STATIC_INPUT**.
- **Texture Fractions**: `sand_frac_top`, `sand_frac_sub`, `silt_frac_top`, `silt_frac_sub`, `clay_frac_top`, `clay_frac_sub`, `gravel_frac_top`, `gravel_frac_sub` (%). Complete (0% missing). Safe **STATIC_INPUT**.
- **Bulk Density**:
  - `bulkdens_top_major`, `bulkdense_top_mean` ($\text{g/cm}^3$): Complete (0% missing). Safe **STATIC_INPUT**.
  - `bulkdens_sub_major`: **136 missing values (28.81%)**. Classified as **EXCLUDE_INITIAL**.
  - `bulkdens_sub_mean`: **2 missing values (0.42%)**. Recommended as primary **STATIC_INPUT** after standard median imputation.
- **Organic Carbon & Water Table**: `org_carb_top_major`, `org_carb_top_mean`, `org_carb_sub_major`, `org_carb_sub_mean`, `organic_frac_top`, `organic_frac_sub`, `wtd` (water table depth, m). Complete (0% missing). Safe **STATIC_INPUT**.
- **`hsg_major`**: Hydrologic Soil Group (C: 231, D: 230, C/D: 11). Complete. Safe categorical **STATIC_INPUT**.

### 4.5 Geology Attributes (`camels_ind_geol.csv` — 8 attributes)
Derived from GLiM (Hartmann & Moosdorf, 2012) and GLHYMPS (Huscroft et al., 2018):
- `geol_porosity` (fraction), `geol_permeability` ($\text{m}^2$, $\log_{10}$), `carb_rocks_frac` (fraction): Complete (0% missing). Safe **STATIC_INPUT**.
- `geol_class_1st` (dominant lithology), `geol_class_1st_frac`: Complete. 9 lithological classes (Basic Volcanic: 179, Metamorphic: 176, Acid Plutonic: 39, Unconsolidated: 32, etc.). Safe **STATIC_INPUT**.
- `geol_class_2nd` (secondary lithology), `geol_class_2nd_frac`: 255 catchments (54.02%) have NaN in `geol_class_2nd` because a single lithological unit covers 100% of the catchment. Safe **STATIC_INPUT** with categorical `"None"` filling for NaNs and 0.0 for `geol_class_2nd_frac`.

### 4.6 Anthropogenic Attributes (`camels_ind_anth.csv` — 26 attributes)
Derived from India-WRIS, GRaND v1.3, WorldPop, and Meiyappan & Jain (2012):
- **Dams & Storage Infrastructure**:
  - `num_dams` (count) & `res_store_sum` ($10^3\text{ m}^3$): From India-WRIS. Complete (0% missing). 74 catchments have 0 dams and 0 storage. Safe **STATIC_INPUT**.
  - `n_dams` (count) & `total_storage` ($\text{m}^3$): From GRaND. Correlation with `num_dams` is $r = 0.9822$. Classified as **EXCLUDE_INITIAL (REDUNDANT)** to prevent multicollinearity with India-WRIS data.
  - `first_dam_year`, `last_dam_year`: 74 missing values (15.68%) corresponding to catchments with 0 dams. Safe **STATIC_INPUT** (fill missing with sentinel 0 or relative age feature).
  - Dam usage fractions: `irrigation_frac`, `hydroelec_frac`, `drinking_frac`, `flood_frac`, `overflow_frac`, `navigation_frac`, `tailing_frac`. 66 missing values when no dams exist. Safe **STATIC_INPUT** (impute missing to 0.0).
  - `reservoir_index`: **LEAKAGE_RISK / EXCLUDE_INITIAL** (Directly divides by annual streamflow; missing in 244 catchments).
- **Population Density**: `pop_density_2000`, `2005`, `2010`, `2015`, `2020` ($\text{people/km}^2$). Complete. Safe **STATIC_INPUT**.
- **Historical Land Use Dynamics**: `urban_frac_1985`, `1995`, `2005` & `crops_frac_1985`, `1995`, `2005`. Complete. Safe **STATIC_INPUT**.

---

## 5. HydroRIVERS & HydroBASINS Spatial Network Audit

### 5.1 HydroRIVERS Asia (`HydroRIVERS_v10_as.gdb`)
The Asian extent of HydroRIVERS contains **1,428,959 river reaches** in EPSG:4326. Every reach represents a discrete stream segment with distinct topological and geometric properties.

| Field Name | Data Type | Physical Meaning | Topological / Network Role | Classification | Ungauged Suitability |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `HYRIV_ID` | `int64` | 8-digit unique reach identifier | Node / Edge primary key in graph | METADATA | Graph Index |
| `NEXT_DOWN` | `int64` | `HYRIV_ID` of next downstream reach | Directed adjacency matrix edge $(u \rightarrow v)$ | RIVER_NETWORK | **YES (Topology)** |
| `MAIN_RIV` | `int64` | `HYRIV_ID` of terminal basin outlet | Macro-network membership / clustering | RIVER_NETWORK | **YES (Clustering)** |
| `LENGTH_KM` | `float64` | Physical segment length (km) | Edge routing distance / weight | RIVER_NETWORK | **YES (Edge Weight)** |
| `DIST_DN_KM` | `float64` | Distance from reach outlet to ocean (km) | Reach position along macro-longitudinal profile | RIVER_NETWORK | **YES (Node Feature)** |
| `DIST_UP_KM` | `float64` | Distance to headwater divide (km) | Upstream longitudinal flow distance | RIVER_NETWORK | **YES (Node Feature)** |
| `CATCH_SKM` | `float64` | Incremental reach catchment area ($\text{km}^2$) | Local incremental lateral inflow scale | RIVER_NETWORK | **YES (Node Feature)** |
| `UPLAND_SKM` | `float64` | Total upstream contributing area ($\text{km}^2$) | Cumulative basin drainage scale | RIVER_NETWORK | **YES (Node Feature)** |
| `ENDORHEIC` | `int64` | Binary flag (0 = exorheic, 1 = endorheic) | Inland sink drainage boundary condition | RIVER_NETWORK | **YES (Node Feature)** |
| `ORD_STRA` | `int64` | Strahler stream order (1 to 9) | Hierarchical branching complexity | RIVER_NETWORK | **YES (Node Feature)** |
| `ORD_CLAS` | `int64` | Classical stream order (1 = main stem) | Backbone river trunk vs tributary distinction | RIVER_NETWORK | **YES (Node Feature)** |
| `ORD_FLOW` | `int64` | Discharge-based size class (1 to 10) | Derived from modeled discharge (`DIS_AV_CMS`) | EXCLUDE_INITIAL | **NO (Flow Proxy)** |
| `DIS_AV_CMS` | `float64` | Modeled long-term discharge ($\text{m}^3/\text{s}$) | Simulated discharge from WaterGAP 2.2 | EXCLUDE_INITIAL | **NO (Model Prior)** |
| `HYBAS_L12` | `int64` | Matching HydroBASINS Level 12 `HYBAS_ID` | Foreign key linking reach to polygonal basin | RIVER_NETWORK | **YES (Foreign Key)** |

### 5.2 HydroBASINS Asia (`hybas_as_lev01_v1c.shp` to `hybas_as_lev12_v1c.shp`)
The 12 hierarchical shapefile levels provide nested catchment polygons defined by the topological Pfafstetter system:
- **Level 01**: 1 continent polygon.
- **Level 02**: 9 sub-continental regions.
- **Level 03**: 45 major basins.
- **Level 04**: 199 large regional basins.
- **Level 05**: 711 river basins.
- **Level 06**: 2,489 intermediate sub-basins (**Recommended scale for spatial train/test splits**).
- **Levels 07–12**: Increasing granularity from 8,811 (Lev 07) up to 160,508 sub-basins (Lev 12).

#### Fields Across All HydroBASINS Levels:
1. `HYBAS_ID`: Unique 10-digit sub-basin polygon identifier.
2. `NEXT_DOWN`: `HYBAS_ID` of the next downstream polygon (0 for terminal sinks/coastal outlets).
3. `NEXT_SINK`: `HYBAS_ID` of the terminal inland sink or pour point.
4. `MAIN_BAS`: Primary basin identifier corresponding to the entire connected drainage basin.
5. `DIST_SINK` & `DIST_MAIN`: Flow distances to sink and main stem (km).
6. `SUB_AREA` & `UP_AREA`: Local sub-basin area and total upstream drainage area ($\text{km}^2$).
7. `PFAF_ID`: Hierarchical decimal integer code representing the Pfafstetter code (e.g. at Level 6, a 6-digit code).
8. `ENDO` (0, 1, 2) & `COAST` (0, 1): Drainage regime and oceanic outlet indicators.
9. `ORDER` & `SORT`: Topological hierarchy and hydrologic sequence facilitating directed acyclic graph (DAG) traversals.

### 5.3 Enabling Leakage-Safe Basin Splitting via HydroBASINS
In an ungauged river setting, random catchment splitting causes severe spatial auto-correlation and upstream-downstream leakage: if a downstream gauge is in the test set while an upstream tributary gauge is in the training set, the test set is not genuinely ungauged!
- **Zero-Leakage Splitting Strategy**:
  1. Use HydroBASINS `MAIN_BAS` and `PFAF_ID` at Level 04 or Level 06 to cluster CAMELS-IND catchments into hydrologically isolated regional basin blocks.
  2. Assign entire contiguous river basins (e.g. Godavari, Krishna, Narmada, Mahanadi, Cauvery) exclusively to either Train, Validation, or Test.
  3. This guarantees that **no test catchment shares any upstream drainage, downstream drainage, or shared river channel** with any training catchment.

---

## 6. Identifier Consistency & Data Quality Audit

### 6.1 CAMELS-IND Catchment Identifier Inconsistencies

During our multi-file audit, a systematic identifier formatting discrepancy was identified across the CAMELS-IND dataset components:

| Dataset Component | File / Path | ID Representation | Sample Values | Type |
| :--- | :--- | :--- | :--- | :--- |
| **Forcing CSV Files** | `catchment_mean_forcings/*.csv` | 5-character string zero-padded | `03001.csv`, `03002.csv`, `10001.csv` | File Name |
| **Catchment Shapefiles** | `shapefiles_catchment/merged/all_catchments.shp` | 5-character string zero-padded | `'03001'`, `'03002'`, `'17025'` | `str` (DBF) |
| **Catchment Shapefiles** | `shapefiles_catchment/catchments.shp` | 5-character string zero-padded | `'03002'`, `'03005'`, `'17025'` | `str` (DBF) |
| **Attribute CSV Files** | `attributes_csv/camels_ind_*.csv` | Integer (leading zero dropped) | `3001`, `3002`, `10001` | `int64` |
| **Attribute TXT Files** | `attributes_txt/camels_ind_*.txt` | Integer | `3001`, `3002`, `10001` | `int64` |
| **Observed Streamflow** | `streamflow_observed.csv` | Unpadded integer string column name | `'3001'`, `'3002'`, `'10001'` | Column `str` |
| **LSTM Pred Streamflow** | `lstm_pred_streamflow.csv` | Unpadded integer string column name | `'3001'`, `'3002'`, `'10001'` | Column `str` |

#### Standardization Rule (Without Modifying Raw Files):
In all downstream data loaders and pipelines, standardise every gauge ID into the canonical **5-character zero-padded string**:
```python
standard_id = str(raw_id).zfill(5)  # e.g., 3001 -> '03001'
```
This guarantees flawless joining across shapefiles, attributes, daily forcings, and streamflow matrices without modifying or renaming any raw files.

---

### 6.2 Data Quality & Anomaly Report

1. **Forcing Completeness Anomaly (`pet(mm/day)`)**:
   - Exactly 366 days (the entire leap year 1980: 1980-01-01 to 1980-12-31) are NaN in `pet(mm/day)` across **all 472 catchments** (172,752 total missing values).
   - In contrast, `pet_gleam(mm/day)` has **zero missing values** across all 472 catchments and all 14,976 days.
   - *Resolution*: Exclude `pet(mm/day)` in favour of `pet_gleam(mm/day)`.
2. **Observed Streamflow Gaps**:
   - `streamflow_observed.csv` has 60.33% total missing values across the 472 columns.
   - 159 catchments have 0 valid observations (100% missing, purely ungauged).
   - 242 catchments have $\ge 30\%$ observed streamflow availability (forming the `CAMELS_IND_Catchments_Streamflow_Sufficient` subset).
   - 205 catchments have $\ge 50\%$ availability; 158 catchments have $\ge 70\%$ availability.
   - *Zero-flow analysis*: Valid streamflow records contain 0 negative values. Exactly 10.36% of valid records are true zeros ($0.0\text{ m}^3/\text{s}$), representing seasonal dry riverbeds common in Peninsular India.
3. **Constant and Near-Constant Attributes**:
   - `low_prec_timing` (`camels_ind_clim.csv`): Identical string (`"winter"`) for all 472 catchments (0 variance).
   - `evap_canopy_min` (`camels_ind_clim.csv`): Identical float (`0.0`) for all 472 catchments (0 variance).
   - `slope_min` (`camels_ind_topo.csv`): 470 of 472 catchments have `0.0`.
   - `high_prec_timing` (`camels_ind_clim.csv`): 457 of 472 catchments have `"monsoon"`.
4. **Subsoil Bulk Density Missing Values**:
   - `bulkdens_sub_major` in `camels_ind_soil.csv` is missing in 136 catchments (28.81%).
   - `bulkdens_sub_mean` is missing in only 2 catchments (0.42%).
   - *Resolution*: Adopt `bulkdens_sub_mean` as the model feature.
5. **Dam Purpose Missingness in Dam-Free Catchments**:
   - In `camels_ind_anth.csv`, `first_dam_year`, `last_dam_year`, and dam purpose fractions have 66 to 74 missing values. These catchments have 0 dams.
   - *Resolution*: Safely impute dam fractions to `0.0` and dam age to `0`.

---

## 7. Complete Feature Inventory Summary

The complete feature inventory has been generated and saved to [`reports/feature_inventory.csv`](./feature_inventory.csv). It contains **269 systematically classified variables**:

```
reports/feature_inventory.csv
├── STATIC_INPUT:             120 Features (Physiographic, terrain, soil, geology, climate normals)
├── LEAKAGE_RISK:              75 Features (73 hydro signatures + reservoir_index + flow_availability)
├── RIVER_NETWORK_FEATURE:     23 Features (HydroRIVERS connectivity & HydroBASINS hierarchy)
├── METADATA:                  20 Features (Identifiers, station names, coordinates, date indices)
├── DYNAMIC_INPUT:             16 Features (Daily rainfall, temperature, radiation, moisture, ET)
├── EXCLUDE_INITIAL:           14 Features (Redundant vars, zero-variance vars, WaterGAP priors)
└── TARGET:                     1 Feature  (streamflow_observed)
```

---

## 8. Final Summary & Key Findings

### 8.1 Total Number of CAMELS-IND Catchments
- **Total catchments in CAMELS-IND All Catchments**: **472 Catchments**.
- Delineated across 15 major river basins of Peninsular India.

### 8.2 Catchments with Sufficient Observed Streamflow
- **242 Catchments** meet the standard $\ge 30\%$ flow data availability threshold (compiled in `CAMELS_IND_Catchments_Streamflow_Sufficient`).
- **159 Catchments** have **0%** observed streamflow (never gauged in the dataset).
- *Note*: Final training selection threshold will be determined during dataset splitting design.

### 8.3 Recommended Candidate Dynamic Inputs (16 Variables)
All available daily from 1980-01-01 to 2020-12-31 with **0.00% missing values**:
1. `prcp(mm/day)` — Daily precipitation (IMD)
2. `tmax(C)` — Maximum air temperature (IMD)
3. `tmin(C)` — Minimum air temperature (IMD)
4. `srad_lw(w/m2)` — Surface downward longwave radiation (IMDAA)
5. `srad_sw(w/m2)` — Surface downward shortwave radiation (IMDAA)
6. `wind_u(m/s)` — 10m zonal wind velocity (IMDAA)
7. `wind_v(m/s)` — 10m meridional wind velocity (IMDAA)
8. `rel_hum(%)` — 2m relative humidity (IMDAA)
9. `pet_gleam(mm/day)` — Potential evapotranspiration (GLEAM v3)
10. `aet_gleam(mm/day)` — Actual evapotranspiration (GLEAM v3)
11. `evap_canopy(mm/day)` — Canopy evaporation (IMDAA)
12. `evap_surface(mm/day)` — Soil surface evaporation (IMDAA)
13. `sm_lvl1(kg/m2)` — Soil moisture 0–0.1 m (IMDAA)
14. `sm_lvl2(kg/m2)` — Soil moisture 0.1–0.35 m (IMDAA)
15. `sm_lvl3(kg/m2)` — Soil moisture 0.35–1.0 m (IMDAA)
16. `sm_lvl4(kg/m2)` — Soil moisture 1.0–3.0 m (IMDAA)

### 8.4 Recommended Candidate Static Inputs (120 Variables)
- **Topography (9)**: `elev_mean`, `elev_median`, `elev_min`, `elev_max`, `gauge_elevation`, `slope_mean`, `slope_median`, `slope_max`, `dpsbar`, `ghi_area`.
- **Spatial Positioning (2)**: `ghi_lat`, `ghi_lon` (for geographic distance and positional encodings in GAT).
- **Climate Normals (38)**: `p_mean`, `p_max`, `p_mean_anum`, `p_monthly_variability`, `p_annual_variability`, `p_unif`, `high_prec_freq`, `high_prec_dur`, `max_high_prec_dur`, `low_prec_freq`, `low_prec_dur`, `max_low_prec_dur`, `asynchronicity`, `tmin_mean`, `tmax_mean`, `pet_mean`, `pet_min`, `pet_max`, `pet_mean_anum`, `pet_gleam_mean`, `aet_gleam_mean`, `evap_canopy_mean`, `evap_canopy_max`, `evap_canopy_anum`, `evap_surface_mean`, `evap_surface_min`, `evap_surface_max`, `evap_surface_anum`, `aridity_p_pet`, `aridity_pet_aet`, `ai_mean`, `rel_hum_mean`, `srad_lw_mean`, `srad_sw_mean`, `wind_mean`, `sm_lvl1_mean`, `sm_lvl2_mean`, `sm_lvl3_mean`, `sm_lvl4_mean`.
- **Land Cover (14)**: `water_frac`, `trees_frac`, `flooded_veg_frac`, `crops_frac`, `built_area_frac`, `bare_frac`, `range_frac`, `dom_land_cover`, `dom_land_cover_frac`, `lai_mean`, `lai_min`, `lai_max`, `lai_diff`.
- **Soil Characteristics (27)**: `soil_depth`, `soil_conductivity_top`, `soil_conductivity_sub`, `soil_awc_top`, `soil_awc_sub`, `soil_awsc_min`, `soil_awsc_max`, `soil_awsc_major`, `sand_frac_top`, `sand_frac_sub`, `silt_frac_top`, `silt_frac_sub`, `clay_frac_top`, `clay_frac_sub`, `gravel_frac_top`, `gravel_frac_sub`, `bulkdens_top_major`, `bulkdense_top_mean`, `bulkdens_sub_mean`, `org_carb_top_major`, `org_carb_top_mean`, `org_carb_sub_major`, `org_carb_sub_mean`, `organic_frac_top`, `organic_frac_sub`, `hsg_major`, `wtd`.
- **Geology (7)**: `geol_porosity`, `geol_permeability`, `carb_rocks_frac`, `geol_class_1st`, `geol_class_1st_frac`, `geol_class_2nd`, `geol_class_2nd_frac`.
- **Anthropogenic (23)**: `num_dams`, `res_store_sum`, `first_dam_year`, `last_dam_year`, `irrigation_frac`, `hydroelec_frac`, `drinking_frac`, `flood_frac`, `overflow_frac`, `navigation_frac`, `tailing_frac`, `pop_density_2000`–`2020` (5 vars), `urban_frac_1985`–`2005` (3 vars), `crops_frac_1985`–`2005` (3 vars).

### 8.5 Recommended River-Network Inputs (23 Variables)
- **From HydroRIVERS (10)**:
  - Adjacency / Graph Edges: `NEXT_DOWN`, `MAIN_RIV`.
  - Edge Weights / Physical Distances: `LENGTH_KM`, `DIST_DN_KM`, `DIST_UP_KM`.
  - Node Area Scaling: `CATCH_SKM` (local), `UPLAND_SKM` (cumulative upstream).
  - Topological Hierarchy: `ORD_STRA` (Strahler stream order), `ORD_CLAS` (Classical stream order), `ENDORHEIC`.
- **From HydroBASINS (13)**:
  - Basin Hierarchy & Grouping: `HYBAS_ID`, `NEXT_DOWN`, `NEXT_SINK`, `MAIN_BAS`, `PFAF_ID`.
  - Regional Flow Distances & Scaling: `DIST_SINK`, `DIST_MAIN`, `SUB_AREA`, `UP_AREA`.
  - Regime & Sequence: `ENDO`, `COAST`, `ORDER`, `SORT`.

### 8.6 Variables That MUST NOT Be Model Inputs (Streamflow-Derived Leakage)
1. **Target**: `streamflow_observed` (strictly the prediction and loss target).
2. **All 73 CAMELS-IND Hydrology Signatures**: `q_mean`, `runoff_ratio`, `streamflow_elas`, `slope_fdc`, `bfi`, `q_cv`, `q_10`..`q_90`, `q_zero`, `q_low_days`, `freq_q_low`, `q_high_days`, `freq_q_high`, `annual_q`, `mean_anum_flow`, `cen_time`, `gini_flow`, `annual_max_*` (1, 3, 7, 30, 90 day), `annual_min_7day`, `month_1day_max`, `month_1day_min`, `doy_*`, monthly mean flows (12), monthly CV flows (12), seasonal flows (4), seasonal percentiles (6), rise/fall rates and days (6), `num_hyd_alt`.
3. **`reservoir_index`**: Proved to be $\text{Total Storage} / \text{Annual Streamflow}$.
4. **`flow_availability`**: Metric derived from target observation coverage.

### 8.7 Variables Requiring Initial Exclusion / Further Investigation (14 Variables)
1. `tavg(C)` — Linear combination of `tmax` and `tmin`.
2. `wind(m/s)` — Algebraic combination of `wind_u` and `wind_v`.
3. `pet(mm/day)` — Missing entire year 1980 (366 days).
4. `lstm_pred_streamflow` — Existing model prediction / baseline candidate.
5. `DIS_AV_CMS` (HydroRIVERS) — WaterGAP 2.2 model simulated discharge.
6. `ORD_FLOW` (HydroRIVERS) — Size class derived from `DIS_AV_CMS`.
7. `n_dams` (GRaND) — Collinear with India-WRIS `num_dams` ($r = 0.982$).
8. `total_storage` (GRaND) — Collinear with India-WRIS `res_store_sum`.
9. `cwc_area` — Redundant with DEM-delineated `ghi_area` ($r = 0.9996$).
10. `slope_min` — Near-zero variance (470 / 472 catchments have 0.0).
11. `low_prec_timing` — Zero variance (100% `"winter"`).
12. `evap_canopy_min` — Zero variance (100% `0.0`).
13. `high_prec_timing` — Extreme class imbalance (96.8% `"monsoon"`).
14. `bulkdens_sub_major` — 28.81% missing values (substitute `bulkdens_sub_mean`).

### 8.8 Potential Leakage Issues
- **Target Leakage**: Any inclusion of hydrological signatures or `reservoir_index` in the input vector directly exposes the target streamflow distribution to the model.
- **Spatial Auto-Correlation Leakage**: Randomly splitting catchments will place nested or adjacent catchments into both train and test splits, allowing the model to memorize upstream/downstream flow dynamics rather than generalizing to unseen river basins.
- **Baseline Leakage**: Using `lstm_pred_streamflow` as a feature would make our model dependent on another ML model rather than predicting floods from raw hydrometeorology.

### 8.9 Data Quality Issues
- Standardized ID string padding is required (`str(id).zfill(5)`) to bridge `int64` attribute tables and 5-digit zero-padded shapefiles and forcing filenames.
- 1980 data gap in Singer PET makes `pet_gleam` mandatory for 1980–2020 temporal consistency.
- Subsoil bulk density major class requires replacement with the complete mean class (`bulkdens_sub_mean`).
- Dam purpose fractions require 0.0 imputation for dam-free catchments.

---

## 9. Recommended Next Step

> [!IMPORTANT]
> **Recommended Next Step**:  
> **"Create a leakage-safe catchment-level train/validation/test split after reviewing spatial connectivity between CAMELS-IND catchments, HydroRIVERS and HydroBASINS."**
