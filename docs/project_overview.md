# Project Overview & Architectural Vision

**Project**: ML-Based Flood Prediction for Ungauged Rivers  
**Planned Architecture**: Regional Temporal Convolutional Network (TCN) + River-Network Graph Attention Network (GAT) + Quantile Regression  
**Status**: Architecture Planned & Under Active Development (Preprocessing & Graph Structure Established)

---

## 1. Architectural Blueprint

The complete target modeling system integrates multi-scale temporal rainfall-runoff dynamics, topological fluvial message-passing, and non-parametric uncertainty quantification:

```
                            INPUT PIPELINE
       ┌──────────────────────────────────────────────────────┐
       │ CAMELS-IND Dynamic Meteorology + Soil Moisture       │
       │ (16 Physical Forcings + 4 Cyclical Calendar Features)│
       │ [Batch x Sequence_Length x 20]                       │
       └──────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                     REGIONAL TCN ENCODER (Planned)
       ┌──────────────────────────────────────────────────────┐
       │ • Stacked Dilated Causal 1D Convolutions             │
       │ • Exponentially Growing Receptive Field (1 to 64 days)│
       │ • Conditioned on Static Catchment Physiography       │
       │   (Topography, Soils, Geology, Dams: 129 Features)   │
       └──────────────────────────┬───────────────────────────┘
                                  │
                                  ▼ Latent Temporal Catchment Embeddings h_v
                    RIVER-NETWORK GAT (Planned)
       ┌──────────────────────────────────────────────────────┐
       │ • Directed Advective Message Passing along           │
       │   HydroRIVERS Asian Network (Downhill Flow DAG)      │
       │ • Multi-Head Topological Attention                   │
       │ • Edge Routing Distances & Drainage Area Ratios      │
       │   z_v = W_0 h_v + SUM alpha_uv W_1 [h_u || e_uv]     │
       └──────────────────────────┬───────────────────────────┘
                                  │
                                  ▼ Spatiotemporal Embeddings z_v
                 QUANTILE REGRESSION HEAD (Planned)
       ┌──────────────────────────────────────────────────────┐
       │ • Fully Connected Linear Output Projection           │
       │ • Multi-Quantile Loss (Pinball Loss Function)        │
       │ • Non-Crossing Quantiles: tau in {0.10, 0.50, 0.90}  │
       └──────────────────────────┬───────────────────────────┘
                                  │
                                  ▼
                          OUTPUT PREDICTIONS
       ┌──────────────────────────────────────────────────────┐
       │ • P10: Lower Bound (Low Flow / Drought Level)        │
       │ • P50: Median Forecast (Expected Discharge in m3/s)  │
       │ • P90: Upper Extreme Bound (Flood Inundation Hazard) │
       └──────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Implementation Status Note**:  
> This architecture represents the **planned capstone design**. Steps 1, 2, and 3 have completed the data audit, leakage-safe splits, train-only scalers, and graph topology matrices. The actual deep learning models are under active development and have not yet been trained or implemented.

---

## 2. Core Architectural Components

### 2.1 Regional Temporal Convolutional Network (TCN)
- **Role**: Learns the local catchment rainfall-runoff response across multiple temporal scales (flashy surface runoff over 1–3 days vs slow baseflow release from deep soil moisture over 30–60 days).
- **Advantage over RNNs/LSTMs**: Causal convolutions prevent future temporal leakage; dilated filters provide an expansive receptive field with non-recurrent parallel computation; residual connections stabilize deep gradient propagation.
- **Regional Conditioning**: Invariant static features (soil hydraulic conductivity, mean slope, land use fractions) are injected into the temporal layers to modulate hydraulic response parameters across different geographic regions.

### 2.2 River-Network Graph Attention Network (GAT)
- **Role**: Models the hydraulic transport and routing of streamflow from upstream tributaries into downstream mainstems.
- **Topological Graph**: Built upon HydroRIVERS Asia directed acyclic reach connectivity.
- **Dynamic Attention ($\alpha_{u, v}$)**: Learns how much hydraulic weight a tributary contributes to a downstream confluence during flood conditions, modulated by physical channel routing distance and drainage area scaling.

### 2.3 Quantile Regression & Calibrated Uncertainty
- **Role**: Provides risk-aware flood prediction bounds ($P_{10}, P_{50}, P_{90}$) rather than deterministic point estimates.
- **Loss Function**: Pinball loss (tilted absolute loss) evaluated across the specified quantiles, optimizing both calibration (coverage reliability) and sharpness (interval width).
