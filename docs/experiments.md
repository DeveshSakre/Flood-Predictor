# Experimental Progression & Benchmark Matrix

This document outlines the planned experimental phases, benchmark architectures, and ablation studies for the project.

---

## 1. Experimental Progression Overview

```
Phase 1: Baselines (Step 4)      Phase 2: Regional TCN        Phase 3: River GAT + Quantile
┌───────────────────────────┐    ┌───────────────────────┐    ┌───────────────────────────┐
│ Exp 1.1: Ridge Regression │    │ Exp 2.1: Base TCN     │    │ Exp 3.1: TCN + GAT Point  │
│ Exp 1.2: Global XGBoost   │ -> │ Exp 2.2: Regional TCN │ -> │ Exp 3.2: TCN + GAT        │
│ Exp 1.3: Standard LSTM    │    │          (Static Cond)│    │          Quantile Reg     │
│ Exp 1.4: Published LSTM   │    │                       │    │ Exp 3.3: Ungauged Demo    │
│          (Mangukiya 2023) │    │                       │    │          (5 Clean Basins) │
└───────────────────────────┘    └───────────────────────┘    └───────────────────────────┘
```

---

## 2. Phase 1: Baseline Models (Step 4 Focus)

Before introducing graph neural networks or complex quantile mechanisms, we establish rigorous benchmark floors:

1. **Exp 1.1: Ridge Regression**:
   - Linear baseline mapping normalized daily rainfall, antecedent 7-day cumulative rainfall, and basic static catchment attributes to daily discharge.
2. **Exp 1.2: Global Gradient Boosted Trees (XGBoost)**:
   - Tabular model trained across all 169 training catchments using engineered rolling lag features (1, 3, 7, 14, 30 days) and full static attributes.
3. **Exp 1.3: Standard LSTM (Catchment-Independent)**:
   - Standard 2-layer LSTM trained on meteorological time series without graph topology.
4. **Exp 1.4: Published LSTM Baseline Benchmark**:
   - Evaluate against the pre-existing regional LSTM predictions from Mangukiya et al. (2023) provided in `lstm_pred_streamflow.csv`.

---

## 3. Phase 2: Regional Temporal Convolutional Network (TCN)

- **Exp 2.1: Pure Temporal TCN**:
  - Dilated causal 1D convolution stack with kernel size $k=3$, residual blocks, and dilation factors $d \in \{1, 2, 4, 8, 16, 32\}$.
- **Exp 2.2: Regional Static-Conditioned TCN**:
  - Condition temporal hidden states with the 129 static catchment physiographic attributes via dense feature modulation.

---

## 4. Phase 3: River-Network GAT & Quantile Uncertainty

- **Exp 3.1: Regional TCN + River-Network GAT (Point Prediction)**:
  - Connect catchment-level TCN latent representations through the HydroRIVERS directed reach graph.
  - Evaluate NSE and KGE gains from topological message passing compared to standalone TCN.
- **Exp 3.2: Full Framework (Regional TCN + River GAT + Quantile Regression)**:
  - Multi-quantile pinball loss optimization predicting $P_{10}, P_{50}, P_{90}$.
  - Evaluate PICP (target 80% coverage), MPIW, and peak flood timing.
- **Exp 3.3: Operational Ungauged Deployment Demonstration**:
  - Forward execution on the 5 verified zero-flow catchments (`05020`, `15029`, `17004`, `17010`, `17020`) to generate continuous daily hydrographs and spatial flood risk maps.

---

## 5. Summary Benchmark Scorecard (Target Table)

| Model Architecture | Spatial Message Passing | Uncertainty Bounds | Train Catchments | Test Catchments | Test Median NSE | Test Median KGE | Test PICP (80% target) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ridge Baseline | None | No | 169 | 37 | *Pending Exp 1.1* | *Pending* | N/A |
| Global XGBoost | None | No | 169 | 37 | *Pending Exp 1.2* | *Pending* | N/A |
| Standard LSTM | None | No | 169 | 37 | *Pending Exp 1.3* | *Pending* | N/A |
| Published LSTM (2023)| None | No | Benchmark | 37 | *Pre-computed* | *Pre-computed* | N/A |
| Regional TCN | None | No | 169 | 37 | *Pending Exp 2.2* | *Pending* | N/A |
| **Regional TCN + GAT** | **HydroRIVERS** | **$P_{10}/50/90$** | **169** | **37** | *Planned Step 7* | *Planned* | *Planned* |
