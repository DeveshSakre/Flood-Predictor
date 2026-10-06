# Reports Directory

This directory contains technical audit reports, milestone deliverables, and architecture documentation.

---

## Completed Technical Reports

1. [data_audit_report.md](data_audit_report.md)
   - Step 1: Comprehensive audit of CAMELS-IND, HydroRIVERS, and HydroBASINS datasets.
   - Characterization of 472 catchments, temporal coverage (1980–2020), missingness analysis, and feature categorization.

2. [catchment_split_audit.md](catchment_split_audit.md)
   - Step 2: Leakage-safe catchment partitioning.
   - Definition of 169 train, 36 validation, 37 held-out test, and 5 operational ungauged demonstration basins.
   - Upstream-downstream connected pair isolation (zero cross-partition edges).

3. [feature_preprocessing_report.md](feature_preprocessing_report.md)
   - Step 3: Feature selection and normalization.
   - Catalog of 20 dynamic, 129 static, and 10 graph features (159 total).
   - Training-only scaler parameters, log transformations, and imputation strategies.

4. [graph_construction_report.md](graph_construction_report.md)
   - Step 3: Directed acyclic river graph construction.
   - Verification of 13 hydrologic routing edges in training topology and zero cross-partition edges.

---

## Upcoming Reports

- `github_repository_setup.md` — Current milestone report documenting the collaborative repository structure.
- `xgboost_baseline_report.md` — Step 4 benchmark report.
- `regional_tcn_report.md` — Step 5 & 6 temporal modeling report.
- `gat_routing_report.md` — Step 7 & 8 spatiotemporal graph report.
- `uncertainty_calibration_report.md` — Step 9 quantile regression evaluation.
