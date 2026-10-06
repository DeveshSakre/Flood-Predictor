# Results Directory

This directory stores evaluation metrics, comparison tables, and figures generated across experimental phases.

---

## Directory Structure

```text
results/
├── README.md
├── baseline/            # Step 4: XGBoost baseline metrics (CSV, plots)
├── regional_tcn/        # Steps 5 & 6: TCN and Regional TCN performance
├── regional_tcn_gat/    # Steps 7 & 8: Coupled spatiotemporal performance
└── quantile/            # Step 9: Quantile calibration, PICP, and MPIW
```

---

## Metric Standards

All experimental evaluations on the 37 held-out test catchments must record:
1. **Per-Catchment Metrics CSV**:
   - `gauge_id`, `NSE`, `KGE`, `RMSE`, `MAE`, `PBIAS`, `peak_rel_error_pct`
2. **Aggregated Summary JSON**:
   - `NSE_median`, `NSE_q25`, `NSE_q75`, `KGE_median`, `fraction_nse_gt_0`
3. **Hydrographs & CDF Plots**:
   - Comparison hydrographs for selected high-flow events
   - Cumulative empirical CDF curves of NSE across unseen catchments

*Note: Large raw simulation arrays should remain local or compressed; only final aggregated tables and publication-quality figures should be committed.*
