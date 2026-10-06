# Experiments Directory

This directory tracks experiment configurations, logs, and execution notes for each experimental phase.

---

## Directory Structure

```text
experiments/
├── README.md
├── baseline/            # Step 4: XGBoost tabular baseline runs
├── regional_tcn/        # Steps 5 & 6: Temporal TCN and Regional TCN runs
├── regional_tcn_gat/    # Steps 7 & 8: Graph Attention Network coupled runs
└── quantile/            # Step 9: Quantile regression uncertainty runs
```

---

## Experiment Phases

| Phase | Model | Primary Focus | Output Directory |
| :--- | :--- | :--- | :--- |
| **Phase 1** | XGBoost Baseline | Tabular benchmark on 169 source catchments | `experiments/baseline/` |
| **Phase 2** | LSTM & TCN Baselines | Sequence modeling of hydrologic memory (180 days) | `experiments/regional_tcn/` |
| **Phase 3** | Regional TCN | Conditioning temporal representations with static attributes | `experiments/regional_tcn/` |
| **Phase 4** | River-Network GAT | Upstream-downstream flood routing on river DAG | `experiments/regional_tcn_gat/` |
| **Phase 5** | Quantile Uncertainty | Simultaneous P10, P50, P90 probabilistic bounds | `experiments/quantile/` |

---

## Tracking Guidelines

When running an experiment:
1. Save the specific configuration YAML snapshot in the respective experiment directory.
2. Log random seeds, git commit hash, and validation metrics.
3. Keep raw checkpoints in local storage (ignored by `.gitignore`).
4. Commit only summary markdown logs, performance CSVs, and visualization plots.
