# Team Collaboration & Git Workflow Guide

This document defines the Git collaboration model, code standards, and role division for our 5-person capstone engineering team.

---

## 1. Team Roles & Responsibilities

| Role / Member | Feature Branch | Core Responsibilities |
| :--- | :--- | :--- |
| **Member 1 (Data & Preprocessing)** | `member1-data` | Data loaders (`src/data/loaders.py`), batching generators, rolling antecedent precipitation feature engineering, scaler integration, missing value handling. |
| **Member 2 (Temporal Modeling / TCN)** | `member2-tcn` | Standard LSTM baseline, Dilated Temporal Convolutional Network (`src/models/tcn.py`), Regional TCN module with static attribute conditioning (`src/models/regional_tcn.py`). |
| **Member 3 (Spatial Modeling / GAT)** | `member3-gat` | River-Network Graph Attention Network (`src/models/gat.py`), topological message-passing layers, attention weight routing, coupled Regional TCN + GAT architecture (`src/models/regional_tcn_gat.py`). |
| **Member 4 (Uncertainty Quantification)** | `member4-uncertainty` | Quantile loss functions (`src/uncertainty/quantile_loss.py`), Pinball loss optimization, quantile regression head ($P_{10}, P_{50}, P_{90}$), prediction interval calibration and sharpness metrics. |
| **Member 5 (Baselines & Evaluation)** | `member5-evaluation` | Ridge & XGBoost baselines (`src/models/xgboost_baseline.py`), hydrological evaluation pipeline (`src/evaluation/`), NSE/KGE/Peak Timing metrics, visualization plots, end-to-end integration. |

---

## 2. Branching Architecture

```
main (Production-ready, verified code only)
 │
 ├── member1-data            (Data pipeline & feature engineering)
 ├── member2-tcn             (TCN architecture & temporal modeling)
 ├── member3-gat             (River-network GAT & spatial routing)
 ├── member4-uncertainty     (Quantile regression & calibration)
 └── member5-evaluation      (XGBoost baseline & evaluation metrics)
```

---

## 3. Step-by-Step GitHub Workflow

Every team member must adhere to this 11-step lifecycle for all code changes:

```
1. Clone Repository ────> 2. Pull Latest main ────> 3. Checkout Feature Branch
                                                            │
                                                            ▼
6. Commit Cleanly   <──── 5. Run Unit Tests  <──── 4. Implement Modular Code
        │
        ▼
7. Push to Remote   ────> 8. Open Pull Request ───> 9. Code Review & Approval
                                                            │
                                                            ▼
11. Pull Latest main <─── 10. Squash/Merge to main
```

### Detailed Commands:

```bash
# 1. Clone repository (first time only)
git clone https://github.com/<your-org>/ml-flood-prediction-ungauged-rivers.git
cd ml-flood-prediction-ungauged-rivers

# 2. Update local main
git checkout main
git pull origin main

# 3. Create or switch to your assigned feature branch
git checkout -b member2-tcn   # First time
git checkout member2-tcn      # Subsequent work

# 4. Implement your changes in src/ or experiments/

# 5. Run tests locally to ensure nothing is broken
pytest tests/

# 6. Stage and commit with descriptive messages
git add src/models/tcn.py configs/tcn_config.yaml
git commit -m "feat(tcn): implement dilated causal convolution block with residual connections"

# 7. Push feature branch to GitHub
git push -u origin member2-tcn

# 8. Create Pull Request (PR) on GitHub against 'main'
#    - Fill in PR description detailing changes
#    - Reference the issue or project milestone

# 9. Code Review
#    - At least one other team member must review and approve the PR

# 10. Merge to main on GitHub

# 11. Synchronize local main before starting next task
git checkout main
git pull origin main
```

---

## 4. Code Standards & Anti-Leakage Rules

1. **Never Commit Raw Data**: Raw CSVs, NetCDF, GDB, and shapefiles must remain in local `data/raw/` directories and excluded by `.gitignore`.
2. **Never Commit Machine-Specific Absolute Paths**: Always use relative paths or configuration values from `configs/data_config.yaml`.
3. **No Target Leakage**: Never pass target streamflow into input tensors or use test-set statistics to fit scalers.
4. **Clean Commits**: Do not commit Jupyter autosaves (`.ipynb_checkpoints`), temporary logs, or scratch files.
