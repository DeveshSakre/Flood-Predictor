# Models Directory

This directory stores lightweight serialized model checkpoints, weights, and production artifacts.

---

## Storage & Tracking Policy

- **Untracked by Git (`.gitignore`)**:
  - `*.pt`, `*.pth`, `*.ckpt` (PyTorch model checkpoints)
  - `*.joblib`, `*.pkl` (Scikit-Learn/XGBoost fitted binaries > 10MB)
  - Model weights exceeding 10MB must be uploaded to institutional cloud storage (e.g. Zenodo, Google Drive, or Hugging Face) and referenced via URLs.

- **Tracked by Git**:
  - Small architecture configurations (`.yaml` / `.json`)
  - Metadata schemas documenting input/output dimensions and feature order
  - Lightweight scalers already versioned in `data/processed/scalers/`

---

## Checkpoint Naming Convention

When saving checkpoints locally during development:
- `xgboost_baseline_seed42.joblib`
- `tcn_seed42_epoch{epoch:03d}.pt`
- `regional_tcn_seed42_best.pt`
- `regional_tcn_gat_seed42_best.pt`
