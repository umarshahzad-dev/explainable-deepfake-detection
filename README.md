# Explainable Deepfake Detection

Research-oriented system under development for deepfake detection and explainability.

## Milestone 1: Dataset inventory

The inventory indexes media paths without copying or decoding files. It records only labels supported by recognized directory conventions. Unknown labels remain blank. No train/validation/test split is generated yet.

### Windows setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m deepfake_data.inventory --limit-per-dataset 100
```

After checking the preview report, remove `--limit-per-dataset 100` for a full inventory.

The configured roots are under `data/raw/` and should reference existing centrally stored datasets. Inventory output is written to `artifacts/inventory/` and must not be committed.

**Caution:** DF40 provenance and official split membership are unverified. The inventory is not a license to use these samples for training. Identity-level leakage checks and media integrity checks are future milestones.
