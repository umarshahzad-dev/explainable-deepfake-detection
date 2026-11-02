# Milestone 1.1 — Audited inventory

This update **replaces** `src/deepfake_data/inventory.py` and `tests/test_inventory.py`. It does not overwrite the original datasets or update source files from the previous v1 project.

- Registers `.gif` files as `animated_image` without decoding into frames.
- Reports extension totals and excluded auxiliary extensions (`.npy`, `.zip`, etc.).
- Records missing dataset roots and unsupported media.
- Leaves unknown labels unknown; does not infer train/test membership.
- Avoids assuming DF40 method names indicate labels.

## Install

Extract the ZIP **into the project root** (`D:\University\explainable-deepfake-detection-system-v2`) and allow overwriting the two files. The ZIP has `src/`, `tests/`, and this README at top level.

```powershell
cd D:\University\explainable-deepfake-detection-system-v2
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m deepfake_data.inventory
```

## Verification of known DF40 disk counts

Inspect `artifacts/inventory/report.json` and confirm its `discovered_extensions.df40` counts against the independent file audit. The independent audit reported 620259 PNG, 6000 JPG, 2000 JPEG, 247 GIF, and 352516 NPY in the extracted root; counts may differ if the project junction points at a different directory or the data changes. Non-media files are intentionally excluded.

The directory listing alone does not establish correct train/test split or per-identity separation. Do not train from this manifest yet.

## Git

```powershell
git add src/deepfake_data/inventory.py tests/test_inventory.py README-AUDIT.md
git commit -m "fix(ingestion): audit dataset formats and include GIF files"
git push origin main
```
