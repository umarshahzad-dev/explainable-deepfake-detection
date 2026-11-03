# Explainable Deepfake Detection â€” dataset foundation

Labels are **0 = fake** and **1 = real**. DF40 samples come from the authors' testing-data release; training on them is an experiment, not an untouched official DF40 benchmark. Inventory and splits never move or modify raw media. Generated manifests, image outputs and logs belong outside Git.

## Inventory and splits

Python 3.11 is required. The inventory and splitter use the standard library. Paths are configured in `configs/datasets.json` and point to the central physical dataset roots. From the project root:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.inventory --limit-per-dataset 100
# Path-only full scan; may take time and substantial memory:
.\.venv\Scripts\python.exe -m deepfake_data.inventory
```

The command writes `artifacts/inventory/manifest.csv` and `report.json`. A limited scan cannot establish totals or unmatched metadata counts; a full scan does not decode media. Original JSON files are read, not changed. Image and GIF inspection can be run with `python -m deepfake_data.inspect_media MANIFEST --limit 20 --output artifacts/inspection.json`; a selected GIF can be converted with `python -m deepfake_data.gif_frames GIF --output artifacts/gif_frames --max-frames 10`. These commands require Pillow.

Generate relationship candidates for both holdouts, then review the CSV evidence. An unknown or suspected relationship is not proof of independence; set `reviewed_independent` only where documented evidence supports it. The review requires an evidence field.

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.relationships artifacts/inventory/manifest.csv --holdout celebdf --output artifacts/reviews/celebdf.csv
.\.venv\Scripts\python.exe -m deepfake_data.relationships artifacts/inventory/manifest.csv --holdout ffpp --output artifacts/reviews/ffpp.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol experimental --seed 42 --output artifacts/splits/within.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol experimental --holdout celebdf --reviews artifacts/reviews/celebdf.csv --output artifacts/splits/holdout-celebdf.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol experimental --holdout ffpp --reviews artifacts/reviews/ffpp.csv --output artifacts/splits/holdout-ffpp.csv
```

Each split report includes class coverage and group/available-hash overlap checks. `--hash` hashes each file and can take a long time. Both classes in train and test are necessary, not sufficient, for a meaningful evaluation. The experimental within-dataset split is training-only because unreviewed training content could overlap evaluation. FF++ YouTube source groups follow the documented target/source filenames; DeepFakeDetection, Celeb-DF identities and cross-dataset relationships require further evidence. Identity-level safety is not established.

## Bounded face extraction and visual inspection

Use the existing extraction environment; no new PyTorch download or package installation is needed. Set the project source path for the current PowerShell session:

```powershell
$py = "D:\University\explainable-deepfake-detection-system\venv\Scripts\python.exe"
$env:PYTHONPATH = "$PWD\src"
& $py -c "import torch, cv2; from facenet_pytorch import MTCNN; print(torch.__version__, torch.cuda.is_available(), cv2.__version__)"
# Run pytest from the project venv without installing packages into the extraction environment:
& $py -c "import sys; sys.path.insert(0,'src'); sys.path.append(r'D:\University\explainable-deepfake-detection-system-v2\.venv\Lib\site-packages'); import pytest; raise SystemExit(pytest.main(['-q']))"
```

**Run this inspection first.** It uses four explicitly selected files, so a full inventory is not required. The marked difficult video had a blurred candidate in a previous smoke test; add other known difficult videos with `--difficult-video` or provide a manifest with `--manifest` to select across dataset/label buckets. Inspect `contact_sheet.jpg`, individual originals/crops and `quality_report.json` before a larger run.

```powershell
& $py -m deepfake_data.quality `
  --difficult-video "D:\University\datasets\deepfake\celebdf\Celeb-DF-v2\Celeb-synthesis\id0_id16_0000.mp4" `
  --video "D:\University\datasets\deepfake\celebdf\Celeb-DF-v2\Celeb-real\id0_0000.mp4" `
  --video "D:\University\datasets\deepfake\ffpp\manipulated_sequences\Deepfakes\c23\videos\000_003.mp4" `
  --video "D:\University\datasets\deepfake\ffpp\original_sequences\youtube\c23\videos\000.mp4" `
  --output artifacts/quality-review --max-videos 4 --max-frames 2 --candidates-per-segment 2 --device cuda
```

The defaults use a square MTCNN-box crop, 1.5Ã— margin, 299Ã—299 output, confidence â‰¥0.95, source face â‰¥80Ã—80, sharpness â‰¥30, padding â‰¤35%, up to 20 temporal segments and three candidates per segment. Each segment selects at most one eligible frame using confidence, size, sharpness and padding. These thresholds and the ranking are engineering defaults, not validated optima. The deterministic face strategy selects the largest detection by default; `--face-strategy highest_confidence` and optional `--crop-mode aligned` are available. No identity tracking is performed, so the selected person can change in multi-face videos.

After visual review and explicit approval, the following **bounded batch** processes at most 200 training videos and records a console log. The current `artifacts/splits/within.csv` may be only a preview: replace it with the reviewed, full split manifest intended for the experiment. Check free disk space before running and use a separate output directory from inspection.

```powershell
Get-PSDrive D | Select-Object Free
New-Item -ItemType Directory -Force artifacts/logs | Out-Null
& $py -m deepfake_data.extract --manifest artifacts/splits/within.csv --output artifacts/faces-xception `
  --partition train --max-videos 200 --max-frames 20 --candidates-per-segment 3 --device cuda 2>&1 |
  Tee-Object -FilePath artifacts/logs/extraction.log -Append
```

Repeat the exact command to resume. Per-video checkpoints record selected frames, rejected candidates, read errors, zero yield, source provenance, parameters and detector runtime. Changed settings or incompatible older crops are rejected; use a new output directory. Crops are written via temporary files and recorded only after completion. `--partition val` or `--partition test` selects those videos when a verified split exists. The run is sequential and bounded but its duration and storage use depend on the media and GPU.

After extraction, combine crops with eligible still images and previously converted GIF frames:

```powershell
& $py -m deepfake_data.training_manifest artifacts/splits/within.csv --faces artifacts/faces-xception --gif-frames artifacts/gif_frames --output artifacts/training/images.csv
```

The manifest builder reports legacy or incompatible video crop configurations instead of silently mixing them. Review that report, source relationships, split coverage, class balance and the crops before using an experiment. These outputs do not establish independent evaluation or training readiness.

## Full reconciliation and readiness gate

A full scan is path-only, takes time and may use substantial disk-backed temporary storage. It does not decode all media. Preserve the generated report with its matching manifest. Finalization records original-partition conflicts, dimension counts and the manifest SHA-256. The detailed reconciliation command writes all local metadata issues and unmatched JSON records as CSVs.

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.inventory
.\.venv\Scripts\python.exe -m deepfake_data.finalize_inventory --manifest artifacts/inventory/manifest.csv --report artifacts/inventory/report.json
.\.venv\Scripts\python.exe -m deepfake_data.reconcile_details --manifest artifacts/inventory/manifest.csv --output artifacts/reconciliation
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol strict --seed 42 --output artifacts/splits/within-full.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol strict --holdout celebdf --output artifacts/splits/holdout-celebdf-full.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol strict --holdout ffpp --output artifacts/splits/holdout-ffpp-full.csv
.\.venv\Scripts\python.exe -m deepfake_data.relationships artifacts/inventory/manifest.csv --holdout celebdf --output artifacts/reviews/celebdf-full.csv
.\.venv\Scripts\python.exe -m deepfake_data.relationships artifacts/inventory/manifest.csv --holdout ffpp --output artifacts/reviews/ffpp-full.csv
```

Review relationship queues before passing any review file to `--reviews`; evidence is required. Regenerate a protocol after review. The full scan currently records DF40 original-partition conflicts, not a verified DF40 test partition.

Run the gate using the existing extraction environment (set `$py` and `$env:PYTHONPATH` as above). It emits JSON and Markdown, verifies the inventory/split hash chain and distinguishes bounded extraction from training/evaluation readiness:

```powershell
& $py -m deepfake_data.readiness --manifest artifacts/splits/within-full.csv `
  --split-report artifacts/splits/within-full.report.json --inventory-report artifacts/inventory/report.json `
  --quality-dir artifacts/quality-review-verified --df40-samples artifacts/readiness/df40-samples.json `
  --output-dir artifacts/faces-xception --batch-videos 200 --report artifacts/readiness/within-full.json
```

A PASS for extraction is limited to the declared batch and evidence; it does not authorize a run or establish independent evaluation. A 200-video FF++ training batch, only after explicit approval, uses `artifacts/splits/within-full.csv` in the extraction command above. Repeat that exact command to resume. The minimum free-space threshold and uncompressed-size allowance are estimates; monitor disk usage and stop if needed.

## Combined three-dataset experiment

The current allocation is a blocked review draft, not an approved split. See [COMBINED-PIPELINE.md](COMBINED-PIPELINE.md) for source-link evidence, proposed quotas, bounded checks and the conditional future processing workflow.
