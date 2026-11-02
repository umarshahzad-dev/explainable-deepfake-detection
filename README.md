# Explainable Deepfake Detection — dataset foundation

Labels are **0 = fake** and **1 = real**. DF40 samples come from the authors' testing-data release; training on them is an experiment, not an untouched official DF40 benchmark. Inventory and splits never move or modify raw media. Generated manifests, image outputs and logs belong outside Git.

## Setup

Python 3.11 and the existing `.venv` are expected. From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

The inventory and splitter use the standard library. Optional `.[inspection]` installs Pillow for image/GIF decoding. Optional `.[extraction]` installs OpenCV and facenet-pytorch, including PyTorch and its dependencies; these can be large. MTCNN weights are supplied with facenet-pytorch. Installing packages requires network access and may select a CPU-only PyTorch build; choose and install an appropriate CUDA-enabled PyTorch build separately for GPU use. No optional package or weights are installed by these commands unless requested.

## Inventory and validate

Configured paths in `configs/datasets.json` point directly to the central physical dataset roots, avoiding nested junctions. Change them if moving the project. Preview first:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.inventory --limit-per-dataset 100
# Path-only full scan; may take time and substantial memory:
.\.venv\Scripts\python.exe -m deepfake_data.inventory
```

The command writes `artifacts/inventory/manifest.csv` and `report.json`. A limited scan cannot establish totals or unmatched metadata counts. The full scan does not decode media. Original JSON files are read, not changed. For a small image/GIF decode sample after installing `.[inspection]`:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.inspect_media artifacts/inventory/manifest.csv --limit 20 --output artifacts/inspection.json
```

This checks the first frame, dimensions and GIF frame count; it does not establish that every file or frame is valid. GIFs remain individual inventory records; do not send GIF files directly to a still-image training loader. To convert a selected GIF without re-cropping faces:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.gif_frames "D:\path\example.gif" --output artifacts/gif_frames --max-frames 10
```

## Splits and limitations

```powershell
# Known-label sample groups can be used for experimental training only:
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --protocol experimental --seed 42 --output artifacts/splits/experimental.csv
# Strict within-dataset and separate held-out-dataset protocols:
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --output artifacts/splits/strict.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --holdout celebdf --output artifacts/splits/holdout-celebdf.csv
.\.venv\Scripts\python.exe -m deepfake_data.split artifacts/inventory/manifest.csv --holdout ffpp --output artifacts/splits/holdout-ffpp.csv
```

Reports accompany each CSV. `--ratios 0.7 0.15 0.15` and `--seed` are configurable. `--hash` computes SHA-256 from every listed media file, potentially taking a long time; existing hashes are checked for cross-partition collisions. Missing hashes are not evidence of independence.

The inventory marks video or metadata sample groups `sample_verified`: this proves frames in the same recorded sample belong together, not that other samples or datasets are unrelated. Suspected and unknown relationships are **training-only** in the experimental protocol, and quarantined in strict evaluation. Independent test partitions therefore remain empty until source relationships are reviewed and encoded as `group_id` and `group_status=verified`. For held-out rotations, training records additionally need a reviewed `verified_independent_of` value (`celebdf`, `ffpp`, or `celebdf|ffpp`). Assign one common group ID to all verified related records; do not use the same ID for unrelated samples. These fields are evidence supplied by the researcher, not an automatic identity-verification system. There is no demonstrated identity-safe or independent physical test set yet.

## Video extraction

After explicitly installing `.[extraction]`, start with one selected video. Outputs must be outside raw datasets:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.extract --manifest artifacts/splits/experimental.csv --output artifacts/faces --max-videos 1 --max-frames 5 --device cpu
```

`--device cuda` requires a compatible CUDA PyTorch installation. Frames are uniformly spaced, the largest detected face above threshold is aligned using five landmarks, and visually similar consecutive selected crops are skipped. Per-video `complete.json` records original video path and frame indices, failures, settings and output names. Completed videos are skipped on rerun; interrupted videos restart using the same configuration. Outputs are never intended to replace source media. DF40 still images are reused without MTCNN; review their size and quality before training.

After reviewing the smoke output, disk capacity, split report and processing costs, an explicitly authorized full pass can use:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.extract --manifest artifacts/splits/experimental.csv --output artifacts/faces --all-videos --max-frames 20 --device cuda
```

This command processes videos assigned to `train`. Combine completed crops with existing DF40 still images into an input manifest:

```powershell
.\.venv\Scripts\python.exe -m deepfake_data.training_manifest artifacts/splits/experimental.csv --faces artifacts/faces --output artifacts/training/images.csv
```

The output retains labels, partitions, source paths and frame indices. The accompanying report lists missing video extractions and GIFs requiring conversion. It is an input manifest; image quality and training transforms still require validation. No Xception training implementation or full-dataset integrity certification is included.
