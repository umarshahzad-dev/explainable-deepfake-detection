# DF40-focused v2 allocation

The Phase 3A allocator reads the corrected Phase 1 inventory, the prior source
group enrichment, connected source components, the completed SHA audit, and the
reserved DF40 baseline evaluation selection. It does not copy, decode, or alter
source media. A release is frozen only after SHA-256, verified-label, split,
source-component, exact-byte, Celeb-DF boundary, and 26-method coverage checks
pass.

Run from the project root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m deepfake_data.finalize_v2
```

SHA progress is printed every 5,000 eligible records. The local SQLite cache at
`artifacts/allocation/df40-focused-v2-final/allocation-hash-cache.sqlite3`
supports restart. A cached digest is reused only when the current file size and
nanosecond modification time still match. If interrupted, rerun the same command
with the same inputs and seed. The script refuses to overwrite an existing
frozen CSV; choose a new output directory for a changed release.

The split seed is 2026 with 70/10/10/10 target ratios. Verified source/sample
groups and exact-byte duplicates are allocated together. The source graph labels
cross-dataset filename/domain links as `contains_suspected_links`; those links
remain in the release as relationship evidence, while allocation uses the
underlying source/sample group. The report counts suspected relationship
components spanning partitions, and they remain a scientific leakage limitation
rather than verified identity claims. The official Celeb-DF test boundary and
selected DF40 baseline development groups are pinned to test, so test
proportions can differ from the target ratio. The existing DF40 baseline set has
already informed development and is not an untouched benchmark. SadTalker and
Wav2Lip remain recorded as excluded.

All eligible media are retained rather than imposing a global sample cap.
Unknown labels, missing media, failed hashes, empty files, conflicting-label
duplicates, and duplicate inventory paths are quarantined with reasons. Exact
hashes detect byte-identical files only and do not establish universal
person-identity independence.

The release directory also contains `allocation-counts.csv`,
`source-group-counts.csv`, and `quarantine-audit.csv` for focused review.

The report proposes balanced batch exposure (50/50 fake/real), with DF40/FF++/
Celeb-DF dataset exposure 50/25/25 per class. Within a selected fake dataset,
sample methods uniformly, then source groups uniformly, and draw one image per
group per batch while rotating images across epochs. Sample real source groups
uniformly within dataset. Stored files remain imbalanced; this strategy is a
future sampler policy and is not implemented as part of allocation. The reported PNG
storage estimate uses the previous observed mean size and approximately 9.26
accepted crops per historical video; it is planning evidence, not a guaranteed
yield. Existing v1 PNG reuse remains zero until a crop importer verifies each
image's integrity, source, partition, and preprocessing signature.
