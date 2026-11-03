# Combined Xception dataset: blocked review workflow

**Status: NOT A FINAL SPLIT. Do not run bulk processing or training.** `artifacts/combined/proposed-allocation-v2.csv` is a review artifact: every actual `project_partition` is `quarantine`, and `proposal_status=DRAFT_NOT_APPROVED`. Candidate validation/calibration/test assignments are not an established independent evaluation. The linked [review](artifacts/combined/review-v2.md) and JSON reports describe what must be resolved before a release manifest can exist.

## Evidence and policy

- Labels are fixed: fake 0, real 1. Unknown labels stay quarantined. Original JSON partition and project assignment are separate. The 100 selected DF40 class blocks checked have identical test and validation records. The authors' [rearrangement code](https://github.com/YZY-stack/DF40/blob/main/DeepfakeBench_DF40/preprocessing/rearrange.py) includes explicit assignments of records to both fields; this observation does not establish intent. DF40 testing-release material is repurposed for experimental use, never as an untouched official benchmark.
- The [FaceForensics dataset documentation](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md) describes DeepFakeDetection target/source actor and scene names. The local source graph has three components. Source actor same-scene originals are missing for 269 manipulated videos. Links within this dataset do not establish separation from other datasets.
- The official Celeb-DF test list is proposed for test, never for training; all actual assignments remain quarantined. The draft also quarantines a non-test video when either celebrity ID and sequence is a candidate match to an official-test source. This is conservative filename evidence, not a claim that filename similarity proves source identity. Other cross-dataset relationships remain unreviewed.
- For FF++ YouTube, the draft uses verified connected source pairs and deterministic seed 42 with proposed 70/10/10/10 train/validation/calibration/test allocation. It links recognizable DF40 FF-domain source IDs as suspected evidence and does not propose those images for training when their component touches a held-out candidate. Other relationships can still be missing. The source-component graph separately records suspected source edges and candidate identity memberships; neither is automatically treated as confirmed identity.
- Proposed DF40 training is capped at 10 media per original method/sample group and 5,000 per method/label. The cap is a review default, not an optimized model choice. `sampling_weights` can later generate train-only equal dataset, class, method and group weights; a future training implementation must actually apply them. These choices do not remove dataset or image-quality shortcuts.

## Reproduce the bounded audit and draft

Use the existing project environment, from the project root. Inventory is path-only but traverses all configured roots; it does not decode all files. Existing full results are under `artifacts/inventory/`.

```powershell
$py = ".\.venv\Scripts\python.exe"
& $py -m pytest -q
& $py -m deepfake_data.inventory
& $py -m deepfake_data.finalize_inventory --manifest artifacts/inventory/manifest.csv --report artifacts/inventory/report.json
& $py -m deepfake_data.reconcile_details --manifest artifacts/inventory/manifest.csv --output artifacts/reconciliation
& $py -m deepfake_data.source_links --manifest artifacts/inventory/manifest.csv --output artifacts/combined/evidence-inventory.csv --report artifacts/combined/source-link-report.json
& $py -m deepfake_data.relationship_audit --report artifacts/combined/relationship-audit.json
& $py -m deepfake_data.source_components --manifest artifacts/combined/evidence-inventory.csv --output artifacts/combined/relationships
& $py -m deepfake_data.combined_draft --manifest artifacts/combined/evidence-inventory.csv --components artifacts/combined/relationships/source-components.csv --components-report artifacts/combined/relationships/source-components-report.json --output artifacts/combined/proposed-allocation-v2.csv --report artifacts/combined/proposed-report-v2.json --seed 42
& $py -m deepfake_data.audit_combined_draft --manifest artifacts/combined/proposed-allocation-v2.csv --report artifacts/combined/proposed-audit-v2.json
& $py -m deepfake_data.combined_report --directory artifacts/combined --proposal-name proposed-report-v2.json --audit-name proposed-audit-v2.json --output artifacts/combined/review-v2.md --quality-dir artifacts/combined/storage-quality --stills-dir artifacts/combined/storage-stills
```

The review includes proposed counts by dataset, method, label, partition and source group, quarantine reasons, and disk estimates. The source-group audit distinguishes verified local actor links from filename-based suspected links and unknown cross-dataset relationships. Original datasets and JSON files are never written.

For a **bounded** source-byte duplicate preview and repeatable cache:

```powershell
& $py -m deepfake_data.hash_sources --manifest artifacts/combined/proposed-allocation-v2.csv --use-proposals --stratified --max-files 100 --cache artifacts/combined/source-hashes.sqlite --output artifacts/combined/hash-preview.csv --report artifacts/combined/hash-preview.json
& $py -m deepfake_data.audit_combined_draft --manifest artifacts/combined/proposed-allocation-v2.csv --report artifacts/combined/proposed-audit-v2.json --hash-report artifacts/combined/hash-preview.json --hash-csv artifacts/combined/hash-preview.csv
```

Do not infer independence from this preview. A full read-only byte-hash pass uses `--all` instead of `--max-files 100`; it may take hours. Repeat the same command to reuse the cache. Exact file bytes cannot identify a source-video frame represented by a recompressed or cropped image. Cache reuse is based on file size and modification time.

The following is the historical v2 source-byte command. That 39,467-file pass has since completed; the current policy and its later hash checks are described at the top of this document:

```powershell
& $py -m deepfake_data.hash_sources --manifest artifacts/combined/proposed-allocation-v2.csv --cache artifacts/combined/source-hashes.sqlite --output artifacts/combined/source-hashes-v2.csv --report artifacts/combined/source-hashes-v2.json --all --use-proposals
```

For bounded video-to-image investigation (not automatic source verification):

```powershell
& $py -m deepfake_data.cross_media_probe --pairs artifacts/combined/cross-media-pairs.csv --output artifacts/combined/cross-media-probe --max-pairs 2 --samples-per-video 2 --device cuda
```

The report and contact sheet are manual review aids; distances do not prove or disprove shared source identity.

A small previously prepared still-image smoke manifest can be checked without touching source data:

```powershell
& $py -m deepfake_data.process_stills --manifest artifacts/combined/still-smoke-input.csv --output artifacts/combined/still-smoke --max-items 2
& $py -m deepfake_data.training_manifest artifacts/combined/still-smoke-input.csv --faces artifacts/faces-xception --stills artifacts/combined/still-smoke --output artifacts/combined/still-smoke-manifest.csv
& $py -m deepfake_data.validate_prepared --manifest artifacts/combined/still-smoke-manifest.csv --decode --hash-outputs --report artifacts/combined/still-smoke-validation.json
```

The last command intentionally exits with a blocker: two smoke images cannot establish a multi-dataset split or independence.

## Historical strict-review procedure

There is currently **no `artifacts/combined/releases/final-v2.csv`**. Do not edit the draft's quarantine fields to bypass that absence. The new `deepfake_data.release` command is fail-closed: it requires a matching complete source-hash CSV and report, hashed evidence for every reviewed source component, a manifest-bound cross-media review covering all active cross-partition source pairs, class and dataset coverage, and modern DF40 real/fake evaluation coverage. It checks component and exact-byte leakage and refuses to overwrite a release. Human review remains an attestation, not software proof of person-identity independence.

The current template in `artifacts/combined/release-review-v2/` has 4,871 **unresolved** components. The 60-file stratified hash audit is incomplete, MidJourney and the DFD evaluation candidates remain quarantined, and the only cross-media probe compares two candidate pairs. The [release preflight](artifacts/combined/release-preflight-v2.json) therefore fails. Do not change template decisions without examining and recording the actual evidence. To generate a fresh template, choose a new output directory so existing review decisions are preserved:

```powershell
& $py -m deepfake_data.release_review_template --draft artifacts/combined/proposed-allocation-v2.csv --components artifacts/combined/relationships/source-components.csv --component-report artifacts/combined/relationships/source-components-report.json --output artifacts/combined/release-review-new
```

After the relevant reviews, complete source hashes, and the missing DF40 evaluation coverage are resolved, this is the fail-closed release command. It **will refuse** the current incomplete evidence and will not create the release:

```powershell
& $py -m deepfake_data.release --draft artifacts/combined/proposed-allocation-v2.csv --proposal-report artifacts/combined/proposed-report-v2.json --components artifacts/combined/relationships/source-components.csv --component-report artifacts/combined/relationships/source-components-report.json --reviews artifacts/combined/release-review-v2/component-reviews.csv --hash-csv artifacts/combined/source-hashes-v2.csv --hash-report artifacts/combined/source-hashes-v2.json --cross-media-review artifacts/combined/release-review-v2/cross-media-review.json --output artifacts/combined/releases/final-v2.csv --report artifacts/combined/release-final-v2.json
```

The current `validate_prepared` command deliberately cannot declare independent readiness while cross-dataset source evidence is missing.

Once a separately reviewed and frozen `final-split.csv` and its provenance report exist, disk capacity is checked, and the user explicitly approves bulk processing, the intended processing commands are:

```powershell
$py = ".\.venv\Scripts\python.exe"
$final = "artifacts/combined/releases/final-v2.csv" # Does not exist yet.
Get-PSDrive D | Select-Object Free
nvidia-smi
New-Item -ItemType Directory -Force artifacts/combined/logs | Out-Null
foreach ($part in @('train','val','calibration','test')) {
  & $py -m deepfake_data.extract --manifest $final --output artifacts/combined/video-faces --partition $part --all-videos --allow-empty --max-frames 20 --candidates-per-segment 3 --device cuda 2>&1 | Tee-Object -FilePath "artifacts/combined/logs/video-$part.log" -Append
  if ($LASTEXITCODE -ne 0) { throw "Video extraction failed for $part" }
  & $py -m deepfake_data.process_stills --manifest $final --output artifacts/combined/stills --partition $part --all-items --allow-empty --size 299 --max-gif-frames 10 2>&1 | Tee-Object -FilePath "artifacts/combined/logs/stills-$part.log" -Append
  if ($LASTEXITCODE -ne 0) { throw "Still processing failed for $part" }
}
& $py -m deepfake_data.training_manifest $final --faces artifacts/combined/video-faces --stills artifacts/combined/stills --output artifacts/combined/images.csv
& $py -m deepfake_data.validate_prepared --manifest artifacts/combined/images.csv --decode --hash-outputs --report artifacts/combined/prepared-validation.json
```

Repeat the same per-partition extraction and still-processing commands to resume. Check `errors-<partition>.json`, `processing-report-<partition>.json`, the full skipped-record JSONL, and zero-yield reports; mismatched parameters or provenance are rejected. Images and GIFs are decoded according to content, converted to RGB, letterboxed without a second face crop, and resized to 299×299. Letterboxing and cross-dataset differences may themselves be visual shortcuts and require review. Video frames retain their original source path, group and frame index.

After a valid image manifest and approved independent evaluation exist, `python -m deepfake_data.sampling_weights --manifest artifacts/combined/images.csv --output artifacts/combined/train-weights.csv --report artifacts/combined/train-weights.json` generates **train-only** weights. It does not train a model or change evaluation rows.

## Historical focused allocation audit (v2)

See [review-v2.md](artifacts/combined/review-v2.md), [source components](artifacts/combined/relationships/source-components-report.json), [bounded duplicate summary](artifacts/combined/duplicate-sample-summary.json), and [cross-media probe](artifacts/combined/cross-media-probe/report.json). The current proposal improves evaluation coverage by reserving the 518 official Celeb-DF entries as test candidates, while keeping actual assignments quarantined. Two disjoint DeepFakeDetection actor components (584 videos) and 1,882 MidJourney real/fake records are reserved for possible future evaluation, but are **not assigned to test** because cross-dataset source independence is unverified. Validation/calibration/test currently have no DF40 data and the release gate rejects them.

The bounded hash audit covers 2,056 unique files across a 2,000-record FF++ training-only sample and a 60-record stratified sample, with four files shared between the samples. No cross-partition exact-byte match appeared in that limited set. It is not a complete duplicate audit. A bounded visual probe compared one DF40 image with two possible Celeb-DF source videos; its perceptual distances are review aids, not identity decisions. The machine has approximately 59.8 GiB free. A four-video/16-candidate CUDA sample took 78.3 seconds including startup; 14 output PNGs from FF++, Celeb-DF and DF40 produced a tentative dataset-weighted estimate of 26.1 GiB, versus an 84.7 GiB conservative allowance for the current proposal. Neither estimate guarantees space or throughput.
# Best-effort allocation (current policy)

## One-command preparation

After checking free space and the release report, run `.\.venv\Scripts\python.exe scripts\prepare_dataset.py`. This is the full run, so start it only when ready for bulk processing. It verifies the frozen release hash and CUDA, keeps a 15 GiB free-space reserve, and writes final PNGs under `data/processed/xception-v1/<partition>/<real|fake>/<source-key>/`. It uses 10 temporal segments, three candidates per segment, and a 15 Laplacian-variance sharpness cutoff; the previous value of 30 rejected all 30 candidates from one visually usable DFD video. The 15 threshold is an engineering choice and may admit softer faces. Confidence 0.95, minimum face dimensions 80 pixels, and maximum padding 0.35 remain in force.

The `.state` directory holds resumable video and still checkpoints, the frozen preprocessing signature, progress log, and a status report. Diagnostics remain outside the dataset. A restart uses the same command. Changed preprocessing settings, code, or source metadata reject incompatible checkpoints; choose a new output location rather than changing parameters midway. The image manifest and completion report appear only after image decoding, dimensions, partition/group isolation and output-hash checks pass. Zero-yield videos and read errors are listed in the report. Identity independence remains unverified even after preparation completes.

For a bounded GPU smoke test, use `.\.venv\Scripts\python.exe scripts\prepare_dataset.py --smoke`; it writes only to `artifacts/diagnostics/prepare-smoke-v6`, selects one source from each dataset, and uses two segments with two candidates each. The sample is a functionality check, not a coverage or quality validation of the full release. The prior outputs under `artifacts/combined` are not reused. The free-space reserve is checked between sources; monitor space while processing a long video. A rough 10-frame estimate is 14.1 GiB with a 42.2 GiB three-times allowance, based on only four video crops and six stills.

The practical policy supersedes the older review prerequisites described below. It retains the existing conservative source components, including their suspected filename links; it does not assert that those links prove identity. Unknown person identities and cross-media relationships are limitations, not automatic blockers. The official Celeb-DF test list remains confined to test. Previously quarantined DF40 frames are only reconsidered when their existing source component is the same as an active held-out FF++ source; they remain in draft quarantine until the complete byte audit and release gate pass. Unknown labels, unresolved source conflicts and other existing quarantines are retained. The DF40 testing release is repurposed for an experimental split; this is not an official benchmark.

The 39,467-file v2 SHA-256 audit is reused via its size/mtime-validated cache. Reallocation can place identical bytes in different partitions, so entire conflicting source components are quarantined and the resulting active set is hashed again. Byte hashes do not find re-encoded videos or cropped frames, and do not establish identity independence. The complete practical hash report, component map and proposal are bound by SHA-256 at release; the release gate independently checks labels, component leakage, exact-byte leakage, coverage, source changes and the Celeb-DF boundary. The previous stricter review mode remains available without `--best-effort`.

The current reproducible draft uses seed 42, existing DF40 quotas of at most 10 images per original sample and 5,000 per method/label, plus at most 50 source-linked images per method and evaluation partition and five per original group. Quotas and differing image sources do not correct model bias; training-only sampling weights must be applied by the future trainer if used. The complete report records method and class counts. Small Celeb-DF validation/calibration sets and unverified person identity limit conclusions.

To reproduce the policy, use fresh output names (the allocation refuses to overwrite an existing draft):

```powershell
$py = '.\.venv\Scripts\python.exe'
& $py -m deepfake_data.practical_allocation --draft artifacts/combined/proposed-allocation-v2.csv --proposal artifacts/combined/proposed-report-v2.json --components artifacts/combined/relationships/source-components.csv --component-report artifacts/combined/relationships/source-components-report.json --hashes artifacts/combined/source-hashes-v2.csv --hash-report artifacts/combined/source-hashes-v2.json --output artifacts/combined/practical-draft-new.csv --report artifacts/combined/practical-proposal-new.json --seed 42
& $py -m deepfake_data.practical_allocation --supplement --draft artifacts/combined/practical-draft-new.csv --proposal artifacts/combined/practical-proposal-new.json --components artifacts/combined/relationships/source-components.csv --output artifacts/combined/practical-draft-supplement-new.csv --report artifacts/combined/practical-proposal-supplement-new.json
& $py -m deepfake_data.hash_sources --manifest artifacts/combined/practical-draft-supplement-new.csv --cache artifacts/combined/source-hashes.sqlite --output artifacts/combined/practical-hashes-new.csv --report artifacts/combined/practical-hashes-new.json --all --use-proposals
```

Never promote if the hash command reports overlaps or errors. If hashes report overlap, quarantine all affected components using `python -m deepfake_data.practical_allocation --quarantine-conflicts` with the matching draft, proposal, component mapping, hash CSV and report, and new output/report names; then repeat the full cache-backed hash command for the resulting draft. `deepfake_data.release --best-effort` requires the matching proposal, source-component and complete hash reports, checks them again and refuses to overwrite a frozen manifest. A successful release still does not mean crops, class balance or scientific evaluation have been validated. The existing strict-mode instructions below are historical and apply only when that mode is deliberately selected.

The current verified candidate and frozen output use:

```powershell
& $py -m deepfake_data.release --best-effort --draft artifacts/combined/practical-draft-v3.csv --proposal-report artifacts/combined/practical-proposal-v3.json --components artifacts/combined/relationships/source-components.csv --component-report artifacts/combined/relationships/source-components-report.json --hash-csv artifacts/combined/practical-hashes-v3.csv --hash-report artifacts/combined/practical-hashes-v3.json --output artifacts/combined/releases/best-effort-v1.csv --report artifacts/combined/release-best-effort-v1.json
```

The release command deliberately refuses to overwrite the existing manifest. Confirm `ready` in the release report and compare its `release_sha256` with `Get-FileHash -Algorithm SHA256` before extraction. The v2 expanded candidate contained 26 cross-partition duplicate hashes; 15 complete source components were quarantined before the v3 audit. The prior allocation had already quarantined 1,505 components whose hashes crossed its newly proposed partitions.

The frozen best-effort release has 34,622 active source records and seed 42. Its SHA-256 is `5aa0ce1f9c5f9db2b430f492ec672536dd98f63d375c17f0bb5bb644d4952aeb`. It is a preparation allocation, not a validated training or independent evaluation dataset. Training has 28,264 fake and 2,352 real source records; Celeb-DF contributes only 65 fake training videos, and its calibration subset contains no fake video. Do not interpret source-record balance as the final frame balance. The v3 report lists counts by method and label. A deliberately conservative earlier two-ID filename component rule remains in place; it can over-quarantine Celeb-DF examples and is not proof of identity. The DFD actor holdout candidates and MidJourney remain quarantined under the existing safeguards.

