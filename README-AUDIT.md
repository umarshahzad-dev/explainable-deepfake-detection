# Dataset audit

## Verified observations

- The earlier `df40` configuration pointed to the parent of `extracted`: every method was reported as `extracted`, and 28 archive ZIPs were counted alongside extracted contents. The fixed configuration uses direct central dataset roots. FF++'s project-local root contains junctions, so pointing at the physical root avoids silently skipping both video trees.
- FF++ includes original and manipulated videos in `c23`. Celeb-DF-v2 includes an official testing-video list. DF40 has 28 extracted method directories and 82 metadata JSON files. All but `whichfaceisreal` have directly named metadata; this does not prove complete coverage.
- Inspected DF40 JSON uses collection, class, partition, sample and frame-path nesting. Some real references point into Celeb-DF-v2 and FaceForensics++, creating known provenance overlap. Source-provided real/fake labels are used only when mapping is unambiguous.
- The prior report recorded 9,431 FF++ videos, 6,529 Celeb-DF videos, 626,505 DF40 media, 352,516 NPY files and 30 ZIPs. Those are historical results, not independent validation of the new scan. Consult the current full `report.json` for actual disk accounting; limited previews cannot settle the PNG/JPEG discrepancy.

## Remaining risks

Physical media have not been comprehensively decoded, and no identity-safe independent test set has been verified. The file names and JSON group samples but do not alone prove cross-dataset independence. The experimental training protocol can include related content; strict evaluation quarantines unresolved relationships. MidJourney GIF animation and DF40 face-crop suitability require bounded image inspection with Pillow. Some JSON references refer to external release paths rather than local DF40 media; these are counted separately and are not silently classified as missing DF40 files. Metadata duplicates, conflicts, absent references and unlisted media appear in the generated report. An exact hash scan is optional and expensive.

## Change groups and suggested commits

- Inventory and source metadata: `fix(ingestion): reconcile dataset paths and DF40 metadata`
- Experimental and strict splits: `feat(data): add provenance-aware split protocols`
- Bounded extraction and image inspection: `feat(ingestion): add resumable face extraction`

The test suite exercises small synthetic fixtures; it is not a validation of the full physical dataset. See README for commands and dependency requirements.

## Follow-up audit and corrections

- The official FF++ documentation identifies manipulated names as `<target sequence>_<source sequence>` ([source](https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md)). Standard YouTube originals and derivatives are now connected through both available source identifiers. Local inspection found no FF++ official split files. The DeepFakeDetection actors layout uses different identifiers and is not asserted to follow this mapping.
- Celeb-DF synthesis names suggest relationships but do not by themselves verify video or identity linkage. They remain sample groups, not verified source or identity groups. DF40 `cdf`/`ff` domains are treated as suspected overlap with the respective holdout.
- Cross-dataset decisions are exported for review with sample examples and an evidence field. Suspected, verified and unknown overlaps cannot train in a held-out rotation; reviewed independence requires evidence. This is a review workflow, not automatic proof.
- The experimental within-dataset output is training-only because unresolved source overlap could otherwise contaminate its test set. The strict within-dataset split includes only verified source groups. Three bounded preview reports are generated under `artifacts/splits`; none has sufficient train/test class coverage, and no independence claim is made.
- A selected GIF conversion can now be joined to the image manifest without changing its original group or partition. Video extraction checkpoints completed frames and skipped-frame decisions in a durable log. These behaviors have synthetic tests but no physical decode test in the current environment.
- Full DF40 JSON reconciliation is still memory-intensive: one JSON file is loaded at a time, but the combined lookup is retained in memory. Full reconciled inventory remains pending; the output from a limited inventory is not a full audit.

- Bounded physical header checks found representative PNG dimensions of 256×256 (`e4s`), 512×512 (`CollabDiff`) and 1024×1024 (`MidJourney`). One `whichfaceisreal` file named `0.png` actually has a JPEG/JFIF signature. These are header checks, not face suitability or decode validation; extension counts and file-content counts can differ.
- A read-only FF++ source-ID smoke check found 1,000 YouTube originals forming 500 connected groups, with the largest group containing two source IDs. This validates the local filename grouping calculation, not identity separation.
- The earlier extension-count discrepancy is explained by `whichfaceisreal/__MACOSX`: 1,001 AppleDouble sidecars named `.png` and 1,000 named `.jpeg` account exactly for the reported differences (620,259 versus 619,258 PNG; 2,000 versus 1,000 JPEG). The inventory intentionally skips `__MACOSX`, which is not training media. A header audit found those sidecars have AppleDouble signatures; the corresponding non-sidecar files have JPEG signatures, including the 1,001 with `.png` names. The 28 outer archives explain the difference between 30 ZIPs under the parent root and two nested ZIPs inside `extracted`. These are format and path observations, not image decodability checks.
- A GIF block inspection found one image frame in each of the 247 MidJourney GIFs; none is animated by that structural check. The pixel data were not decoded.

## Full DF40 path-only accounting

A full recursive read-only scan of `df40/extracted` counted 620,259 `.png`, 6,000 `.jpg`, 2,000 `.jpeg`, 247 `.gif`, 352,516 `.npy`, two nested `.zip`, five extensionless files and two `.ds_store` files, including `__MACOSX`. Excluding the 1,001 PNG-named and 1,000 JPEG-named AppleDouble sidecars produces 619,258 `.png` and 1,000 `.jpeg`; the 28 archive ZIPs reside outside this root. The scan took approximately nine minutes on this machine. It counted paths only and did not reconcile all JSON entries or decode media.

## Full reconciliation and readiness run (2026-11-03)

- The path-only inventory indexed 9,431 FF++ videos, 6,529 Celeb-DF-v2 videos and 626,505 DF40 media. DF40 contains 428,292 matched fake and 4,000 matched real records, and 194,213 local media without a selected-JSON match. The latter remain unlabeled. There are 318,261 distinct unmatched metadata paths (510,473 JSON occurrences), 624,504 duplicate metadata records and no inaccessible roots in this scan. `artifacts/reconciliation/` contains full issue queues; external release references are counted separately.
- All 432,292 matched DF40 media appear in multiple original JSON partitions. Their labels can be unambiguous, but original-partition provenance is conflicting. This is recorded in `metadata_status` and held-out protocols quarantine those groups. Source JSON remains unchanged.
- The FF++ YouTube source audit connected 1,000 originals and all 5,000 standard manipulated derivatives into 500 groups. The 3,068 DeepFakeDetection manipulations and 363 actor originals remain unresolved. Celeb-DF's official test list contains 518 unique locally present entries: 178 real and 340 fake, with no label conflicts. DF40 JSON names 178 locally present Celeb-DF and 859 locally present FF++ source videos; these are metadata source references, not proof that every related DF40 frame is present.
- Bounded checks decoded eight DF40 sample images across six methods; six have metadata-backed label candidates (four fake, two real) and two remain unknown. Their full-reconciliation status exposes the partition conflicts. The whichfaceisreal sample has a `.png` suffix but JPEG content. Eight DF40 images and four video files were hashed in a bounded sample with no within-sample exact duplicates. This is not a full duplicate scan.
- The selected candidate for the single experiment is the strict FF++ source-grouped within-dataset protocol: 4,224 train videos, 720 validation and 1,056 test, with both labels in each. This is a feasible candidate, not identity-safe or independently evaluated. Both cross-dataset protocols currently contain no train or validation records because independence reviews are not complete. The FF++ batch extraction technical gate passes for a 200-video bound, while training/evaluation readiness fails; user approval is still required before extraction.
- A four-video CUDA sample decoded eight 299×299 crops and recorded two blurry and six lower-ranked candidates. A synthetic invalid MP4 produced a clear open failure. Full codec, frame-read and GPU peak-VRAM coverage remain unverified. The samples and all protocol/readiness reports are under `artifacts/` and are not Git-tracked.
