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
