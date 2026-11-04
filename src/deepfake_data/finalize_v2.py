"""Build a SHA-verified, source-grouped DF40-focused release candidate.

This command reads source media and inventories only. It never decodes or copies
media. Release files are written only after all mandatory integrity gates pass.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .split import file_sha256

ACTIVE = ("train", "val", "calibration", "test")
EXCLUDED_METHODS = {"sadtalker", "wav2lip"}
TARGET_DF40_METHODS = {
    "CollabDiff", "DiT", "MRAA", "MidJourney", "RDDM", "SiT",
    "StyleGAN2", "StyleGAN3", "StyleGANXL", "e4s", "facedancer",
    "fsgan", "heygen", "hyperreenact", "inswap", "lia", "mcnet",
    "mobileswap", "one_shot_free", "pirender", "pixart", "simswap",
    "starganv2", "tpsm", "uniface", "whichfaceisreal",
}


class UnionFind:
    def __init__(self, values: Iterable[str] = ()) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: str, right: str) -> str:
        a, b = self.find(left), self.find(right)
        if a == b:
            return a
        root, child = sorted((a, b))
        self.parent[child] = root
        return root


def stable_partition(component: str, seed: int = 2026) -> str:
    """Assign whole source components deterministically, independent of row order."""
    n = int.from_bytes(hashlib.sha256(f"{seed}:{component}".encode()).digest()[:8], "big")
    bucket = n % 10000
    if bucket < 7000:
        return "train"
    if bucket < 8000:
        return "val"
    if bucket < 9000:
        return "calibration"
    return "test"


def hash_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _csv_rows(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        yield from csv.DictReader(stream)


def _load_mapping(path: Path, key: str, value: str) -> dict[str, str]:
    result = {}
    for row in _csv_rows(path):
        if row.get(key) and row.get(value):
            result[row[key]] = row[value]
    return result


def _load_component_details(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    ids, evidence = {}, {}
    for row in _csv_rows(path):
        group, component = row.get("group_id", ""), row.get("component_id", "")
        if group and component:
            if group in ids and (ids[group] != component or evidence[group] != row.get("evidence_status", "unlinked_or_within_dataset_only")):
                raise ValueError(f"Conflicting source component rows for {group}")
            ids[group] = component
            evidence[group] = row.get("evidence_status", "unlinked_or_within_dataset_only")
    return ids, evidence


def _load_old_groups(path: Path, corrected: Path) -> dict[str, str]:
    """Join the prior enriched grouping by dataset+relative path, never by row order."""
    groups: dict[str, str] = {}
    # Keep only the path keys and group strings in memory rather than 642k row dicts.
    valid_keys = {"\0".join((r.get("dataset", ""), r.get("relative_path", "")))
                  for r in _csv_rows(corrected)}
    for row in _csv_rows(path):
        key = "\0".join((row.get("dataset", ""), row.get("relative_path", "")))
        if key in valid_keys and row.get("group_id"):
            groups[key] = row["group_id"]
    return groups


def _load_baseline(path: Path) -> tuple[set[str], set[str], set[str]]:
    paths, groups, components = set(), set(), set()
    for row in _csv_rows(path):
        paths.add("\0".join((row.get("dataset", ""), row.get("relative_path", ""))))
        if row.get("group_id"):
            groups.add(row["group_id"])
        if row.get("source_component_id"):
            components.add(row["source_component_id"])
    return paths, groups, components


def _cache_hash(db: sqlite3.Connection, path: str, size: int, mtime_ns: int,
                prior_cache: dict[str, tuple[str, int, int]]) -> tuple[str, str]:
    cached = db.execute("SELECT sha256,size,mtime_ns,status FROM hashes WHERE path=?", (path,)).fetchone()
    if cached and cached[1] == size and cached[2] == mtime_ns and cached[3] == "ok":
        return cached[0], "resume_cache"
    old = prior_cache.get(path.casefold())
    if old and old[1:] == (size, mtime_ns):
        db.execute("INSERT OR REPLACE INTO hashes VALUES(?,?,?,?,?)", (path, old[0], size, mtime_ns, "ok"))
        return old[0], "completed_hash_audit"
    try:
        digest = hash_file(Path(path))
        after = Path(path).stat()
        if (after.st_size, after.st_mtime_ns) != (size, mtime_ns):
            raise OSError("source changed while hashing")
        db.execute("INSERT OR REPLACE INTO hashes VALUES(?,?,?,?,?)", (path, digest, size, mtime_ns, "ok"))
        return digest, "new_sha256"
    except (OSError, PermissionError) as exc:
        db.execute("INSERT OR REPLACE INTO hashes VALUES(?,?,?,?,?)", (path, "", size, mtime_ns, "error:" + str(exc)))
        return "", "hash_error"


def build_release(*, inventory: Path, old_groups: Path, components: Path,
                  completed_hashes: Path, baseline_selection: Path, output_dir: Path,
                  seed: int = 2026, progress_every: int = 5000,
                  protected_v1: Path | None = None,
                  inventory_report: Path | None = None,
                  component_report: Path | None = None,
                  completed_hash_report: Path | None = None,
                  release_filename: str = "df40-focused-v2.csv",
                  expected_v1_sha256: str = "5aa0ce1f9c5f9db2b430f492ec672536dd98f63d375c17f0bb5bb644d4952aeb") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if Path(release_filename).name != release_filename or not release_filename.endswith(".csv"):
        raise ValueError("Release filename must be a plain CSV filename")
    release_path = output_dir / release_filename
    if release_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing release: {release_path}")
    state_path = output_dir / "allocation-hash-cache.sqlite3"
    db = sqlite3.connect(state_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("CREATE TABLE IF NOT EXISTS hashes(path TEXT PRIMARY KEY,sha256 TEXT,size INTEGER,mtime_ns INTEGER,status TEXT)")
    db.execute("CREATE TABLE IF NOT EXISTS records(id INTEGER PRIMARY KEY,dataset TEXT,method TEXT,label TEXT,media_type TEXT,path TEXT,rel TEXT,group_id TEXT,component TEXT,source_status TEXT,official_test INTEGER,baseline INTEGER,sha256 TEXT,hash_status TEXT,metadata_status TEXT,payload TEXT,partition TEXT,reason TEXT)")
    columns = {row[1] for row in db.execute("PRAGMA table_info(records)")}
    if "relationship_component" not in columns:
        db.execute("ALTER TABLE records ADD COLUMN relationship_component TEXT")
    if "component_evidence" not in columns:
        db.execute("ALTER TABLE records ADD COLUMN component_evidence TEXT")
    db.execute("CREATE TABLE IF NOT EXISTS run_meta(key TEXT PRIMARY KEY,value TEXT)")
    if protected_v1 is not None:
        protected_digest = file_sha256(protected_v1)
        if protected_digest != expected_v1_sha256:
            raise ValueError("Frozen v1 release digest does not match its preservation audit")
    else:
        protected_digest = ""
    inventory_digest = file_sha256(inventory)
    if inventory_report is not None:
        report = json.loads(inventory_report.read_text(encoding="utf-8"))
        if report.get("manifest_sha256") != inventory_digest:
            raise ValueError("Phase 1 inventory does not match its validation report")
    if component_report is not None:
        report = json.loads(component_report.read_text(encoding="utf-8"))
        if report.get("component_csv_sha256") != file_sha256(components):
            raise ValueError("Source component map does not match its evidence report")
    if completed_hash_report is not None:
        report = json.loads(completed_hash_report.read_text(encoding="utf-8"))
        actual_hash_csv_digest = file_sha256(completed_hashes)
        if (report.get("complete") is not True or report.get("errors") or
                report.get("manifest_changed_during_scan") is not False or
                report.get("hash_csv_sha256") != actual_hash_csv_digest):
            raise ValueError("Completed SHA audit CSV is incomplete or does not match its report")
    source_fingerprints = {"policy_version": "3-empty-file-quarantine-and-best-effort-suspected-links",
                           "inventory": inventory_digest, "old_groups": file_sha256(old_groups),
                           "components": file_sha256(components), "completed_hashes": file_sha256(completed_hashes),
                           "baseline_selection": file_sha256(baseline_selection), "seed": str(seed)}
    for name, path in (("inventory_report", inventory_report), ("component_report", component_report),
                       ("completed_hash_report", completed_hash_report)):
        if path is not None:
            source_fingerprints[name] = file_sha256(path)
    stage_signature = hashlib.sha256(json.dumps(source_fingerprints, sort_keys=True).encode()).hexdigest()
    # A cache may be resumed only for files with exactly matching size and nanosecond mtime.
    prior_cache: dict[str, tuple[str, int, int]] = {}
    for row in _csv_rows(completed_hashes):
        if row.get("status") == "ok" and row.get("sha256"):
            try:
                prior_cache[str(Path(row["absolute_path"]).resolve()).casefold()] = (
                    row["sha256"], int(row["size"]), int(row["mtime_ns"]))
            except (KeyError, ValueError, OSError):
                continue
    staged_signature = db.execute("SELECT value FROM run_meta WHERE key='stage_signature'").fetchone()
    stage_reused = bool(staged_signature and staged_signature[0] == stage_signature and db.execute("SELECT COUNT(*) FROM records").fetchone()[0])
    baseline_paths, baseline_groups, baseline_components = _load_baseline(baseline_selection)
    duplicate_paths = 0
    if not stage_reused:
        group_component, group_evidence = _load_component_details(components)
        enriched = _load_old_groups(old_groups, inventory)
        db.execute("DELETE FROM records")
        db.commit()
        candidates = 0
        quarantines = Counter()
        for row in _csv_rows(inventory):
            ds, method = row.get("dataset", ""), row.get("method", "")
            label, media = row.get("label", ""), row.get("media_type", "")
            selected_method = ds != "df40" or (method in TARGET_DF40_METHODS and method.casefold() not in EXCLUDED_METHODS)
            eligibility = row.get("experiment_eligibility") == "eligible_labeled_record"
            supervised = label in {"0", "1"} and media in {"video", "image", "gif"}
            if ds not in {"ffpp", "celebdf", "df40"}:
                continue
            key = "\0".join((ds, row.get("relative_path", "")))
            path = row.get("absolute_path", "")
            original_group = row.get("group_id", "")
            group = enriched.get(key) or original_group
            relationship_component = group_component.get(group, group)
            component_evidence = group_evidence.get(group, "unknown_group_only")
            # The existing graph deliberately labels filename/domain links as suspected.
            # Preserve those IDs as evidence, but only split on the source/sample group
            # itself so a broad heuristic component cannot consume the entire method.
            component = group if component_evidence == "contains_suspected_links" else relationship_component
            is_official_test = int(ds == "celebdf" and row.get("original_partition", "") == "official_test")
            verified_baseline_component = (component_evidence != "contains_suspected_links" and
                                            relationship_component in baseline_components)
            is_baseline = int(key in baseline_paths or group in baseline_groups or verified_baseline_component)
            reasons = []
            partition = "quarantine"
            if not selected_method:
                partition, reasons = "excluded", ["method_excluded_by_experiment" if method.casefold() in EXCLUDED_METHODS else "method_not_in_target_set"]
            elif not eligibility:
                reasons = [row.get("eligibility_reason") or "not_eligible_by_corrected_inventory"]
            elif not supervised:
                reasons = ["label_or_media_type_unresolved"]
            elif not group:
                reasons = ["source_group_unresolved"]
            elif not Path(path).is_file():
                reasons = ["media_missing_before_hash"]
            elif is_official_test or is_baseline:
                partition = "test"
                reasons = ["official_celebdf_test_boundary" if is_official_test else "reserved_df40_baseline_development_component"]
            else:
                partition, reasons = "pending", ["source_group_hash_allocation_pending"]
            if partition in {"quarantine", "excluded"}:
                quarantines[(partition, reasons[0])] += 1
            if selected_method and eligibility and supervised and Path(path).is_file():
                candidates += 1
            payload = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            db.execute("INSERT INTO records(dataset,method,label,media_type,path,rel,group_id,component,source_status,official_test,baseline,metadata_status,payload,partition,reason,relationship_component,component_evidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (ds, method, label, media, path, row.get("relative_path", ""), group, component,
                        row.get("relationship_status", "unknown"), is_official_test, is_baseline,
                        row.get("metadata_status", ""), payload, partition, ";".join(reasons), relationship_component, component_evidence))
            if candidates and candidates % 5000 == 0:
                db.commit()
                print(f"Inventory rows staged; hash candidates={candidates:,}", flush=True)
        db.commit()
        duplicate_paths = db.execute("SELECT COUNT(*) FROM records WHERE (dataset,rel) IN (SELECT dataset,rel FROM records GROUP BY dataset,rel HAVING COUNT(*)>1)").fetchone()[0]
        if duplicate_paths:
            db.execute("UPDATE records SET partition='quarantine',reason='duplicate_inventory_path' WHERE (dataset,rel) IN (SELECT dataset,rel FROM records GROUP BY dataset,rel HAVING COUNT(*)>1)")
        db.execute("INSERT OR REPLACE INTO run_meta VALUES('stage_signature',?)", (stage_signature,))
        db.commit()
    else:
        print("Reusing staged inventory and completed hash checkpoints", flush=True)
        duplicate_paths = db.execute("SELECT COUNT(*) FROM records WHERE partition='quarantine' AND reason='duplicate_inventory_path'").fetchone()[0]

    # Hash all candidate media. Existing audit/cache entries are reused only after stat verification.
    candidates = db.execute("SELECT COUNT(*) FROM records WHERE partition IN ('pending','test') AND label IN ('0','1') AND path<>''").fetchone()[0]
    hashed = 0
    last_id = 0
    while True:
        candidate_rows = db.execute("SELECT id,path FROM records WHERE id>? AND partition IN ('pending','test') AND label IN ('0','1') AND path<>'' ORDER BY id LIMIT 500", (last_id,)).fetchall()
        if not candidate_rows:
            break
        for record_id, path in candidate_rows:
            last_id = record_id
            try:
                source_path = Path(path)
                stat = source_path.stat()
                if stat.st_size == 0:
                    db.execute("UPDATE records SET partition='quarantine',reason='empty_source_file',sha256='',hash_status='empty_file' WHERE id=?", (record_id,))
                    hashed += 1
                    if hashed % progress_every == 0:
                        db.commit()
                        print(f"SHA-256 checked {hashed:,}/{candidates:,} eligible files", flush=True)
                    continue
                cache_path = str(source_path.resolve()).casefold()
                digest, status = _cache_hash(db, cache_path, stat.st_size, stat.st_mtime_ns, prior_cache)
                if not digest:
                    db.execute("UPDATE records SET partition='quarantine',reason='source_sha256_failed',sha256='',hash_status=? WHERE id=?", (status, record_id))
                else:
                    db.execute("UPDATE records SET sha256=?,hash_status=? WHERE id=?", (digest, status, record_id))
            except OSError as exc:
                db.execute("UPDATE records SET partition='quarantine',reason=?,hash_status='stat_error' WHERE id=?", ("source_stat_failed:" + str(exc), record_id))
            hashed += 1
            if hashed % progress_every == 0:
                db.commit()
                print(f"SHA-256 checked {hashed:,}/{candidates:,} eligible files", flush=True)
    db.commit()

    # Exact-byte duplicates join source components before allocation. A digest with both labels
    # is quarantined as a label conflict; identical content can never cross partitions.
    groups = {r[0] for r in db.execute("SELECT DISTINCT component FROM records WHERE component<>'' AND partition IN ('pending','test')")}
    uf = UnionFind(groups)
    digest_first: dict[str, str] = {}
    digest_labels: dict[str, set[str]] = defaultdict(set)
    for component, digest, label in db.execute("SELECT component,sha256,label FROM records WHERE sha256<>'' AND partition IN ('pending','test')"):
        digest_labels[digest].add(label)
        if digest in digest_first:
            uf.union(component, digest_first[digest])
        else:
            digest_first[digest] = component
    # Fold all components carrying a hash with baseline or official test anchors into the test group.
    forced_test = set()
    for component, part, baseline, official in db.execute("SELECT DISTINCT component,partition,baseline,official_test FROM records WHERE component<>''"):
        if part == "test" or baseline or official:
            forced_test.add(uf.find(component))
    forced_test = {uf.find(value) for value in forced_test}
    conflicting_digests = {digest for digest, labels in digest_labels.items() if len(labels) > 1}
    conflicting_roots = set()
    for component, digest in db.execute("SELECT component,sha256 FROM records WHERE sha256<>''"):
        if digest in conflicting_digests:
            conflicting_roots.add(uf.find(component))
    last_id = 0
    while True:
        allocation_rows = db.execute("SELECT id,component,sha256,partition,baseline,official_test FROM records WHERE id>? AND partition IN ('pending','test') ORDER BY id LIMIT 1000", (last_id,)).fetchall()
        if not allocation_rows:
            break
        for row_id, component, digest, original_part, baseline, official in allocation_rows:
            last_id = row_id
            root = uf.find(component)
            if root in conflicting_roots:
                db.execute("UPDATE records SET partition='quarantine',reason='exact_byte_label_conflict' WHERE id=?", (row_id,))
            elif original_part == "test" or root in forced_test or baseline or official:
                db.execute("UPDATE records SET partition='test',reason=CASE WHEN official_test=1 THEN 'official_celebdf_test_boundary' WHEN baseline=1 THEN 'reserved_df40_baseline_development_component' ELSE 'exact_byte_link_to_reserved_test_component' END WHERE id=?", (row_id,))
            else:
                db.execute("UPDATE records SET partition=?,reason='deterministic_source_component_split' WHERE id=?", (stable_partition(root, seed), row_id))
    db.commit()

    # Summaries and fail-closed checks.
    blockers: list[str] = []
    rows = db.execute("SELECT dataset,method,label,media_type,group_id,component,sha256,partition,reason,hash_status,official_test,baseline,source_status,relationship_component,component_evidence FROM records")
    counts = Counter(); method_counts = Counter(); groups_by_part = defaultdict(set); hashes_by_part = defaultdict(set)
    groups_by_detail = defaultdict(set)
    image_estimates = Counter(); byte_estimates = Counter()
    leakage_components = defaultdict(set); hash_partitions = defaultdict(set)
    active_method_fake = set(); active_df40 = set(); unknown_relation = Counter(); quarantined = Counter()
    suspected_component_partitions = defaultdict(set)
    for ds, method, label, media, group, component, digest, part, reason, hash_status, official, baseline, rel_status, rel_component, component_evidence in rows:
        counts[(ds, method, label, part)] += 1
        method_counts[(ds, method, label, part, media)] += 1
        if part in ACTIVE:
            if label not in {"0", "1"}:
                blockers.append(f"active record has unknown label: {ds}/{method}/{group}")
            groups_by_part[(part, ds)].add(component)
            groups_by_detail[(part, ds, method, label)].add(group or component)
            leakage_components[component].add(part)
            if component_evidence == "contains_suspected_links":
                suspected_component_partitions[rel_component].add(part)
            if not digest or hash_status not in {"new_sha256", "completed_hash_audit", "resume_cache"}:
                blockers.append(f"active record lacks verified hash: {ds}/{method}/{group}")
            else:
                hash_partitions[digest].add(part)
            if ds == "df40":
                active_df40.add(method)
                if label == "0" and part == "train":
                    active_method_fake.add(method)
            if rel_status in {"unknown", "suspected", "", None}:
                unknown_relation[part] += 1
            average_bytes = {"ffpp": 130860, "celebdf": 99465, "df40": 105672}[ds]
            expected_frames = 9.26 if media == "video" else 1.0
            image_estimates[(part, ds, label)] += expected_frames
            byte_estimates[(part, ds, label)] += expected_frames * average_bytes
        elif part == "quarantine":
            quarantined[reason] += 1
    component_overlap = {k: sorted(v) for k, v in leakage_components.items() if len(v) > 1}
    suspected_overlaps = {k: sorted(v) for k, v in suspected_component_partitions.items() if len(v) > 1}
    hash_overlap = {k: sorted(v) for k, v in hash_partitions.items() if len(v) > 1}
    if component_overlap:
        blockers.append(f"{len(component_overlap)} source components cross active partitions")
    if hash_overlap:
        blockers.append(f"{len(hash_overlap)} exact SHA-256 values cross active partitions")
    if active_method_fake != TARGET_DF40_METHODS:
        blockers.append("DF40 target methods without fake training data: " + ", ".join(sorted(TARGET_DF40_METHODS-active_method_fake)))
    for part in ACTIVE:
        present = {label for (ds, method, label, p), n in counts.items() if p == part and n and label in {"0", "1"}}
        if present != {"0", "1"}:
            blockers.append(f"partition {part} does not contain both verified classes")
    for ds in ("ffpp", "celebdf", "df40"):
        present = {label for (d, method, label, part), n in counts.items() if d == ds and part == "train" and n and label in {"0", "1"}}
        if present != {"0", "1"}:
            blockers.append(f"training dataset {ds} does not contain both verified classes")
    official_not_test = db.execute("SELECT COUNT(*) FROM records WHERE official_test=1 AND partition<>'test'").fetchone()[0]
    if official_not_test:
        blockers.append(f"{official_not_test} official Celeb-DF test records are not in test")
    baseline_paths_missing = sum(
        1 for key in baseline_paths
        if not db.execute("SELECT 1 FROM records WHERE dataset=? AND rel=? LIMIT 1", tuple(key.split("\0", 1))).fetchone())
    if baseline_paths_missing:
        blockers.append(f"{baseline_paths_missing} selected DF40 baseline evaluation paths are absent from the corrected inventory")
    test_baseline = db.execute("SELECT COUNT(*) FROM records WHERE baseline=1 AND partition='train'").fetchone()[0]
    if test_baseline:
        blockers.append(f"{test_baseline} baseline selection records leaked into training")
    if len(active_df40) != 26:
        blockers.append(f"only {len(active_df40)} of 26 target DF40 methods appear in active partitions")

    protected_hash = ""
    if protected_v1 is not None:
        try:
            protected_hash = file_sha256(protected_v1)
            if protected_hash != expected_v1_sha256:
                blockers.append("Frozen v1 release digest does not match its preservation audit")
        except OSError as exc:
            blockers.append(f"Frozen v1 release unavailable: {exc}")
    summary = {
        "status": "PASS" if not blockers else "FAIL",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_ratios": {"train": .70, "val": .10, "calibration": .10, "test": .10},
        "policy": "All eligible labeled records retained without global caps; verified source/sample components and exact-byte duplicates assigned together; components marked contains_suspected_links are preserved as evidence but split at their source/sample group boundary; official Celeb-DF test and selected DF40 baseline development groups are pinned to test; unknown labels and integrity conflicts quarantined; SadTalker/Wav2Lip excluded.",
        "input_sha256": {str(inventory): inventory_digest, str(old_groups): file_sha256(old_groups), str(components): file_sha256(components), str(completed_hashes): file_sha256(completed_hashes), str(baseline_selection): file_sha256(baseline_selection), **({str(inventory_report): file_sha256(inventory_report)} if inventory_report else {}), **({str(component_report): file_sha256(component_report)} if component_report else {}), **({str(completed_hash_report): file_sha256(completed_hash_report)} if completed_hash_report else {}), **({str(protected_v1): protected_hash} if protected_v1 else {})},
        "candidate_media_hashed": hashed,
        "hash_cache_reuse_count": db.execute("SELECT COUNT(*) FROM records WHERE hash_status='resume_cache'").fetchone()[0],
        "prior_audit_reuse_count": db.execute("SELECT COUNT(*) FROM records WHERE hash_status='completed_hash_audit'").fetchone()[0],
        "new_hash_count": db.execute("SELECT COUNT(*) FROM records WHERE hash_status='new_sha256'").fetchone()[0],
        "hash_errors": db.execute("SELECT COUNT(*) FROM records WHERE partition='quarantine' AND (reason LIKE '%hash%' OR reason LIKE '%stat%')").fetchone()[0],
        "empty_source_files_quarantined": db.execute("SELECT COUNT(*) FROM records WHERE partition='quarantine' AND reason='empty_source_file'").fetchone()[0],
        "verified_nonempty_sha256_records": db.execute("SELECT COUNT(*) FROM records WHERE sha256<>'' AND hash_status IN ('new_sha256','completed_hash_audit','resume_cache')").fetchone()[0],
        "counts": [{"dataset": k[0], "method": k[1], "label": k[2], "partition": k[3], "count": n} for k,n in sorted(counts.items())],
        "method_media_counts": [{"dataset":k[0],"method":k[1],"label":k[2],"partition":k[3],"media_type":k[4],"count":n} for k,n in sorted(method_counts.items())],
        "source_group_counts": {p: {d: len(groups_by_part[(p,d)]) for d in ("ffpp","celebdf","df40")} for p in ACTIVE},
        "quarantine_reasons": dict(quarantined),
        "relationship_status_warnings_by_partition": dict(unknown_relation),
        "exact_duplicate_digest_count": db.execute("SELECT COUNT(*) FROM (SELECT sha256 FROM records WHERE sha256<>'' GROUP BY sha256 HAVING COUNT(*)>1)").fetchone()[0],
        "exact_duplicate_extra_records": db.execute("SELECT COALESCE(SUM(n-1),0) FROM (SELECT COUNT(*) n FROM records WHERE sha256<>'' GROUP BY sha256 HAVING COUNT(*)>1)").fetchone()[0],
        "exact_hash_conflicting_label_digests": len(conflicting_digests),
        "source_component_partition_leakage": component_overlap,
        "suspected_relationship_component_cross_partition_count": len(suspected_overlaps),
        "suspected_relationship_component_cross_partition_examples": dict(list(sorted(suspected_overlaps.items()))[:100]),
        "cross_partition_exact_hashes": hash_overlap,
        "target_df40_methods": sorted(TARGET_DF40_METHODS),
        "target_df40_methods_with_fake_training": sorted(active_method_fake),
        "reserved_baseline_selection_images": len(baseline_paths),
        "baseline_selection_paths_missing_from_inventory": baseline_paths_missing,
        "baseline_eval_records_in_train": test_baseline,
        "official_celeb_test_violations": official_not_test,
        "limitations": ["Filename/domain relationship evidence remains suspected and does not establish person identity.", "Full byte hashes detect exact file copies only; recompression and derived-frame overlap are not established.", "The baseline DF40 selection was already used for development comparison; test partition is therefore not an untouched benchmark.", "DF40 testing-release provenance is retained; this custom experiment is not an official DF40 benchmark.", "DF40 fake/real source domains differ because only selected methods provide verified real references; use dataset/method/group-aware sampling."],
        "blockers": blockers,
    }

    # Only freeze after integrity and leakage gates pass; otherwise retain diagnostics and resumable hash DB.
    summary["duplicate_inventory_paths_quarantined"] = duplicate_paths
    summary["source_group_counts_by_dataset_method_label_partition"] = [
        {"partition":k[0],"dataset":k[1],"method":k[2],"label":k[3],"groups":len(v)}
        for k,v in sorted(groups_by_detail.items())]
    summary["df40_real_training_methods"] = sorted({m for (ds,m,label,part),n in counts.items() if ds=="df40" and label=="1" and part=="train" and n})
    summary["method_coverage_warnings"] = []
    for method in sorted(TARGET_DF40_METHODS):
        fake_groups = len(groups_by_detail.get(("train","df40",method,"0"),set()))
        real_groups = len(groups_by_detail.get(("train","df40",method,"1"),set()))
        if fake_groups < 10:
            fake_count = counts.get(("df40",method,"0","train"),0)
            summary["method_coverage_warnings"].append(f"{method}: only {fake_count} fake records across {fake_groups} training source groups")
        if real_groups:
            continue
        real_test = counts.get(("df40",method,"1","test"),0)
        if real_test:
            summary["method_coverage_warnings"].append(f"{method}: no real training groups; {real_test} verified real records are in test")
    partition_labels = Counter()
    dataset_partition_labels = Counter()
    for item in summary["counts"]:
        if item["partition"] in ACTIVE and item["label"] in {"0", "1"}:
            partition_labels[(item["partition"], item["label"])] += item["count"]
            dataset_partition_labels[(item["dataset"], item["partition"], item["label"])] += item["count"]
    summary["partition_class_totals"] = [
        {"partition":p,"label":l,"count":partition_labels[(p,l)]}
        for p in ACTIVE for l in ("0","1")]
    summary["dataset_partition_class_totals"] = [
        {"dataset":d,"partition":p,"label":l,"count":n}
        for (d,p,l),n in sorted(dataset_partition_labels.items())]
    summary["expected_processed_images"] = [
        {"partition":k[0],"dataset":k[1],"label":k[2],"estimated_images":round(v)}
        for k,v in sorted(image_estimates.items())]
    expected_by_partition_label = Counter()
    for item in summary["expected_processed_images"]:
        expected_by_partition_label[(item["partition"], item["label"])] += item["estimated_images"]
    summary["expected_processed_images_by_partition_class"] = [
        {"partition":p,"label":l,"estimated_images":expected_by_partition_label[(p,l)]}
        for p in ACTIVE for l in ("0","1")]
    train_estimates = {row["label"]: sum(item["estimated_images"] for item in summary["expected_processed_images"] if item["partition"]=="train" and item["label"]==row["label"]) for row in ({"label":"0"},{"label":"1"})}
    summary["expected_train_fake_fraction"] = round(train_estimates["0"] / max(1, sum(train_estimates.values())), 4)
    summary["estimated_png_storage_bytes"] = round(sum(byte_estimates.values()))
    summary["free_bytes_at_allocation_run"] = shutil.disk_usage(output_dir).free
    summary["reuse_policy"] = {"existing_v1_png_count": 48529, "approved_reuse_count": 0,
                               "reason": "v1 checkpoint provenance exists but per-image crop hashes have not been verified by the release importer"}
    summary["training_sampler_policy"] = {
        "stored_samples": "retain every active sample; do not downsample the allocation to balance files",
        "batch_exposure": "sample label 50/50, then dataset DF40 50%, FF++ 25%, Celeb-DF 25% within each class",
        "within_fake_dataset": "sample manipulation method uniformly within the chosen dataset, then sample a source group uniformly; draw one image per group in a batch and rotate eligible frames/images across epochs",
        "within_real_dataset": "sample real source groups uniformly; DF40 real groups are available only from the methods with verified real references",
        "method_imbalance_control": "uniform method exposure prevents large DF40 methods from dominating; small methods repeat more often, so monitor per-method validation and overfitting",
        "real_domain_caveat": "only a subset of DF40 methods has verified real-reference imagery; dataset and manipulation shortcuts remain possible"}
    report_path = output_dir / "readiness.json"
    if blockers:
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _write_human_report(output_dir / "readiness.md", summary)
        db.close()
        print(json.dumps({"status": "FAIL", "blockers": blockers}, indent=2), flush=True)
        return summary

    # Stable release rows include quarantined/excluded inventory records for auditable accounting.
    db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    fields = ["dataset", "release", "method", "source_domain", "relative_path", "absolute_path", "media_type", "label", "original_json_label", "original_partition", "project_partition", "sample_id", "group_id", "source_group_id", "source_component_id", "relationship_component_id", "component_evidence_status", "source_group_status", "relationship_status", "validation_status", "metadata_sources", "metadata_status", "experiment_eligibility", "eligibility_reason", "sha256", "sha_status", "partition_reason", "quarantine_reason", "baseline_eval_reserved"]
    temp = release_path.with_suffix(".csv.partial")
    with temp.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        query = "SELECT payload,group_id,component,partition,reason,sha256,hash_status,relationship_component,component_evidence FROM records ORDER BY dataset,method,rel"
        for payload, group, component, part, reason, digest, hash_status, relationship_component, component_evidence in db.execute(query):
            row = json.loads(payload)
            row.update({"project_partition": part, "source_group_id": group, "source_component_id": component, "relationship_component_id": relationship_component, "component_evidence_status": component_evidence, "source_group_status": row.get("group_status", "unknown"), "sha256": digest or "", "sha_status": hash_status or "not_hashed", "partition_reason": reason if part in ACTIVE else "", "quarantine_reason": reason if part == "quarantine" else (reason if part == "excluded" else ""), "baseline_eval_reserved": "true" if "baseline" in reason else "false"})
            writer.writerow(row)
        out.flush()
        os.fsync(out.fileno())
    # Windows rename is atomic and refuses an existing destination.
    if os.name == "nt":
        os.rename(temp, release_path)
    else:
        os.link(temp, release_path)
        temp.unlink()
    summary["release_path"] = str(release_path)
    summary["release_version"] = Path(release_filename).stem
    summary["release_sha256"] = file_sha256(release_path)
    summary["status"] = "PASS_FROZEN"
    count_path = output_dir / "allocation-counts.csv"
    with count_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["dataset", "method", "label", "partition", "count"])
        writer.writeheader()
        writer.writerows(summary["counts"])
    group_path = output_dir / "source-group-counts.csv"
    with group_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["partition", "dataset", "method", "label", "groups"])
        writer.writeheader()
        writer.writerows(summary["source_group_counts_by_dataset_method_label_partition"])
    summary["allocation_counts_sha256"] = file_sha256(count_path)
    summary["source_group_counts_sha256"] = file_sha256(group_path)
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_human_report(output_dir / "readiness.md", summary)
    db.close()
    return summary


def _write_human_report(path: Path, report: dict) -> None:
    lines = ["# DF40-focused v2 allocation readiness", "", f"Status: **{report['status']}**", "", f"Seed: `{report['seed']}`", "", f"Policy: {report.get('policy','')}", "", f"Release SHA-256: `{report.get('release_sha256','not frozen')}`", "", "## Partition class totals", "", "| Partition | Fake (0) records | Real (1) records |", "|---|---:|---:|"]
    class_totals = {(r["partition"],r["label"]):r["count"] for r in report.get("partition_class_totals", [])}
    for part in ACTIVE:
        lines.append(f"| {part} | {class_totals.get((part,'0'),0):,} | {class_totals.get((part,'1'),0):,} |")
    lines.extend(["", "## Mandatory checks", ""])
    if report["blockers"]:
        lines.extend(["FAIL:", *[f"- {item}" for item in report["blockers"]]])
    else:
        lines.append("All configured integrity, class coverage, group, exact-hash and method-coverage gates passed.")
    lines.extend(["", "## Leakage and provenance checks", "",
                  f"- Eligible paths SHA-256 checked: {report.get('candidate_media_hashed',0):,}; hash errors: {report.get('hash_errors',0):,}.",
                  f"- Prior completed audit rows revalidated: {report.get('sha256_audit',{}).get('prior_audit_records_revalidated_exactly',report.get('prior_audit_reuse_count',0)):,}.",
                  f"- Verified source-group components crossing partitions: {len(report.get('source_component_partition_leakage',{}))}.",
                  f"- Exact SHA-256 values crossing partitions: {len(report.get('cross_partition_exact_hashes',{}))}.",
                  f"- Suspected relationship components spanning partitions: {report.get('suspected_relationship_component_cross_partition_count',0):,} (recorded as unresolved evidence).",
                  f"- Existing DF40 baseline selection: {report.get('reserved_baseline_selection_images',0):,} selected paths; missing {report.get('baseline_selection_paths_missing_from_inventory',0)}; training leakage {report.get('baseline_eval_records_in_train',0)}.",
                  f"- Target DF40 fake methods represented in train: {len(report.get('target_df40_methods_with_fake_training',[]))}/26.",
                  f"- DF40 methods with real training groups: {', '.join(report.get('df40_real_training_methods',[])) or 'none'}.",
                  f"- Quarantined empty files: {report.get('sha256_audit',{}).get('zero_byte_files_quarantined',report.get('empty_source_files_quarantined',0))}; exact-byte label-conflict rows: {report.get('quarantine_reason_detail',{}).get('exact_byte_label_conflict_rows',0)}; unresolved-label rows: {report.get('quarantine_reasons',{}).get('verified binary label is unavailable or conflicting',0):,}.",
                  "- Official Celeb-DF test records remain in test; DF40 testing-release provenance remains explicit."])
    if report.get("method_coverage_warnings"):
        lines.extend(["", "Warnings:", *[f"- {item}" for item in report["method_coverage_warnings"]]])
    lines.extend(["", "## Counts by dataset, method, label and partition", "", "| Dataset | Method | Label | Partition | Records |", "|---|---|---:|---|---:|"])
    for row in report["counts"]:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['label']} | {row['partition']} | {row['count']} |")
    lines.extend(["", "## Source groups by partition", ""])
    for part, values in report["source_group_counts"].items():
        lines.append(f"- {part}: " + ", ".join(f"{ds} {n:,}" for ds,n in values.items()))
    if report.get("expected_processed_images_by_partition_class"):
        image_totals = {(r["partition"],r["label"]):r["estimated_images"] for r in report["expected_processed_images_by_partition_class"]}
        lines.extend(["", "## Estimated processed image totals", "", "| Partition | Fake (0) images | Real (1) images |", "|---|---:|---:|"])
        for part in ACTIVE:
            lines.append(f"| {part} | {image_totals.get((part,'0'),0):,} | {image_totals.get((part,'1'),0):,} |")
        lines.append(f"\nEstimated storage: {report.get('estimated_png_storage_bytes',0)/1024**3:.2f} GiB; free at allocation run: {report.get('free_bytes_at_allocation_run',0)/1024**3:.2f} GiB.")
    if report.get("training_sampler_policy"):
        lines.extend(["", "## Training exposure policy", ""])
        for key,value in report["training_sampler_policy"].items():
            lines.append(f"- {key.replace('_',' ')}: {value}")
    if report.get("phase3b_crop_reuse_ready") is not None:
        lines.extend(["", f"## Phase 3B crop reuse: {'READY' if report['phase3b_crop_reuse_ready'] else 'NOT READY'}", ""])
        lines.extend([f"- {item}" for item in report.get("phase3b_crop_reuse_blockers",[])])
    lines.extend(["", "## Limitations", *[f"- {item}" for item in report["limitations"]], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--inventory", type=Path, default=root / "artifacts/inventory/df40-phase1-final/manifest.csv")
    parser.add_argument("--old-groups", type=Path, default=root / "artifacts/combined/practical-draft-v3.csv")
    parser.add_argument("--components", type=Path, default=root / "artifacts/combined/relationships/source-components.csv")
    parser.add_argument("--completed-hashes", type=Path, default=root / "artifacts/combined/source-hashes-v2.csv")
    parser.add_argument("--baseline-selection", type=Path, default=root / "artifacts/evaluations/df40-baseline-v1-corrected-audit/selection.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "artifacts/allocation/df40-focused-v2.1-final")
    parser.add_argument("--release-filename", default="df40-focused-v2.1.csv")
    parser.add_argument("--protected-v1", type=Path, default=root / "artifacts/combined/releases/best-effort-v1.csv")
    parser.add_argument("--inventory-report", type=Path, default=root / "artifacts/inventory/df40-phase1-final/report.json")
    parser.add_argument("--component-report", type=Path, default=root / "artifacts/combined/relationships/source-components-report.json")
    parser.add_argument("--completed-hash-report", type=Path, default=root / "artifacts/combined/source-hashes-v2.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--progress-every", type=int, default=5000)
    args = parser.parse_args(argv)
    result = build_release(inventory=args.inventory, old_groups=args.old_groups,
                           components=args.components, completed_hashes=args.completed_hashes,
                           baseline_selection=args.baseline_selection, output_dir=args.output_dir,
                           seed=args.seed, progress_every=args.progress_every, protected_v1=args.protected_v1,
                           inventory_report=args.inventory_report, component_report=args.component_report,
                           completed_hash_report=args.completed_hash_report, release_filename=args.release_filename)
    print(json.dumps({"status": result["status"], "blockers": result["blockers"],
                      "candidate_media_hashed": result["candidate_media_hashed"],
                      "release_path": result.get("release_path", "")}, indent=2), flush=True)
    return 0 if result["status"] == "PASS_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
