"""Parent-aware partitions, explicit identity limitations, and immutable-source manifests."""
from collections import Counter, defaultdict, deque
import hashlib
import random
import re
from pathlib import Path
from deepfake_detection.common.paths import Workspace, read_json, read_jsonl, write_json, write_jsonl, digest_file, fingerprint

METHODS = ("Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures")


def stable_key(text, seed=42):
    return hashlib.sha256(f"{seed}:{text}".encode()).hexdigest()


def dependency_families(rows):
    roots = {}
    def find(x):
        roots.setdefault(x, x)
        if roots[x] != x:
            roots[x] = find(roots[x])
        return roots[x]
    tokens = []
    for row in rows:
        group = ["parent:" + p for p in row.get("parents", [])] + ["person:" + p for p in row.get("known_person_ids", [])]
        group = group or ["sample:" + row["sample_id"]]
        for token in group[1:]:
            left, right = find(group[0]), find(token)
            roots[max(left, right)] = min(left, right)
        tokens.append(group[0])
    return [find(token) for token in tokens]


def ffpp_assignments(splits, seed=42):
    assignments = {}
    validation = list(splits["val"])
    random.Random(seed).shuffle(validation)
    midpoint = (len(validation) + 1) // 2
    partitions = {"train": splits["train"], "val": validation[:midpoint],
                  "calibration": validation[midpoint:], "test": splits["test"]}
    for split, pairs in partitions.items():
        for pair in pairs:
            if len(pair) != 2:
                raise ValueError("Official FF++ entries must be two-parent pairs")
            for parent in pair:
                parent = str(parent).zfill(3)
                if parent in assignments and assignments[parent] != split:
                    raise ValueError(f"Official split overlaps at parent {parent}")
                assignments[parent] = split
    return assignments


def stratified_limit(rows, limit, seed):
    if limit is None:
        return list(rows)
    if limit < 1:
        raise ValueError("Limit must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["label"])].append(row)
    queues = [deque(sorted(groups[key], key=lambda r: stable_key(r["sample_id"], seed))) for key in sorted(groups)]
    result = []
    while queues and len(result) < limit:
        remaining = []
        for queue in queues:
            if len(result) < limit:
                result.append(queue.popleft())
            if queue:
                remaining.append(queue)
        queues = remaining
    return result


def check_splits(rows):
    parents, people, hashes, ids = defaultdict(set), defaultdict(set), defaultdict(set), Counter()
    counts = Counter()
    invalid = []
    verified = 0
    for row in rows:
        split = row.get("split")
        if split not in {"train", "val", "calibration", "test", "external"} or row.get("label") not in {0, 1}:
            invalid.append(row.get("sample_id", "unknown"))
        if row.get("dataset") == "dfd" and split != "external":
            invalid.append(row.get("sample_id", "dfd_not_external"))
        ids[row.get("sample_id")] += 1
        counts[(row.get("dataset", "unknown"), split, row.get("label"))] += 1
        if not row.get("parents"):
            invalid.append(row.get("sample_id", "missing_parents"))
        for parent in row.get("parents", []):
            parents[parent].add(split)
        for person in row.get("known_person_ids", []):
            people[person].add(split)
        if row.get("sha256"):
            hashes[row["sha256"]].add(split)
        if row.get("crop_sha256"):
            hashes["crop:" + row["crop_sha256"]].add(split)
        verified += bool(row.get("identity_verified"))
    overlap = lambda values: sorted(k for k, partitions in values.items() if len(partitions) > 1)
    report = {"parent_overlap": overlap(parents), "known_person_overlap": overlap(people),
              "duplicate_content_overlap": overlap(hashes), "duplicate_sample_ids": sorted(k for k, n in ids.items() if n > 1),
              "invalid_records": sorted(set(invalid)), "records": len(rows),
              "counts": [{"dataset": d, "split": s, "label": c, "count": n} for (d, s, c), n in sorted(counts.items())],
              "identity_verified_records": verified, "person_disjoint_verified": bool(rows) and verified == len(rows),
              "limitation": "Person identity is unverified wherever reviewed registry coverage is absent. Parent-disjointness is not person-disjointness."}
    report["ok"] = not any(report[k] for k in ("parent_overlap", "known_person_overlap", "duplicate_content_overlap", "duplicate_sample_ids", "invalid_records"))
    report.update(identity_fields(rows,report["known_person_overlap"]))
    report["person_disjoint_verified"]=bool(rows) and report["person_unverified_records"]==0 and report["ok"]
    return report


def identity_fields(rows,overlap=()):
    groups=defaultdict(list)
    for row,family in zip(rows,dependency_families(rows)): groups[family].append(row)
    verified_families=[];unverified_families=[];details=[]
    verified_records=0
    for family,members in sorted(groups.items()):
        ids=sorted({person for row in members for person in row.get("known_person_ids",[])})
        clean=lambda row: bool(row.get("identity_verified") and row.get("known_person_ids") and not set(row["known_person_ids"]).intersection(overlap))
        verified_records+=sum(clean(row) for row in members)
        verified=all(clean(row) for row in members)
        record={"family_token":family,"verified_person_ids":ids}
        (verified_families if verified else unverified_families).append(record)
        reason="" if verified else "verified person overlaps partitions" if set(ids).intersection(overlap) else "incomplete reviewed person coverage"
        details.append({**record,"split":",".join(sorted({row["split"] for row in members})),"n_records":len(members),
                        "unverified_reason":reason,"recommended_action":"retain reviewed mapping" if verified else "review all parents/aliases; quarantine or repartition conflicts"})
    return {"person_disjoint_verified_records":verified_records,"person_unverified_records":len(rows)-verified_records,
            "person_verified_families":verified_families,"person_unverified_families":unverified_families,
            "verified_person_overlap":list(overlap),"identity_family_rows":details}


def admit(rows, parent_splits, *, policy="source-parent", prior_rows=()):
    rejected, candidates = [], []
    for original in rows:
        row = dict(original)
        mapped = [parent_splits.get(p) for p in row.get("parents", [])]
        reason = None
        if not mapped or any(s is None for s in mapped):
            reason = "unresolved_parent"
        elif len(set(mapped)) != 1:
            reason = "cross_partition_parents"
        elif policy in {"strict-person","require-verified"} and not row.get("identity_verified"):
            reason = "unverified_person_identity" if policy=="require-verified" else "unverified_person"
        if reason:
            rejected.append({**row, "reason": reason})
        else:
            row["split"] = mapped[0]
            candidates.append(row)
    conflicts = check_splits([*prior_rows, *candidates])
    bad_people = set(conflicts["known_person_overlap"])
    bad_hashes = set(conflicts["duplicate_content_overlap"])
    admitted = []
    for row in candidates:
        if bad_people.intersection(row.get("known_person_ids", [])):
            rejected.append({**row, "reason": "known_person_cross_partition"})
        elif row.get("sha256") in bad_hashes:
            rejected.append({**row, "reason": "duplicate_content_cross_partition"})
        else:
            admitted.append(row)
    return admitted, rejected


def videos(folder):
    return sorted((p for p in Path(folder).glob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"}), key=lambda p: p.name)


def _row(dataset, method, path, label, parents, registry):
    known, verified = [], True
    identities = registry.get("identities", {})
    for parent in parents:
        identity = identities.get(parent)
        if not isinstance(identity, dict) or not identity.get("verified") or not identity.get("person_id"):
            verified = False
        else:
            known.append(str(identity["person_id"]))
    sample_id = f"{dataset}:{method}:{path.stem}"
    subject = registry.get("subjects", {}).get(sample_id, {})
    return {"sample_id": sample_id, "dataset": dataset, "method": method,
            "permission_record": registry.get("permissions", {}).get(dataset, "User-supplied local research data; retain accepted terms separately"),
            "subject_verified": label == 1 or bool(subject.get("verified") and subject.get("evidence")),
            "subject_evidence": subject.get("evidence"),
            "verification_source": "manual_registry" if subject.get("verified") and subject.get("evidence") else None,
            "raw_path": str(path.absolute()), "label": label, "parents": parents,
            "known_person_ids": sorted(set(known)), "identity_verified": verified,
            "cluster": "|".join(sorted(parents)), "split": "external" if dataset == "dfd" else "unassigned"}


def _ffpp(cfg, registry):
    root = Path(cfg["raw_root"])
    splits = {s: read_json(Path(cfg["official_splits"]) / (s + ".json")) for s in ("train", "val", "test")}
    mapping = ffpp_assignments(splits, cfg["seed"])
    originals = videos(root / "original_sequences/youtube/c23/videos")
    available = {p.stem for p in originals}
    parent_splits = {f"ffpp:{p}": s for p, s in mapping.items() if p in available}
    rows = [_row("ffpp", "original", p, 1, [f"ffpp:{p.stem}"], registry) for p in originals]
    for method in METHODS:
        for path in videos(root / f"manipulated_sequences/{method}/c23/videos"):
            tokens = path.stem.split("_")
            parents = [f"ffpp:{token}" for token in tokens] if len(tokens) == 2 and all(t.isdigit() for t in tokens) else []
            rows.append(_row("ffpp", method, path, 0, parents, registry))
    return rows, parent_splits


def _celebdf(cfg, registry):
    root = Path(cfg["raw_root"]) / "Celeb-DF-v2"
    originals = videos(root / "Celeb-real")
    ids = sorted({int(re.match(r"id(\d+)_", p.stem).group(1)) for p in originals if re.match(r"id(\d+)_", p.stem)})
    random.Random(cfg["seed"]).shuffle(ids)
    train_n = int(len(ids) * 0.70)
    val_n = int(len(ids) * 0.15 + 0.5)
    val_ids = ids[train_n:train_n + val_n]
    midpoint = (len(val_ids) + 1) // 2
    parts = {"train": ids[:train_n], "val": val_ids[:midpoint], "calibration": val_ids[midpoint:], "test": ids[train_n + val_n:]}
    parent_splits = {f"celebdf:id{i}": split for split, values in parts.items() for i in values}
    rows = []
    for folder, label, method in (("Celeb-real", 1, "original"), ("Celeb-synthesis", 0, "Celeb-DF-v2")):
        for path in videos(root / folder):
            found = re.findall(r"id(\d+)", path.stem)
            required = 1 if label == 1 else 2
            parents = [f"celebdf:id{int(i)}" for i in found[:required]] if len(found) >= required else []
            rows.append(_row("celebdf", method, path, label, parents, registry))
    return rows, parent_splits


def _dfd(cfg, registry):
    root = Path(cfg["raw_root"])
    real = videos(root / "original_sequences/actors/c23/videos")
    available = {str(int(m.group(1))) for p in real if (m := re.match(r"(\d+)__", p.stem))}
    parent_splits = {f"dfd:actor:{i}": "external" for i in available}
    rows = []
    for paths, label, method in ((real, 1, "original"), (videos(root / "manipulated_sequences/DeepFakeDetection/c23/videos"), 0, "DFD")):
        for path in paths:
            match = re.match(r"(\d+)__", path.stem) if label == 1 else re.match(r"(\d+)_(\d+)__", path.stem)
            parents = [f"dfd:actor:{int(t)}" for t in match.groups()] if match else []
            row = _row("dfd", method, path, label, parents, registry)
            row["cluster"] = parents[0] if parents else "unknown"
            rows.append(row)
    selected = []
    for label, cap in ((1, cfg["extraction"]["dfd_real_clips"]), (0, cfg["extraction"]["dfd_fake_clips"])):
        groups = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                groups[row["cluster"]].append(row)
        queues = [deque(sorted(groups[k], key=lambda r: stable_key(r["sample_id"], cfg["seed"]))) for k in sorted(groups)]
        count = 0
        while queues and count < cap:
            remaining = []
            for queue in queues:
                if count < cap:
                    selected.append(queue.popleft())
                    count += 1
                if queue:
                    remaining.append(queue)
            queues = remaining
    return selected, parent_splits


def prepare(dataset, cfg, workspace=None, *, limit=None, name=None, overwrite=False,identity_map=None,require_verified_identities=False):
    ws = workspace or Workspace()
    registry = read_json(cfg["identity_registry"]) if cfg.get("identity_registry") else {}
    builders = {"ffpp": _ffpp, "celebdf": _celebdf, "dfd": _dfd}
    if dataset not in builders:
        raise ValueError("Dataset must be ffpp, celebdf or dfd")
    target = ws.manifest(name or dataset)
    if (target / "videos.jsonl").exists() and not overwrite:
        raise FileExistsError(f"Manifest exists; use explicit --overwrite: {target}")
    rows, parent_splits = builders[dataset](cfg, registry)
    map_hash=None
    if identity_map is not None:
        identity_map=Path(identity_map)
        if not identity_map.is_absolute(): identity_map=ws.root/identity_map
        mapping=read_json(identity_map).get("person_id_map")
        if not isinstance(mapping,dict) or any(not isinstance(key,str) or not isinstance(value,str) or not value.strip() for key,value in mapping.items()):
            raise ValueError("Identity map requires nonempty string parent tokens and verified person IDs")
        map_hash=digest_file(identity_map)
        for row in rows:
            coverage={p:mapping[p] for p in row["parents"] if p in mapping}
            row["known_person_ids"]=sorted(set(row.get("known_person_ids",[]))|set(coverage.values()))
            row["parent_person_ids"]=coverage
            row["identity_verified"]=bool(row["parents"]) and (len(coverage)==len(row["parents"]) or row.get("identity_verified",False))
    if not rows:
        raise ValueError(f"No raw videos found for {dataset}")
    # Assign before stratifying so a small pilot retains class/partition diversity.
    for row in rows:
        row["split"] = parent_splits.get(row["parents"][0], "unassigned") if row["parents"] else "unassigned"
    rows = stratified_limit(rows, limit, cfg["seed"])
    for row in rows:
        row.update(pilot=limit is not None, pilot_limit=limit)
        row["sha256"] = digest_file(row["raw_path"])
    prior = []
    manifests = ws.runs / "manifests"
    if manifests.exists():
        for path in manifests.glob("*/videos.jsonl"):
            if path.parent != target and not path.parent.name.endswith("-pilot"):
                existing = read_jsonl(path)
                prior.extend(r for r in existing if r.get("dataset") != dataset)
    admitted, quarantine = admit(rows, parent_splits, policy="require-verified" if require_verified_identities else cfg["identity_policy"], prior_rows=prior)
    report = check_splits([*prior, *admitted])
    report.update({"dataset": dataset, "identity_policy": cfg["identity_policy"], "seed": cfg["seed"],
                   "identity_map_sha256":map_hash,"require_verified_identities":require_verified_identities,
                   "admitted": len(admitted), "quarantined": len(quarantine), "pilot":limit is not None, "pilot_limit": limit,
                   "manifest_path": str(target / "videos.jsonl"), "manifest_hash": fingerprint(admitted),
                   "excluded_youtube_real": dataset == "celebdf"})
    write_jsonl(target / "videos.jsonl", admitted, ws)
    write_jsonl(target / "quarantine.jsonl", quarantine, ws)
    write_json(target / "split_report.json", report, ws)
    return report


def load_video_manifests(workspace=None, names=None):
    ws = workspace or Workspace()
    if names:
        paths = [ws.manifest(n) / "videos.jsonl" for n in names]
    else:
        paths = sorted((ws.runs / "manifests").glob("*/videos.jsonl"))
    return [row for path in paths for row in read_jsonl(path)]
