"""Conservative deterministic splits; unverified rows are quarantined."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def split_rows(rows, seed=42, ratios=(0.7, 0.15, 0.15), holdout=None, protocol="strict"):
    if protocol not in {"strict", "experimental"}: raise ValueError("unknown protocol")
    if len(ratios) != 3 or any(x < 0 for x in ratios) or abs(sum(ratios)-1) > 1e-9:
        raise ValueError("ratios must be three nonnegative values summing to one")
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("group_id", "")].append(row)
    result = []
    for group, members in sorted(groups.items()):
        eligible = bool(group) and all(r.get("group_status") == "verified" and str(r.get("label", "")) in {"0", "1"} for r in members)
        datasets = {r.get("dataset") for r in members}
        if not eligible:
            # Experimental training accepts known-label sample groups, never as independent test.
            trainable = protocol == "experimental" and bool(group) and all(r.get("group_status") == "sample_verified" and str(r.get("label", "")) in {"0", "1"} for r in members)
            if holdout and any(r.get("dataset") == holdout or r.get("source_domain", "").casefold() in ({"cdf", "celeb-real", "youtube-real", "celeb-synthesis"} if holdout == "celebdf" else {"ff"}) for r in members): trainable = False
            partition = "train" if trainable and not holdout else "quarantine"
        elif holdout and holdout in datasets:
            partition = "test" if datasets == {holdout} else "quarantine"
        elif holdout:
            # Verification must explicitly cover the selected held-out dataset.
            independent = all(holdout in r.get("verified_independent_of", "").split("|") for r in members)
            if not independent or ratios[0] + ratios[1] == 0: partition = "quarantine"
            else:
                value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
                partition = "train" if value < ratios[0] / (ratios[0] + ratios[1]) else "val" if ratios[0] + ratios[1] else "quarantine"
        else:
            value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
            partition = "train" if value < ratios[0] else "val" if value < ratios[0]+ratios[1] else "test"
        for row in members:
            result.append({**row, "project_partition": partition})
    return result


def validate(rows):
    groups, hashes = defaultdict(set), defaultdict(set)
    counts = Counter()
    for row in rows:
        part = row.get("project_partition", "")
        counts[(part, row.get("dataset", ""), row.get("method", ""), str(row.get("label", "")))] += 1
        if part in {"train", "val", "test"}:
            if row.get("group_id"): groups[row["group_id"]].add(part)
            if row.get("sha256"): hashes[row["sha256"]].add(part)
    return {"group_overlap": {k:sorted(v) for k,v in groups.items() if len(v)>1}, "hash_overlap": {k:sorted(v) for k,v in hashes.items() if len(v)>1}, "counts": [{"partition": k[0], "dataset": k[1], "method": k[2], "label": k[3], "count":v} for k,v in sorted(counts.items())], "limitations": ["Only supplied SHA-256 values are checked; missing hashes do not establish independence", "Group verification is evidence-dependent, not identity certification"]}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("manifest"); p.add_argument("--output", required=True); p.add_argument("--seed", type=int, default=42); p.add_argument("--ratios", type=float, nargs=3, default=(.7,.15,.15)); p.add_argument("--holdout", choices=("celebdf","ffpp")); p.add_argument("--protocol", choices=("strict","experimental"), default="strict"); p.add_argument("--hash", action="store_true", help="Hash all media; can be very slow"); a=p.parse_args()
    with open(a.manifest, newline="", encoding="utf-8") as f:
        reader=csv.DictReader(f); fields=list(reader.fieldnames or []); rows=list(reader)
    for field in ("project_partition", "verified_independent_of"):
        if field not in fields: fields.append(field)
    if a.hash:
        for row in rows:
            digest=hashlib.sha256()
            try:
                with open(row["absolute_path"], "rb") as source:
                    for chunk in iter(lambda: source.read(1024*1024), b""): digest.update(chunk)
                row["sha256"] = digest.hexdigest()
            except OSError as exc:
                row["sha256"] = ""; row["validation_status"] = "hash_error:" + str(exc)
    rows=split_rows(rows, a.seed, a.ratios, a.holdout, a.protocol); report=validate(rows)
    report["protocol"] = a.protocol
    report["holdout"] = a.holdout
    report["limitations"].append("Experimental training can include related source content; no independence is asserted")
    out=Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["group_overlap"] or report["hash_overlap"]: raise SystemExit("Leakage detected; do not use output")

if __name__ == "__main__": main()
