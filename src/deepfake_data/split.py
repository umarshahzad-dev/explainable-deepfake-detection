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
        labels_known = all(str(r.get("label", "")) in {"0", "1"} for r in members)
        sample_grouped = bool(group) and all(r.get("group_status") in {"verified", "sample_verified"} for r in members)
        if holdout:
            statuses = {r.get("overlap_status", "unknown") for r in members}
            domain_suspect = any(r.get("source_domain", "").casefold() in ({"cdf", "celeb-real", "youtube-real", "celeb-synthesis"} if holdout == "celebdf" else {"ff"}) for r in members)
            if not sample_grouped or not labels_known or len(datasets) > 1 or statuses & {"verified_overlap", "suspected_overlap"} or any(r.get("metadata_status") in {"conflict", "ambiguous", "partition_conflict"} for r in members):
                partition = "quarantine"
            elif holdout in datasets:
                partition = "test"
            elif domain_suspect or statuses != {"reviewed_independent"}:
                partition = "quarantine"
            elif ratios[0] + ratios[1] == 0:
                partition = "quarantine"
            elif eligible:
                value = int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8], "big") / 2**64
                partition = "train" if value < ratios[0] / (ratios[0] + ratios[1]) else "val"
            else:
                partition = "train" if protocol == "experimental" else "quarantine"
        elif protocol == "experimental":
            # Without cross-dataset review, unresolved training content could overlap any test group.
            partition = "train" if sample_grouped and labels_known else "quarantine"
        elif not eligible:
            partition = "quarantine"
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
        if part in {"train", "val", "calibration", "test"}:
            if row.get("group_id"): groups[row["group_id"]].add(part)
            if row.get("sha256"): hashes[row["sha256"]].add(part)
    return {"group_overlap": {k:sorted(v) for k,v in groups.items() if len(v)>1}, "hash_overlap": {k:sorted(v) for k,v in hashes.items() if len(v)>1}, "counts": [{"partition": k[0], "dataset": k[1], "method": k[2], "label": k[3], "count":v} for k,v in sorted(counts.items())], "limitations": ["Only supplied SHA-256 values are checked; missing hashes do not establish independence", "Group verification is evidence-dependent, not identity certification"]}


def file_sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()

def stream_split(manifest, output, seed=42, ratios=(.7,.15,.15), holdout=None, protocol="strict", reviews=None, hash_media=False):
    """Two-pass split: retain distinct group attributes, never all media rows."""
    from .relationships import apply_reviews
    review_map={}
    if reviews:
        if not holdout: raise ValueError("Reviews require a holdout")
        with Path(reviews).open(newline="",encoding="utf-8") as f:
            raw=list(csv.DictReader(f))
        checked=apply_reviews([{"group_id":r.get("group_id","")} for r in raw],Path(reviews),holdout)
        review_map={r["group_id"]:r.get("overlap_status","unknown") for r in checked}
    distinct={}
    with Path(manifest).open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f);fields=list(reader.fieldnames or [])
        for row in reader:
            group=row.get("group_id","")
            status=review_map.get(group,"unknown") if holdout else row.get("overlap_status","")
            values=(group,row.get("dataset",""),str(row.get("label","")),row.get("group_status",""),row.get("source_domain",""),status,row.get("metadata_status",""))
            if values not in distinct:distinct[values]=dict(zip(("group_id","dataset","label","group_status","source_domain","overlap_status","metadata_status"),values))
    group_datasets=defaultdict(set); group_unknown=set(); group_meta_conflict=set()
    for item in distinct.values():
        group_datasets[item["group_id"]].add(item["dataset"])
        if item["label"] not in {"0","1"}:group_unknown.add(item["group_id"])
        if item.get("metadata_status") in {"conflict","ambiguous","partition_conflict"}: group_meta_conflict.add(item["group_id"])
    assignments=split_rows(list(distinct.values()),seed,ratios,holdout,protocol)
    group_partition={row["group_id"]:row["project_partition"] for row in assignments}
    for field in ("project_partition","overlap_status","quarantine_reason","sha256","validation_status"):
        if field not in fields:fields.append(field)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter();hashes=defaultdict(set);groups=defaultdict(set);per_groups=defaultdict(set)
    with Path(manifest).open(newline="",encoding="utf-8") as source,out.open("w",newline="",encoding="utf-8") as dest:
        writer=csv.DictWriter(dest,fieldnames=fields);writer.writeheader()
        for row in csv.DictReader(source):
            group=row.get("group_id","");part=group_partition.get(group,"quarantine")
            row["project_partition"]=part
            if hash_media:
                digest=hashlib.sha256()
                try:
                    with open(row["absolute_path"],"rb") as media:
                        for chunk in iter(lambda:media.read(1024*1024),b""):digest.update(chunk)
                    row["sha256"]=digest.hexdigest()
                except OSError as exc:
                    row["sha256"]="";row["validation_status"]="hash_error:"+str(exc)
            if holdout:row["overlap_status"]=review_map.get(group,"unknown")
            if part=="quarantine":
                domain=row.get("source_domain","").casefold()
                suspect=holdout and row.get("dataset")!=holdout and domain in ({"cdf","celeb-real","youtube-real","celeb-synthesis"} if holdout=="celebdf" else {"ff"})
                status=row.get("overlap_status","")
                if row.get("label","") not in {"0","1"}: reason="unknown_label"
                elif not group: reason="missing_group"
                elif group in group_meta_conflict and holdout: reason="metadata_conflict"
                elif group in group_unknown: reason="group_has_unknown_label"
                elif len(group_datasets[group])>1: reason="cross_dataset_group"
                elif status in {"verified_overlap","suspected_overlap"}: reason=status
                elif suspect: reason="suspected_source_domain"
                elif holdout: reason="unverified_independence"
                else: reason="unverified_source_group"
                row["quarantine_reason"]=reason
            else:row["quarantine_reason"]=""
            writer.writerow(row);counts[(part,row.get("dataset",""),row.get("method",""),row.get("label",""),row.get("media_type",""))]+=1
            if part in {"train","val","calibration","test"}:
                if group:groups[group].add(part);per_groups[part].add(group)
                if row.get("sha256"):hashes[row["sha256"]].add(part)
    labels={p:sorted({label for (part,_,_,label,_),count in counts.items() if part==p and count}) for p in ("train","val","test")}
    report={"protocol":protocol,"holdout":holdout,"class_coverage":labels,"assigned_group_counts":{p:len(per_groups[p]) for p in ("train","val","test")},"minimum_binary_coverage":all(set(labels[p])=={"0","1"} for p in labels),"group_overlap":{k:sorted(v) for k,v in groups.items() if len(v)>1},"hash_overlap":{k:sorted(v) for k,v in hashes.items() if len(v)>1},"counts":[{"partition":k[0],"dataset":k[1],"method":k[2],"label":k[3],"media_type":k[4],"count":v} for k,v in sorted(counts.items())],"limitations":["Experimental training can include related sources", "Available hashes only", "No identity independence established"]}
    report["source_manifest_sha256"]=file_sha256(manifest)
    report["output_manifest_sha256"]=file_sha256(out)
    out.with_suffix(".report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest");p.add_argument("--output",required=True);p.add_argument("--seed",type=int,default=42);p.add_argument("--ratios",type=float,nargs=3,default=(.7,.15,.15));p.add_argument("--holdout",choices=("celebdf","ffpp"));p.add_argument("--protocol",choices=("strict","experimental"),default="strict");p.add_argument("--reviews");p.add_argument("--hash",action="store_true",help="Hash all media; can be very slow");a=p.parse_args()
    report=stream_split(a.manifest,a.output,a.seed,a.ratios,a.holdout,a.protocol,a.reviews,a.hash)
    if report["group_overlap"] or report["hash_overlap"]:raise SystemExit("Leakage detected; do not use output")

if __name__ == "__main__":main()
