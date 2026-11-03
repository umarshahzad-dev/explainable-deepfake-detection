"""Validate an image manifest; lack of complete evidence is a blocker."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

ACTIVE={"train","val","calibration","test"}


def validate(path, decode=False, hash_outputs=False):
    counts=Counter();classes=defaultdict(set);groups=defaultdict(set);hashes=defaultdict(set);paths=set();rows_seen=set();issues=Counter();examples=[];datasets=set()
    from PIL import Image
    with Path(path).open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            part=row.get("project_partition","")
            if part not in ACTIVE:
                issues["nonactive_rows_in_manifest"]+=1;continue
            if row.get("proposal_status")=="DRAFT_NOT_APPROVED":issues["draft_not_approved"]+=1
            source=row.get("absolute_path","");target=row.get("input_path","");group=row.get("group_id","");label=row.get("label","");dataset=row.get("dataset","");method=row.get("method","");frame=str(row.get("source_frame_index",""))
            datasets.add(dataset);counts[(part,dataset,method,label)]+=1;classes[part].add(label)
            key=(dataset,source,frame)
            if key in rows_seen:issues["duplicate_manifest_rows"]+=1
            rows_seen.add(key)
            if target in paths:issues["output_path_collision"]+=1
            paths.add(target)
            if label not in {"0","1"}:issues["invalid_labels"]+=1
            if not group:issues["missing_groups"]+=1
            else:groups[group].add(part)
            if not source or not Path(source).is_file():issues["missing_sources"]+=1
            if not target or not Path(target).is_file():issues["missing_outputs"]+=1;continue
            if decode:
                try:
                    with Image.open(target) as img:
                        img.load()
                        if img.size!=(299,299):issues["wrong_dimensions"]+=1
                except (OSError,ValueError):issues["decode_errors"]+=1
            if hash_outputs:
                digest=hashlib.sha256()
                try:
                    with open(target,'rb') as file:
                        for chunk in iter(lambda:file.read(1048576),b''):digest.update(chunk)
                    hashes[digest.hexdigest()].add(part)
                except OSError:issues["hash_errors"]+=1
    group_overlap={k:sorted(v) for k,v in groups.items() if len(v)>1};hash_overlap={k:sorted(v) for k,v in hashes.items() if len(v)>1}
    active=set(classes);required={"train","val","test"};coverage=required<=active and all(classes[p]=={"0","1"} for p in active)
    if not coverage:issues["partition_class_coverage"]+=1
    if not decode:issues["decoding_not_performed"]+=1
    if not hash_outputs:issues["full_output_hashing_not_performed"]+=1
    if group_overlap:issues["group_leakage"]+=len(group_overlap)
    if hash_overlap:issues["exact_content_leakage"]+=len(hash_overlap)
    # Image hashes and file metadata alone cannot prove cross-dataset source independence.
    issues["cross_dataset_source_independence_unverified"]+=1
    return {"ready":False,"issues":dict(issues),"active_partitions":sorted(active),"datasets":sorted(datasets),"counts":[{"partition":p,"dataset":d,"method":m,"label":l,"count":n} for (p,d,m,l),n in sorted(counts.items())],"group_overlap":group_overlap,"hash_overlap":hash_overlap,"limitations":["Cross-dataset relationship review and source-level duplicate evidence are required", "Output pixel hashes do not prove identity independence", "Compare expected source/split and zero-yield reports before use"]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True,type=Path);p.add_argument("--report",required=True,type=Path);p.add_argument("--decode",action="store_true");p.add_argument("--hash-outputs",action="store_true");a=p.parse_args()
    result=validate(a.manifest,a.decode,a.hash_outputs);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2),encoding="utf-8")
    lines=["# Prepared dataset validation","",f"Ready: {result['ready']}"]+[f"- BLOCKER: {k}: {v}" for k,v in sorted(result['issues'].items())]+[f"- WARNING: {x}" for x in result['limitations']]
    a.report.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print('\n'.join(lines))
    if not result['ready']:raise SystemExit(2)

if __name__=='__main__':main()
