"""Audit review-only combined proposals without declaring an independent test."""
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from .split import file_sha256


def audit(manifest, hash_report=None, hash_csv=None):
    active=Counter();proposed=Counter();groups=defaultdict(set);hashes=defaultdict(set);missing_hash=0;official_violations=0;official_boundary_violations=0;unknown_labels=0;missing_groups=0;quota_groups=Counter();quota_methods=Counter();methods=defaultdict(set)
    with Path(manifest).open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            active[row.get("project_partition","")]+=1
            p=row.get("proposed_partition","");dataset=row.get("dataset","");label=row.get("label","");method=row.get("method","");group=row.get("proposed_source_group","")
            proposed[(p,dataset,label or "unknown")]+=1
            if p in {"train","val","calibration","test"}:
                if label not in {"0","1"}:unknown_labels+=1
                if group:groups[group].add(p)
                else:missing_groups+=1
                digest=row.get("sha256","")
                if digest:hashes[digest].add(p)
                else:missing_hash+=1
                methods[(p,dataset)].add(method)
                if row.get("original_partition")=="official_test" and p=="train":official_violations+=1
                if row.get("dataset")=="celebdf" and row.get("original_partition")=="official_test" and p!="test":official_boundary_violations+=1
                if dataset=="df40" and p=="train":
                    quota_groups[(method,label,row.get("group_id",""))]+=1;quota_methods[(method,label)]+=1
    group_overlap={g:sorted(v) for g,v in groups.items() if len(v)>1};hash_overlap={h:sorted(v) for h,v in hashes.items() if len(v)>1}
    manifest_hash=file_sha256(manifest);external_hash_evidence={"provided":False,"chain_valid":False,"complete":False,"cross_partition_exact_hash_overlaps":{}}
    if hash_report and hash_csv:
        data=json.loads(Path(hash_report).read_text(encoding="utf-8"))
        valid=data.get("proposal_mode") is True and data.get("manifest_sha256")==manifest_hash and data.get("hash_csv_sha256")==file_sha256(hash_csv)
        external_hash_evidence={"provided":True,"chain_valid":valid,"complete":bool(valid and data.get("complete")),"attempted":data.get("attempted",0) if valid else 0,"eligible_records":data.get("eligible_records",0) if valid else 0,"cross_partition_exact_hash_overlaps":data.get("cross_partition_exact_hash_overlaps",{}) if valid else {}}
    return {"status":"BLOCKED_DRAFT", "manifest_sha256":manifest_hash,"external_hash_evidence":external_hash_evidence,"active_partitions":dict(active),"proposed_counts":[{"partition":p,"dataset":d,"label":l,"count":n} for (p,d,l),n in sorted(proposed.items())],"proposed_methods":[{"partition":p,"dataset":d,"methods":sorted(v)} for (p,d),v in sorted(methods.items())],"group_overlap":group_overlap,"hash_overlap":hash_overlap,"missing_hashes":missing_hash,"official_test_train_violations":official_violations,"official_test_boundary_violations":official_boundary_violations,"unknown_label_active_candidates":unknown_labels,"missing_group_active_candidates":missing_groups,"df40_max_per_original_group":max(quota_groups.values(),default=0),"df40_max_per_method_label":max(quota_methods.values(),default=0),"blockers":["All project assignments remain quarantined", "Cross-dataset source independence and content duplicates are unverified", "Proposed evaluation groups are not an established independent test", "No processed full dataset exists"]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--report',required=True,type=Path);p.add_argument('--hash-report',type=Path);p.add_argument('--hash-csv',type=Path);a=p.parse_args();r=audit(a.manifest,a.hash_report,a.hash_csv);a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps({k:r[k] for k in ('status','active_partitions','missing_hashes','official_test_train_violations','unknown_label_active_candidates','df40_max_per_original_group','df40_max_per_method_label')},indent=2))

if __name__=='__main__':main()
