"""Build a blocked, review-only combined allocation; never a training manifest."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from .inventory import ffpp_source_groups
from .split import file_sha256

VERSION="combined-draft-v2"
CELEB_REAL=re.compile(r"(id\d+)_(\d+)")
CELEB_FAKE=re.compile(r"(id\d+)_id\d+_(\d+)")
FF_PAIR=re.compile(r"(\d{3})_(\d{3})")
FF_SINGLE=re.compile(r"(\d{3})")


def celeb_source_key(stem):
    match=CELEB_FAKE.fullmatch(stem) or CELEB_REAL.fullmatch(stem)
    return f"{match[1]}_{match[2]}" if match else None


def celeb_source_keys(stem):
    """Both celebrity IDs are conservative candidate source links."""
    fake=re.fullmatch(r"(id\d+)_(id\d+)_(\d+)",stem)
    if fake:return {f"{fake[1]}_{fake[3]}", f"{fake[2]}_{fake[3]}"}
    key=celeb_source_key(stem)
    return {key} if key else set()

def ff_partition(group, seed):
    value=int.from_bytes(hashlib.sha256(f"{seed}:{group}".encode()).digest()[:8],"big")/2**64
    return "train" if value<.7 else "val" if value<.8 else "calibration" if value<.9 else "test"


def build(manifest, output, report, ffpp_root, seed=42, per_group=10, per_method_label=5000, components=None, components_report=None):
    if per_group<1 or per_method_label<1:raise ValueError("Quotas must be positive")
    source=Path(manifest);output=Path(output);report=Path(report)
    if source.resolve()==output.resolve():raise ValueError("Output must differ from source")
    official_keys=set();official_paths=set(); ff_groups=ffpp_source_groups(Path(ffpp_root));component_map={};anchors=defaultdict(set)
    if bool(components) != bool(components_report):raise ValueError("Both component mapping and report are required")
    if components:
        evidence=json.loads(Path(components_report).read_text(encoding="utf-8"))
        if evidence.get("manifest_sha256") != file_sha256(source) or evidence.get("component_csv_sha256") != file_sha256(components):raise ValueError("Component evidence does not match input manifest or mapping")
        with Path(components).open(newline="",encoding="utf-8") as f:
            for record in csv.DictReader(f):
                group=record.get("group_id","");component=record.get("component_id","")
                if not group or not component or group in component_map:raise ValueError("Invalid or duplicate component mapping")
                component_map[group]=component
    with source.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset")=="celebdf" and row.get("original_partition")=="official_test":
                official_paths.add(row.get("relative_path", "").casefold())
                official_keys.update(celeb_source_keys(row.get("sample_id", "")))
                component=component_map.get(row.get("group_id",""))
                if component:anchors[component].add("test")
            elif row.get("dataset")=="ffpp" and row.get("group_status")=="verified" and row.get("group_id","").startswith("ffpp:youtube:"):
                partition=ff_partition(row["group_id"],seed);component=component_map.get(row["group_id"])
                if component and partition!="train":anchors[component].add(partition)
    # Temporary SQLite retains bounded-memory quota state for candidate images.
    db=sqlite3.connect("");db.execute("PRAGMA cache_size=-8192");db.execute("CREATE TABLE eligible (path TEXT PRIMARY KEY, method TEXT, label TEXT, group_id TEXT, priority TEXT)")
    pending=[]
    def raw_decide(row):
        dataset=row.get("dataset","");label=row.get("label","");group=row.get("group_id","");method=row.get("method","");sample=row.get("sample_id","");domain=row.get("source_domain","").casefold()
        if label not in {"0","1"}:return "quarantine","unknown_label",group
        if not group:return "quarantine","unknown_source_group",group
        if dataset=="ffpp":
            if row.get("group_status")=="verified" and group.startswith("ffpp:youtube:"):return ff_partition(group,seed),"verified_ffpp_source_pair_candidate",group
            if group.startswith("ffpp:dfd:actors:"):
                if group.endswith(":05") or group.endswith(":10"):return "quarantine","dfd_actor_evaluation_candidate_unverified",group
                return "train","dfd_actor_group_train_candidate",group
            return "quarantine","unresolved_ffpp_source",group
        if dataset=="celebdf":
            if row.get("relative_path","").casefold() in official_paths:return "test","official_celebdf_test_candidate_unverified",group
            key=celeb_source_key(sample)
            if celeb_source_keys(sample) & official_keys:return "quarantine","suspected_shared_official_test_source", "celebdf:suspected:"+key
            return "train","cross_dataset_relationship_unreviewed",("celebdf:suspected:"+key if key else group)
        if dataset=="df40":
            if row.get("metadata_status") not in {"matched","partition_conflict"}:return "quarantine","unresolved_df40_metadata",group
            if method.casefold()=="midjourney":return "quarantine","df40_modern_evaluation_candidate_unverified",group
            if domain=="cdf":
                key=celeb_source_key(sample)
                if not key:return "quarantine","unresolved_celebdf_source",group
                if celeb_source_keys(sample) & official_keys:return "quarantine","suspected_shared_official_test_source","celebdf:suspected:"+key
                return "train","repurposed_df40_testing_release_unreviewed","celebdf:suspected:"+key
            if domain=="ff":
                pair=FF_PAIR.fullmatch(sample);single=FF_SINGLE.fullmatch(sample)
                ids=pair.groups() if pair else (single[1],) if single else ()
                if not ids or any(x not in ff_groups for x in ids):return "quarantine","unresolved_ffpp_source",group
                keys={ff_groups[x] for x in ids}
                if len(keys)!=1:return "quarantine","conflicting_ffpp_source_candidates",group
                linked="ffpp:youtube:"+next(iter(keys))
                if ff_partition(linked,seed)!="train":return "quarantine","suspected_ffpp_heldout_overlap",linked
                return "train","repurposed_df40_testing_release_unreviewed",linked
            return "train","unknown_cross_dataset_source",group
        return "quarantine","unrecognized_dataset",group
    def decide(row):
        partition,reason,group=raw_decide(row)
        component=component_map.get(row.get("group_id",""))
        if components and row.get("group_id") and not component:return "quarantine","missing_component_evidence",group
        if component:
            anchored=anchors.get(component,set())
            if partition=="train" and anchored:return "quarantine","suspected_component_reaches_evaluation",component
            if partition in {"val","calibration","test"} and len(anchored)>1:return "quarantine","suspected_component_crosses_evaluation_partitions",component
            return partition,reason,component
        return partition,reason,group
    # First pass determines DF40 media quotas, using stable hash ordering.
    with source.open(newline="",encoding="utf-8") as f:
        fields=list(csv.DictReader(f).fieldnames or [])
        f.seek(0)
        for row in csv.DictReader(f):
            proposed,reason,group=decide(row)
            if row.get("dataset")=="df40" and proposed=="train":
                path=row.get("relative_path","");priority=hashlib.sha256(f"{seed}:{path}".encode()).hexdigest()
                # Quota by original per-method/sample group: shared-source links are separately tracked.
                pending.append((path,row.get("method",""),row.get("label",""),row.get("group_id",""),priority))
                if len(pending)>=5000:db.executemany("INSERT INTO eligible VALUES(?,?,?,?,?)",pending);pending.clear()
    if pending:db.executemany("INSERT INTO eligible VALUES(?,?,?,?,?)",pending)
    db.execute("CREATE INDEX eligible_order ON eligible(method,label,priority)");db.execute("CREATE TABLE selected(path TEXT PRIMARY KEY)")
    per_method=Counter();per_source=Counter(); selected=[]
    for path,method,label,group in db.execute("SELECT path,method,label,group_id FROM eligible ORDER BY method,label,priority"):
        if per_method[(method,label)]>=per_method_label or per_source[(method,label,group)]>=per_group:continue
        per_method[(method,label)]+=1;per_source[(method,label,group)]+=1;selected.append((path,))
        if len(selected)>=5000:db.executemany("INSERT INTO selected VALUES(?)",selected);selected.clear()
    if selected:db.executemany("INSERT INTO selected VALUES(?)",selected)
    db.commit();output.parent.mkdir(parents=True,exist_ok=True)
    for name in ("project_partition","proposed_partition","proposal_reason","proposed_source_group","proposal_status"):
        if name not in fields:fields.append(name)
    counts=Counter(); groups=defaultdict(set); method_groups=defaultdict(set)
    with source.open(newline="",encoding="utf-8") as src,output.open("w",newline="",encoding="utf-8") as dst:
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for row in csv.DictReader(src):
            proposed,reason,group=decide(row)
            if row.get("dataset")=="df40" and proposed=="train" and not db.execute("SELECT 1 FROM selected WHERE path=?",(row.get("relative_path",""),)).fetchone():proposed,reason="quarantine","df40_quota_excluded"
            row.update(project_partition="quarantine",proposed_partition=proposed,proposal_reason=reason,proposed_source_group=group,proposal_status="DRAFT_NOT_APPROVED")
            writer.writerow(row);category=(proposed,row.get("dataset",""),row.get("method",""),row.get("label","") or "unknown",row.get("media_type",""),reason);counts[category]+=1
            if group:
                groups[(proposed,row.get("dataset","") )].add(group)
                method_groups[(proposed,row.get("dataset",""),row.get("method",""),row.get("label","") or "unknown")].add(group)
    db.close()
    coverage=[{"proposed_partition":p,"dataset":d,"method":m,"label":l,"media_type":t,"reason":r,"count":n} for (p,d,m,l,t,r),n in sorted(counts.items())]
    result={"status":"DRAFT_BLOCKED_NOT_FOR_EXTRACTION_OR_TRAINING","config":{"version":VERSION,"seed":seed,"ratios":[.7,.1,.1,.1],"df40_max_media_per_original_sample":per_group,"df40_max_media_per_method_and_label":per_method_label,"official_celebdf_test_repurposed":False,"components_used":bool(components)},"source_manifest_sha256":file_sha256(source),"draft_manifest_sha256":file_sha256(output),"coverage":coverage,"proposed_group_counts":[{"partition":p,"dataset":d,"groups":len(g)} for (p,d),g in sorted(groups.items())],"proposed_method_group_counts":[{"partition":p,"dataset":d,"method":m,"label":l,"groups":len(g)} for (p,d,m,l),g in sorted(method_groups.items())],"blockers":["Cross-dataset source relationships are not sufficiently verified to activate held-out partitions", "Exact-content duplicate coverage is incomplete", "Source grouping from Celeb-DF and DF40 filenames is a conservative suspicion, not verified source identity", "Candidate video frames have not been extracted or integrity validated", "Celeb-DF official test candidates and suspected related sources still require review; official test entries are never proposed for training", "Reserved MidJourney and actor evaluation candidates have unresolved cross-dataset independence"],"policy":"All project_partition values are quarantine. proposed_partition is a non-runnable design for review only. DF40 author testing release is explicitly repurposed only as a proposed training source, never an official benchmark."}
    report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2),encoding="utf-8");return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--report",required=True);p.add_argument("--ffpp-root",default=r"D:\University\datasets\deepfake\ffpp");p.add_argument("--components",type=Path);p.add_argument("--components-report",type=Path);p.add_argument("--seed",type=int,default=42);p.add_argument("--df40-per-group",type=int,default=10);p.add_argument("--df40-per-method-label",type=int,default=5000);a=p.parse_args();r=build(a.manifest,a.output,a.report,a.ffpp_root,a.seed,a.df40_per_group,a.df40_per_method_label,a.components,a.components_report);print(r["status"]);print("Input SHA-256:",r["source_manifest_sha256"]);print("Draft SHA-256:",r["draft_manifest_sha256"])

if __name__=="__main__":main()
