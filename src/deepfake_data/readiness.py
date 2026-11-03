"""Conservative, evidence-based extraction and evaluation readiness report."""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path


def audit_manifest(path, sample_paths=()):
    counts=Counter(); groups=defaultdict(set); hashes=defaultdict(set); per_partition_groups=defaultdict(set); missing=Counter(); overlaps=Counter(); samples=[]; sample_records={}
    wanted={str(x).replace("\\","/").casefold() for x in sample_paths}
    with Path(path).open(newline="",encoding="utf-8") as stream:
        reader=csv.DictReader(stream); fields=reader.fieldnames or []
        for row in reader:
            part=row.get("project_partition") or "unassigned"; dataset=row.get("dataset",""); label=str(row.get("label","")); media=row.get("media_type",""); group=row.get("group_id",""); status=row.get("overlap_status","") or row.get("relationship_status","") or "unknown"
            sample_key=row.get("absolute_path","").replace("\\","/").casefold()
            if sample_key in wanted: sample_records[sample_key]={"label":label,"metadata_status":row.get("metadata_status", ""),"dataset":dataset}
            for dim,value in (("partition",part),("dataset",dataset),("method",row.get("method","") or "unknown"),("media_type",media),("label",label or "unknown"),("source_domain",row.get("source_domain","") or "unknown"),("metadata_status",row.get("metadata_status","") or "unknown"),("relationship",status),("dataset_label",dataset+":"+(label or "unknown"))):
                counts[(part,dim,value)]+=1
            if group: per_partition_groups[part].add(group)
            if part in {"train","val","calibration","test"}:
                if group: groups[group].add(part)
                else: missing["assigned_missing_group"]+=1
                if label not in {"0","1"}: missing["assigned_unknown_label"]+=1
                digest=row.get("sha256","")
                if digest: hashes[digest].add(part)
                else: missing["assigned_without_hash"]+=1
                if status in {"unknown","suspected","suspected_overlap","verified_overlap","actor_links_verified_cross_dataset_unknown"}: overlaps[status]+=1
            if part=="quarantine":
                reason=row.get("quarantine_reason") or ("unknown_label" if label not in {"0","1"} else "missing_group" if not group else status)
                missing["quarantine:"+reason]+=1
                if len(samples)<20: samples.append({"path":row.get("relative_path",""),"reason":reason})
    return {"fields":fields,"counts":[{"partition":p,"dimension":d,"value":v,"count":n} for (p,d,v),n in sorted(counts.items())],"partition_groups":{p:len(per_partition_groups[p]) for p in ("train","val","calibration","test","quarantine")},"group_overlap":{g:sorted(p) for g,p in groups.items() if len(p)>1},"hash_overlap":{h:sorted(p) for h,p in hashes.items() if len(p)>1},"issues":dict(missing),"assigned_relationships":dict(overlaps),"quarantine_examples":samples,"sample_records":sample_records}


def sample_audit(root):
    result={"videos":0,"images":0,"labels":{},"errors":[],"zero_yield":0,"video_yields":[],"configurations":[]}
    if not root or not Path(root).is_dir(): result["errors"].append("Extraction sample directory unavailable");return result
    try: from PIL import Image
    except ImportError: result["errors"].append("Pillow unavailable for decoding crop samples");return result
    configs=set()
    for path in sorted(Path(root).glob("*/complete.json")):
        try:
            data=json.loads(path.read_text(encoding="utf-8")); config=data["config"]; params=config["parameters"]; prov=config.get("provenance",{});frames=data["frames"]
            result["videos"]+=1;result["zero_yield"]+=not bool(frames);result["video_yields"].append(len(frames));configs.add(json.dumps(params,sort_keys=True))
            key=prov.get("dataset","unknown")+":"+prov.get("label","unknown");result["labels"][key]=result["labels"].get(key,0)+1
            runtime=config.get("runtime",{})
            if config.get("version")!=2 or params.get("size")!=299 or params.get("margin")!=1.5 or params.get("crop_mode")!="square" or params.get("threshold",0)<.95 or params.get("min_face_size",0)<80 or not runtime.get("detector","").endswith(".MTCNN") or runtime.get("device")!="cuda":
                result["errors"].append(f"{path}: unexpected extraction configuration or detector")
            segments=[frame.get("segment") for frame in frames]
            if len(segments)!=len(set(segments)):result["errors"].append(f"{path}: more than one selected frame in a segment")
            if not Path(config["video"]).is_file():result["errors"].append(f"{path}: source video missing")
            for frame in frames:
                target=path.parent/frame["file"]
                with Image.open(target) as img:
                    img.load()
                    if img.size!=(299,299):result["errors"].append(f"{target}: wrong dimensions")
                if (frame.get("confidence",0) < params.get("threshold",.95) or frame.get("reason")!="eligible" or frame.get("face_width",0)<params.get("min_face_size",80) or frame.get("face_height",0)<params.get("min_face_size",80) or frame.get("padding_fraction",1)>params.get("max_padding",.35) or frame.get("sharpness",-1)<params.get("min_sharpness",30)):
                    result["errors"].append(f"{target}: invalid selection metadata")
                result["images"]+=1
        except (OSError,ValueError,KeyError,TypeError) as exc:result["errors"].append(f"{path}: {exc}")
    result["configurations"]=[json.loads(x) for x in sorted(configs)]
    if result["videos"]==0:result["errors"].append("No completed sample videos")
    return result


def df40_sample_audit(rows, records):
    result={"decoded":0,"verified_labels":Counter(),"unknown_labels":0,"partition_conflicts":0,"errors":[]}
    try: from PIL import Image
    except ImportError: result["errors"].append("Pillow unavailable for DF40 sample decoding");return result
    for row in rows:
        path=row.get("path",""); key=path.replace("\\","/").casefold(); reference=records.get(key)
        try:
            with Image.open(path) as image:image.load()
            result["decoded"]+=1
        except (OSError,ValueError) as exc:result["errors"].append(f"{path}: {exc}");continue
        candidate=row.get("label_from_json")
        if candidate is None:result["unknown_labels"]+=1;continue
        if reference is None:result["errors"].append(f"{path}: absent from split manifest")
        elif reference.get("metadata_status") not in {"matched","partition_conflict"} or reference.get("label")!=str(candidate):result["errors"].append(f"{path}: reconciled label or metadata status differs")
        else:
            result["verified_labels"][str(candidate)]+=1
            if reference.get("metadata_status")=="partition_conflict":result["partition_conflicts"]+=1
    result["verified_labels"]=dict(result["verified_labels"]);return result

def assess(manifest, inventory_report=None, quality_dir=None, output_dir=None, minimum_free_gb=10, batch_videos=0, frames_per_video=20, split_report=None, df40_samples=None, source_links_report=None):
    df_rows=json.loads(Path(df40_samples).read_text(encoding="utf-8")) if df40_samples and Path(df40_samples).is_file() else []
    summary=audit_manifest(manifest,[row.get("path", "") for row in df_rows]); sample=sample_audit(quality_dir); df_sample=df40_sample_audit(df_rows, summary["sample_records"]); checks=[]
    def add(name,status,reason):checks.append({"check":name,"status":status,"reason":reason})
    if inventory_report and Path(inventory_report).is_file():
        inventory=json.loads(Path(inventory_report).read_text(encoding="utf-8"));full=not inventory.get("scan_limited",True) and not inventory.get("errors")
    else: inventory={};full=False
    total_manifest=sum(r["count"] for r in summary["counts"] if r["dimension"]=="partition")
    total_inventory=sum(v for k,v in inventory.get("counts",{}).items() if k.startswith("indexed:"))
    from .split import file_sha256
    split_evidence=json.loads(Path(split_report).read_text(encoding="utf-8")) if split_report and Path(split_report).is_file() else {}
    link_evidence=json.loads(Path(source_links_report).read_text(encoding="utf-8")) if source_links_report and Path(source_links_report).is_file() else {}
    source_hash=split_evidence.get("source_manifest_sha256"); inventory_hash=inventory.get("manifest_sha256")
    direct=bool(inventory_hash) and source_hash==inventory_hash
    linked=bool(inventory_hash) and link_evidence.get("source_manifest_sha256")==inventory_hash and bool(link_evidence.get("output_manifest_sha256")) and source_hash==link_evidence.get("output_manifest_sha256")
    chain_valid=(direct or linked) and split_evidence.get("output_manifest_sha256")==file_sha256(manifest)
    complete_manifest=full and total_inventory>0 and total_manifest==total_inventory and chain_valid
    add("full_inventory", "PASS" if complete_manifest else "FAIL", f"Inventory rows={total_inventory}; split rows={total_manifest}; full scan={full}; SHA-256 provenance chain verified={chain_valid}.")
    add("sample_integrity", "PASS" if not sample["errors"] and sample["images"] else "FAIL", f"{sample['images']} sample crops decoded; {len(sample['errors'])} sample errors")
    required_categories={r["value"] for r in summary["counts"] if r["dimension"]=="dataset_label" and r["partition"]=="train" and r["value"].split(":",1)[0] in {"ffpp","celebdf"} and r["value"].split(":",1)[1] in {"0","1"}}
    observed_categories=set(sample["labels"])
    representative=bool(required_categories) and required_categories <= observed_categories
    add("sample_dataset_coverage", "PASS" if representative else "FAIL", f"Required training dataset/label categories={sorted(required_categories)}; observed crop categories={sorted(observed_categories)}")
    df40_training=any(r["partition"]=="train" and r["dimension"]=="dataset" and r["value"]=="df40" and r["count"] for r in summary["counts"])
    df40_ok=not df_sample["errors"] and not df_sample["partition_conflicts"] and set(df_sample["verified_labels"])=={"0","1"}
    add("df40_sample_integrity", "PASS" if df40_ok else "FAIL" if df40_training else "WARNING", f"Decoded {df_sample['decoded']} DF40 sample images; reconciled labels={df_sample['verified_labels']}; errors={len(df_sample['errors'])}; partition conflicts={df_sample['partition_conflicts']}; unknown labels={df_sample['unknown_labels']}")
    add("sample_scope", "WARNING", "Finite sample; identity continuity, DF40 coverage and all-media decoding are not established")
    leakage=bool(summary["group_overlap"] or summary["hash_overlap"])
    add("known_partition_leakage", "FAIL" if leakage else "PASS", "Source groups or available hashes cross partitions" if leakage else "No cross-partition groups or supplied hashes detected")
    missing=summary["issues"]
    add("assigned_labels_groups", "FAIL" if missing.get("assigned_missing_group") or missing.get("assigned_unknown_label") else "PASS", "Assigned records need known labels and source groups" if missing.get("assigned_missing_group") or missing.get("assigned_unknown_label") else "Assigned records have labels and groups")
    counts=summary["counts"]
    active=("train","val","calibration","test") if any(r["partition"]=="calibration" for r in counts) else ("train","val","test")
    labels={part:{r["value"] for r in counts if r["partition"]==part and r["dimension"]=="label" and r["count"]} for part in active}
    coverage=all(labels[p]=={"0","1"} for p in labels)
    add("partition_binary_coverage", "PASS" if coverage else "FAIL", "Both labels required in train, validation and test: "+str(labels))
    unverified=sum(summary["assigned_relationships"].values()); nohash=missing.get("assigned_without_hash",0)
    add("independence", "FAIL" if unverified else "WARNING", f"{unverified} assigned rows have unknown or suspected relationships; absence of flags does not establish identity independence")
    add("exact_hash_coverage", "WARNING" if nohash else "PASS", f"{nohash} assigned rows lack exact hashes; no overlap claim extends to unhashed files")
    target=Path(output_dir or Path(manifest).parent).resolve()
    while not target.exists() and target != target.parent: target=target.parent
    free=shutil.disk_usage(target).free
    estimated = int(batch_videos * frames_per_video * (299*299*3*1.5 + 4096) + 100*1024**2)
    disk_ok = batch_videos > 0 and frames_per_video > 0 and free >= max(minimum_free_gb*1024**3, estimated)
    add("disk_headroom", "PASS" if disk_ok else "FAIL", f"{free/1024**3:.1f} GiB free; batch={batch_videos} videos, up to {frames_per_video} frames/video; approximate allowance {estimated/1024**3:.2f} GiB plus {minimum_free_gb:g} GiB minimum. Specify a positive batch size.")
    train_count=sum(r["count"] for r in counts if r["partition"]=="train" and r["dimension"]=="media_type" and r["value"]=="video")
    add("assigned_training_videos", "PASS" if train_count else "FAIL", f"{train_count} assigned training videos")
    extract_required={"full_inventory","sample_integrity","sample_dataset_coverage","assigned_labels_groups","disk_headroom","assigned_training_videos"}
    if df40_training: extract_required.add("df40_sample_integrity")
    extract_ok=all(c["status"]=="PASS" for c in checks if c["check"] in extract_required)
    train_required=extract_required|{"known_partition_leakage","partition_binary_coverage","independence"}
    # Independence without any reviewed evidence is not a pass.
    train_ok=all(c["status"]=="PASS" for c in checks if c["check"] in train_required) and nohash==0
    reasons=[c["reason"] for c in checks if c["check"] in extract_required and c["status"]!="PASS"]
    return {"extraction_ready":extract_ok,"training_evaluation_ready":train_ok,"overnight_extraction":"Not authorized by this report; "+("technical prerequisites passed, user approval still required" if extract_ok else "blocked: "+"; ".join(reasons)),"checks":checks,"inventory":inventory,"manifest":summary,"sample":sample,"df40_sample":df_sample,"limitations":["PASS checks are limited to supplied evidence", "Known-group and available-hash checks do not prove person-identity independence", "Disk threshold is not a capacity estimate", "Review crop quality and DF40 labels separately", "Video frame indices rely on codec seek behavior; variable-frame-rate media require verification"]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True,type=Path);p.add_argument("--inventory-report",type=Path);p.add_argument("--quality-dir",type=Path);p.add_argument("--output-dir",type=Path);p.add_argument("--minimum-free-gb",type=float,default=10);p.add_argument("--split-report",type=Path);p.add_argument("--source-links-report",type=Path);p.add_argument("--df40-samples",type=Path);p.add_argument("--batch-videos",type=int,default=0);p.add_argument("--frames-per-video",type=int,default=20);p.add_argument("--report",required=True,type=Path);a=p.parse_args()
    if a.minimum_free_gb <= 0:p.error("minimum-free-gb must be positive")
    report=assess(a.manifest,a.inventory_report,a.quality_dir,a.output_dir,a.minimum_free_gb,a.batch_videos,a.frames_per_video,a.split_report,a.df40_samples,a.source_links_report)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2),encoding="utf-8")
    lines=["# Dataset readiness", "",f"Extraction: {'PASS' if report['extraction_ready'] else 'FAIL'}",f"Training/evaluation: {'PASS' if report['training_evaluation_ready'] else 'FAIL'}",report["overnight_extraction"],""]
    lines += [f"- {c['status']}: {c['check']} — {c['reason']}" for c in report["checks"]]
    a.report.with_suffix(".md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":main()
