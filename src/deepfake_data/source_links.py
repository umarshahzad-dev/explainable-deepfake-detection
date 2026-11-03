"""Conservative DeepFakeDetection actor/source grouping and manifest enrichment."""
from __future__ import annotations
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from .split import file_sha256

FAKE = re.compile(r"(\d+)_(\d+)__(.+)__([A-Za-z0-9]+)")
REAL = re.compile(r"(\d+)__(.+)")


def dfd_components(root):
    root=Path(root); originals={}; fakes=[]; parent={}; issues=Counter()
    for path in (root/"original_sequences/actors").rglob("*.mp4"):
        m=REAL.fullmatch(path.stem)
        if not m: issues["original_unparsed"]+=1;continue
        originals[path.stem]=path;parent.setdefault(m[1],m[1])
    def find(actor):
        parent.setdefault(actor,actor)
        while parent[actor]!=actor:
            parent[actor]=parent[parent[actor]];actor=parent[actor]
        return actor
    for path in (root/"manipulated_sequences/DeepFakeDetection").rglob("*.mp4"):
        m=FAKE.fullmatch(path.stem)
        if not m: issues["fake_unparsed"]+=1;continue
        target,source,scene,_=m.groups()
        if target+"__"+scene not in originals:
            issues["target_original_missing"]+=1;continue
        if source+"__"+scene not in originals:issues["source_scene_missing"]+=1
        left,right=find(target),find(source);parent[max(left,right)]=min(left,right);fakes.append((path.stem,target))
    # An actor missing its own original remains evidence of a name, not a verified local source.
    actors={REAL.fullmatch(stem)[1] for stem in originals}
    mapping={}
    for stem in originals:
        match=REAL.fullmatch(stem);mapping[("original",stem)]="ffpp:dfd:actors:"+find(match[1])
    for stem,target in fakes:
        match=FAKE.fullmatch(stem)
        if match[2] in actors: mapping[("DeepFakeDetection",stem)]="ffpp:dfd:actors:"+find(target)
        else: issues["source_actor_missing"]+=1
    groups=Counter(mapping.values())
    return mapping,{"originals":len(originals),"parsed_fakes_with_target":len(fakes),"actors":len(actors),"actor_components":len({find(a) for a in actors}),"group_records":dict(sorted(groups.items())),"issues":dict(issues),"evidence":"FaceForensics official DeepFakeDetection filename convention and local matching target-scene originals", "limitations":["Components establish known actor links within DeepFakeDetection, not separation from other datasets", "Source actor's same-scene original may be unavailable", "No exact duplicate or global identity proof"]}


def enrich(manifest, output, root, report):
    mapping,audit=dfd_components(root);output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);counts=Counter()
    with Path(manifest).open(newline="",encoding="utf-8") as src,output.open("w",newline="",encoding="utf-8") as dst:
        reader=csv.DictReader(src);fields=list(reader.fieldnames or [])
        for name in ("source_link_evidence","previous_group_id"):
            if name not in fields: fields.append(name)
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for row in reader:
            rel=Path(row.get("relative_path",""));method=row.get("method","")
            is_actor_original=row.get("dataset")=="ffpp" and method=="original" and "actors" in rel.parts
            if row.get("dataset")=="ffpp" and (method=="DeepFakeDetection" or is_actor_original):
                group=mapping.get(("original" if is_actor_original else method,rel.stem))
                if group:
                    row["previous_group_id"]=row.get("group_id","");row["group_id"]=group
                    row["group_status"]="sample_verified";row["relationship_status"]="actor_links_verified_cross_dataset_unknown"
                    row["source_link_evidence"]="official_dfd_actor_filenames_and_local_target_scene";counts["linked:"+method]+=1
                else:counts["unresolved:"+method]+=1
            writer.writerow(row)
    audit.update(manifest_counts=dict(counts),source_manifest_sha256=file_sha256(manifest),output_manifest_sha256=file_sha256(output))
    Path(report).parent.mkdir(parents=True,exist_ok=True);Path(report).write_text(json.dumps(audit,indent=2),encoding="utf-8");return audit


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True);p.add_argument("--output",required=True);p.add_argument("--report",required=True);p.add_argument("--ffpp-root",type=Path,default=Path(r"D:\University\datasets\deepfake\ffpp"));a=p.parse_args();print(json.dumps(enrich(a.manifest,a.output,a.ffpp_root,a.report),indent=2))

if __name__=="__main__":main()
