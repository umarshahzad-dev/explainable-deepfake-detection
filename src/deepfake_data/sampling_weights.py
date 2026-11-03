"""Produce auditable train-only dataset/class/method/group sampling weights."""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter,defaultdict
from pathlib import Path
from .split import file_sha256


def create(manifest,output,report):
    datasets=set();labels=defaultdict(set);methods=defaultdict(set);groups=defaultdict(set);media=Counter()
    with Path(manifest).open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("project_partition")!="train":continue
            if row.get("proposal_status")=="DRAFT_NOT_APPROVED":raise ValueError("Draft proposals cannot be sampled")
            d=row.get("dataset","");l=row.get("label","");m=row.get("method","");g=row.get("group_id","");path=row.get("input_path","")
            if not d or l not in {"0","1"} or not m or not g or not path:raise ValueError("Training rows require dataset, binary label, method, group and input path")
            datasets.add(d);labels[d].add(l);methods[(d,l)].add(m);groups[(d,l,m)].add(g);media[(d,l,m,g)]+=1
    if not datasets:raise ValueError("No training rows")
    if any(labels[d]!={"0","1"} for d in datasets):raise ValueError("Every included dataset must contain both labels to use this balancing policy")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);total=0;by_dataset=Counter();count=0
    with Path(manifest).open(newline="",encoding="utf-8") as src,output.open("w",newline="",encoding="utf-8") as dst:
        fields=["input_path","dataset","method","label","group_id","weight"]
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for row in csv.DictReader(src):
            if row.get("project_partition")!="train":continue
            d,l,m,g=(row[x] for x in ("dataset","label","method","group_id"))
            weight=1/(len(datasets)*len(labels[d])*len(methods[(d,l)])*len(groups[(d,l,m)])*media[(d,l,m,g)])
            writer.writerow({"input_path":row["input_path"],"dataset":d,"method":m,"label":l,"group_id":g,"weight":format(weight,'.17g')})
            by_dataset[d]+=weight;total+=weight;count+=1
    result={"policy":"equal dataset, then class, then method, then group, then media within group","training_rows":count,"dataset_weight_totals":dict(by_dataset),"weight_total":total,"source_manifest_sha256":file_sha256(manifest),"weights_sha256":file_sha256(output),"limitations":["Weights do not validate source independence or prevent dataset-specific visual shortcuts", "A training implementation must apply the weights; no training is performed here", "Validation, calibration and test rows are excluded"]}
    report=Path(report);report.parent.mkdir(parents=True,exist_ok=True);report.write_text(json.dumps(result,indent=2),encoding="utf-8");return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--report',required=True);a=p.parse_args();print(json.dumps(create(a.manifest,a.output,a.report),indent=2))

if __name__=='__main__':main()
