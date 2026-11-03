"""Generate and apply reviewable cross-dataset relationship evidence."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = ["group_id", "holdout", "dataset", "source_domain", "sample_examples", "status", "evidence"]


def candidates(rows, holdout):
    groups={}
    for row in rows:
        group=row.get("group_id")
        if not group: continue
        data=groups.setdefault(group,{"datasets":set(),"domains":set(),"samples":set()})
        data["datasets"].add(row.get("dataset",""));data["domains"].add(row.get("source_domain",""))
        sample=row.get("sample_id","")
        if len(data["samples"]) < 3: data["samples"].add(sample)
    for group, data in sorted(groups.items()):
        datasets=sorted(data["datasets"]);domains=sorted(data["domains"])
        suspect=any(x.casefold() in ({"cdf","celeb-real","youtube-real","celeb-synthesis"} if holdout=="celebdf" else {"ff"}) for x in domains)
        status="verified_overlap" if holdout in datasets and len(datasets)>1 else "suspected_overlap" if holdout not in datasets and suspect else "unknown"
        yield {"group_id":group,"holdout":holdout,"dataset":"|".join(datasets),"source_domain":"|".join(domains),"sample_examples":"|".join(sorted(data["samples"])),"status":status,"evidence":"automatic: source-domain or shared group" if status!="unknown" else ""}

def apply_reviews(rows, path, holdout):
    reviews={}
    with path.open(newline="",encoding="utf-8") as f:
        for record in csv.DictReader(f):
            if record.get("holdout") != holdout: continue
            key=record.get("group_id",""); status=record.get("status","")
            if status not in {"unknown","suspected_overlap","verified_overlap","reviewed_independent"}: raise ValueError("Invalid review status")
            if status != "unknown" and not record.get("evidence","").strip(): raise ValueError("Reviewed status requires documented evidence")
            if key in reviews: raise ValueError("Duplicate group in review")
            reviews[key]=status
    return [{**r,"overlap_status":reviews.get(r.get("group_id"),"unknown")} for r in rows]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest",type=Path);p.add_argument("--holdout",required=True,choices=("ffpp","celebdf"));p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.manifest.open(newline="",encoding="utf-8") as source,a.output.open("w",newline="",encoding="utf-8") as dest:
        writer=csv.DictWriter(dest,fieldnames=FIELDS);writer.writeheader();writer.writerows(candidates(csv.DictReader(source),a.holdout))

if __name__ == "__main__":main()
