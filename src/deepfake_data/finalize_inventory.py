"""Finalize a completed inventory with dimensions while preserving source partition provenance."""
import argparse
import csv
import json
import os
from collections import Counter,defaultdict
from pathlib import Path
from .split import file_sha256


def finalize(manifest, report_path):
    manifest=Path(manifest);report_path=Path(report_path)
    report=json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("scan_limited",True) or report.get("errors"):raise ValueError("Cannot finalize incomplete inventory")
    temporary=manifest.with_name(manifest.name+".finalizing");stats=defaultdict(Counter);partition_overlaps=0;rows=0
    with manifest.open(newline="",encoding="utf-8") as src,temporary.open("w",newline="",encoding="utf-8") as dst:
        reader=csv.DictReader(src);fields=reader.fieldnames or []
        if "metadata_status" not in fields:raise ValueError("Inventory lacks metadata status")
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for row in reader:
            if row.get("dataset")=="df40" and "|" in row.get("original_partition",""):
                partition_overlaps+=1
            dataset=row.get("dataset","");rows+=1
            for field in ("method","media_type","label","source_domain","metadata_status"):
                stats[(dataset,field)][row.get(field,"") or "unknown"]+=1
            writer.writerow(row)
    expected=sum(n for key,n in report.get("counts",{}).items() if key.startswith("indexed:"))
    if rows!=expected:raise ValueError(f"Inventory row count mismatch: expected {expected}, found {rows}")
    os.replace(temporary,manifest)
    report["counts"]["metadata_partition_overlaps"]=partition_overlaps
    report["dimensions"]={dataset+":"+field:dict(sorted(counter.items())) for (dataset,field),counter in sorted(stats.items())}
    report["manifest_sha256"]=file_sha256(manifest)
    report_path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    return {"records":rows,"partition_overlaps":partition_overlaps,"manifest_sha256":report["manifest_sha256"]}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True);p.add_argument("--report",required=True);a=p.parse_args();print(json.dumps(finalize(a.manifest,a.report),indent=2))

if __name__=="__main__":main()
