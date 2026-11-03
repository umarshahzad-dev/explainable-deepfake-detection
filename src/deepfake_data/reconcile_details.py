"""Write review queues for all DF40 missing, ambiguous and unlisted records."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from .inventory import canonical_df40, disk_key, metadata_files, metadata_records


def generate(manifest, config_path, output):
    config_path=Path(config_path).resolve();config=json.loads(config_path.read_text(encoding="utf-8-sig"));raw=config_path.parent.parent/config["raw_root"];settings=config["datasets"]["df40"];root=raw/settings["path"];meta=raw/settings["metadata_path"]
    output=Path(output);output.mkdir(parents=True,exist_ok=True);physical=set();ambiguous=set();counts=Counter()
    fields=["relative_path","absolute_path","method","metadata_status","label","metadata_sources"]
    with Path(manifest).open(newline="",encoding="utf-8") as src, (output/"local_metadata_issues.csv").open("w",newline="",encoding="utf-8") as dest:
        writer=csv.DictWriter(dest,fieldnames=fields);writer.writeheader()
        for row in csv.DictReader(src):
            if row.get("dataset")!="df40":continue
            key=disk_key(Path(row["relative_path"]));status=row.get("metadata_status","")
            if status=="ambiguous":ambiguous.add(key)
            else:physical.add(key)
            if status in {"unlisted","ambiguous","conflict","partition_conflict"}:
                writer.writerow({k:row.get(k,"") for k in fields});counts["local:"+status]+=1
    methods={p.name for p in root.iterdir() if p.is_dir()}
    metadata_fields=["metadata_source","path","original_json_label","original_partition","sample_id","reason"]
    with (output/"json_records_unmatched.csv").open("w",newline="",encoding="utf-8") as dest:
        writer=csv.DictWriter(dest,fieldnames=metadata_fields);writer.writeheader()
        for file in metadata_files(meta,methods):
            for record in metadata_records(file):
                key=canonical_df40(record["path"])
                if key is None:counts["external_references"]+=1;continue
                if key in ambiguous:reason="ambiguous_local_mapping"
                elif key not in physical:reason="no_matching_local_file"
                else:continue
                writer.writerow({**{k:record.get(k,"") for k in metadata_fields},"reason":reason});counts["json:"+reason]+=1
    report={"counts":dict(counts),"limitations":["JSON references outside DF40 are counted separately, not assumed absent","Rows list references, not unique source videos","Path matches do not prove identity independence"]}
    (output/"reconciliation-summary.json").write_text(json.dumps(report,indent=2),encoding="utf-8");return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",required=True);p.add_argument("--config",default="configs/datasets.json");p.add_argument("--output",required=True);a=p.parse_args();print(json.dumps(generate(a.manifest,a.config,a.output),indent=2))

if __name__=="__main__":main()
