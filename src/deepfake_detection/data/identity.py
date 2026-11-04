"""Reviewed-person coverage reports; this never infers identities from filenames."""
import csv
import json
from io import StringIO
from deepfake_detection.data.manifests import check_splits
from deepfake_detection.common.paths import Workspace,read_jsonl,write_json,atomic_bytes


def identity_audit(manifests,workspace=None,*,output=None):
    ws=workspace or Workspace()
    rows=[]
    for name in manifests:
        video=ws.manifest(name)/"videos.jsonl"
        rows.extend(read_jsonl(video if video.exists() else ws.manifest(name)/"crops.jsonl"))
    if not rows: raise ValueError("Identity audit requires nonempty manifests")
    result=check_splits(rows)
    result["exit_code"]=0 if result["ok"] and result["person_unverified_records"]==0 else 2
    folder=ws.output(output,explicit=True) if output is not None else ws.report("identity-audit")
    result["report_path"]=str(folder)
    stream=StringIO(newline="")
    fields=["family_token","split","n_records","verified_person_ids","unverified_reason","recommended_action"]
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for family in result["identity_family_rows"]:
        writer.writerow({**family,"verified_person_ids":json.dumps(family["verified_person_ids"])})
    atomic_bytes(folder/"families.csv",stream.getvalue().encode("utf-8"),ws)
    write_json(folder/"summary.json",result,ws)
    return result
