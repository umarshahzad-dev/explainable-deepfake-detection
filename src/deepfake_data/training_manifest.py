"""Combine completed video crops and pre-existing still images into a training manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def build_rows(rows, faces_root, gif_root=None, stills_root=None, state=None):
    state = state if state is not None else {}
    output=[]; skipped=[]; crop_signature=state.get("crop_signature")
    for row in rows:
        if row.get("project_partition") not in {"train","val","calibration","test"}: continue
        if row.get("proposal_status")=="DRAFT_NOT_APPROVED":skipped.append({"path":row.get("absolute_path"),"reason":"draft_not_approved"});continue
        if row.get("label") not in {"0","1"}: skipped.append({"path":row.get("absolute_path"),"reason":"unknown_label"});continue
        if row.get("media_type") in {"image","gif"} and stills_root is not None:
            from .extract import output_key
            folder=stills_root/output_key(row); metadata=folder/"complete.json"
            if not metadata.is_file():skipped.append({"path":row["absolute_path"],"reason":"still_not_processed"});continue
            try:
                result=json.loads(metadata.read_text(encoding="utf-8"));config=result["config"];provenance=config.get("provenance",{})
                if Path(config["source"]).resolve()!=Path(row["absolute_path"]).resolve():raise ValueError("source_mismatch")
                if any(provenance.get(k,"")!=row.get(k,"") for k in ("dataset","method","relative_path","group_id","project_partition","label")):raise ValueError("provenance_mismatch")
                if config.get("size")!=299 or config.get("policy")!="rgb-letterbox-no-recrop":raise ValueError("incompatible_still_config")
                for frame in result["frames"]:
                    image=folder/frame["file"]
                    if image.is_file():output.append({**row,"input_path":str(image.resolve()),"source_frame_index":frame["frame_index"]})
                    else:skipped.append({"path":str(image),"reason":"missing_still_frame"})
            except (ValueError,KeyError,OSError) as exc:skipped.append({"path":row["absolute_path"],"reason":str(exc)})
        elif row.get("media_type") == "image":
            output.append({**row,"input_path":row["absolute_path"],"source_frame_index":""})
        elif row.get("media_type") == "video":
            key=hashlib.sha256((row.get("dataset", "")+":"+row.get("relative_path",row["absolute_path"])).encode()).hexdigest()[:20]
            folder=faces_root/key; metadata=folder/"complete.json"
            if not metadata.is_file(): skipped.append({"path":row["absolute_path"],"reason":"not_extracted"});continue
            result=json.loads(metadata.read_text(encoding="utf-8"))
            config=result.get("config", {})
            if config.get("version") != 2 or "parameters" not in config:
                skipped.append({"path":row["absolute_path"],"reason":"legacy_or_invalid_crop_config"});continue
            if Path(config["video"]).resolve() != Path(row["absolute_path"]).resolve():
                skipped.append({"path":row["absolute_path"],"reason":"source_mismatch"});continue
            provenance=config.get("provenance", {})
            if any(provenance.get(key, "") != row.get(key, "") for key in ("dataset", "method", "relative_path", "group_id", "project_partition")):
                skipped.append({"path":row["absolute_path"],"reason":"provenance_mismatch"});continue
            if not result.get("frames"):
                skipped.append({"path":row["absolute_path"],"reason":"zero_yield"});continue
            current={"parameters":config["parameters"],"runtime":config.get("runtime")}
            if crop_signature is None: crop_signature=current
            elif current != crop_signature:
                skipped.append({"path":row["absolute_path"],"reason":"incompatible_crop_config"});continue
            for frame in result["frames"]:
                path=folder/frame["file"]
                if path.is_file(): output.append({**row,"input_path":str(path.resolve()),"source_frame_index":frame["frame_index"]})
                else: skipped.append({"path":str(path),"reason":"missing_crop"})
        elif row.get("media_type") == "gif" and gif_root is not None:
            source=Path(row["absolute_path"]).resolve()
            folder=gif_root/hashlib.sha256(str(source).encode()).hexdigest()[:20]
            metadata=folder/"complete.json"
            if not metadata.is_file(): skipped.append({"path":str(source),"reason":"gif_requires_conversion"}); continue
            result=json.loads(metadata.read_text(encoding="utf-8"))
            if Path(result["config"]["source"]).resolve() != source:
                skipped.append({"path":str(source),"reason":"source_mismatch"}); continue
            for frame in result["frames"]:
                path=folder/frame["file"]
                if path.is_file(): output.append({**row,"input_path":str(path.resolve()),"source_frame_index":frame["frame_index"]})
                else: skipped.append({"path":str(path),"reason":"missing_gif_frame"})
        else: skipped.append({"path":row.get("absolute_path"),"reason":"gif_requires_conversion"})
    state["crop_signature"] = crop_signature
    return output,skipped


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest",type=Path);p.add_argument("--faces",required=True,type=Path);p.add_argument("--output",required=True,type=Path);p.add_argument("--gif-frames",type=Path);p.add_argument("--stills",type=Path);a=p.parse_args()
    from collections import Counter
    from itertools import islice
    a.output.parent.mkdir(parents=True,exist_ok=True)
    issue_path=a.output.with_suffix(".issues.jsonl"); total=0; issues=Counter(); examples=[];state={}
    with a.manifest.open(newline="",encoding="utf-8") as f, a.output.open("w",newline="",encoding="utf-8") as output, issue_path.open("w",encoding="utf-8") as issue_file:
        reader=csv.DictReader(f);fields=list(reader.fieldnames or [])
        fields += [field for field in ("input_path","source_frame_index") if field not in fields]
        writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader()
        while True:
            batch=list(islice(reader,1000))
            if not batch:break
            rows,skipped=build_rows(batch,a.faces,a.gif_frames,a.stills,state)
            writer.writerows(rows);total+=len(rows)
            for item in skipped:
                issues[item.get("reason","unknown")]+=1
                if len(examples)<50:examples.append(item)
                issue_file.write(json.dumps(item)+"\n")
    report={"included":total,"skipped_count":sum(issues.values()),"skipped_by_reason":dict(issues),"skipped_examples":examples,"full_issues_path":str(issue_path)}
    a.output.with_suffix(".report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(f"Included {total} images; skipped {sum(issues.values())} records")

if __name__ == "__main__":main()
