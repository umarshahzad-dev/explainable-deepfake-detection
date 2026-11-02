"""Combine completed video crops and pre-existing still images into a training manifest."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def build_rows(rows, faces_root):
    output=[]; skipped=[]
    for row in rows:
        if row.get("project_partition") not in {"train","val","test"}: continue
        if row.get("label") not in {"0","1"}: skipped.append({"path":row.get("absolute_path"),"reason":"unknown_label"});continue
        if row.get("media_type") == "image":
            output.append({**row,"input_path":row["absolute_path"],"source_frame_index":""})
        elif row.get("media_type") == "video":
            key=hashlib.sha256((row.get("dataset", "")+":"+row.get("relative_path",row["absolute_path"])).encode()).hexdigest()[:20]
            folder=faces_root/key; metadata=folder/"complete.json"
            if not metadata.is_file(): skipped.append({"path":row["absolute_path"],"reason":"not_extracted"});continue
            result=json.loads(metadata.read_text(encoding="utf-8"))
            if Path(result["config"]["video"]).resolve() != Path(row["absolute_path"]).resolve():
                skipped.append({"path":row["absolute_path"],"reason":"source_mismatch"});continue
            for frame in result["frames"]:
                path=folder/frame["file"]
                if path.is_file(): output.append({**row,"input_path":str(path.resolve()),"source_frame_index":frame["frame_index"]})
                else: skipped.append({"path":str(path),"reason":"missing_crop"})
        else: skipped.append({"path":row.get("absolute_path"),"reason":"gif_requires_conversion"})
    return output,skipped


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("manifest",type=Path);p.add_argument("--faces",required=True,type=Path);p.add_argument("--output",required=True,type=Path);a=p.parse_args()
    with a.manifest.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f);fields=list(reader.fieldnames or []); rows,skipped=build_rows(reader,a.faces)
    fields += [f for f in ("input_path","source_frame_index") if f not in fields]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    a.output.with_suffix(".report.json").write_text(json.dumps({"included":len(rows),"skipped":skipped},indent=2),encoding="utf-8")
    print(f"Included {len(rows)} images; skipped {len(skipped)} records")

if __name__ == "__main__":main()
