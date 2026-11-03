"""Bounded, resumable DF40 still/GIF normalization; never recrops faces."""
from __future__ import annotations
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from .extract import output_key, sampled_indices, validate_output
from .split import file_sha256


def process(row, output, size=299, max_gif_frames=10, image_output=None):
    from PIL import Image,ImageOps
    source=Path(row["absolute_path"]).resolve();output=Path(output)
    if size<1 or max_gif_frames<1:raise ValueError("size and GIF limit must be positive")
    if not source.is_file():raise ValueError("source file missing")
    provenance={k:row.get(k,"") for k in ("dataset","method","relative_path","group_id","project_partition","label")}
    signature={"version":1,"source":str(source),"bytes":source.stat().st_size,"mtime_ns":source.stat().st_mtime_ns,"size":size,"max_gif_frames":max_gif_frames,"policy":"rgb-letterbox-no-recrop","provenance":provenance}
    image_dir=Path(image_output).resolve() if image_output is not None else output.resolve()
    if image_output is not None:signature['image_output']=str(image_dir);image_dir.mkdir(parents=True,exist_ok=True)
    output.mkdir(parents=True,exist_ok=True);state=output/"state.json";complete=output/"complete.json"
    if state.exists() and json.loads(state.read_text(encoding="utf-8"))!=signature:raise ValueError("Incompatible still-image checkpoint")
    if complete.exists():
        try:old=json.loads(complete.read_text(encoding="utf-8"))
        except json.JSONDecodeError:old=None
        if old is not None:
            if old.get("config")!=signature:raise ValueError("Incompatible still-image output")
            if all((output/r["file"]).is_file() for r in old.get("frames",[])):return old
    if not state.exists():state.write_text(json.dumps(signature,indent=2),encoding="utf-8")
    rows=[]
    with Image.open(source) as image:
        source_format=image.format;count=getattr(image,"n_frames",1)
        indices=sampled_indices(count,max_gif_frames) if source_format=="GIF" else [0]
        for index in indices:
            name=f"frame_{index:09d}.png";dest=image_dir/name;temporary=image_dir/(name+".partial")
            marker=output/(name+'.json')
            if marker.is_file() and dest.is_file():
                try: saved=json.loads(marker.read_text(encoding='utf-8'))
                except (OSError,ValueError):saved={}
                if saved.get('sha256')==file_sha256(dest):
                    rows.append({'file':os.path.relpath(dest,output),'frame_index':index});continue
            image.seek(index);normalized=ImageOps.exif_transpose(image).convert("RGB")
            normalized=ImageOps.pad(normalized,(size,size),method=Image.Resampling.LANCZOS,color=(0,0,0),centering=(.5,.5))
            with temporary.open("wb") as stream:
                normalized.save(stream,format="PNG");stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,dest)
            marker_tmp=marker.with_name(marker.name+'.partial');marker_tmp.write_text(json.dumps({'sha256':file_sha256(dest)}),encoding='utf-8');os.replace(marker_tmp,marker)
            rows.append({"file":os.path.relpath(dest,output),"frame_index":index})
    result={"config":signature,"source_format":source_format,"source_frame_count":count,"frames":rows}
    temporary=output/"complete.json.partial";temporary.write_text(json.dumps(result,indent=2),encoding="utf-8");os.replace(temporary,complete);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True);p.add_argument("--partition",choices=("train","val","calibration","test"),default="train");p.add_argument("--max-items",type=int,default=20);p.add_argument("--all-items",action="store_true");p.add_argument("--allow-empty",action="store_true");p.add_argument("--size",type=int,default=299);p.add_argument("--max-gif-frames",type=int,default=10);a=p.parse_args()
    if a.max_items<1 or a.size<1 or a.max_gif_frames<1:p.error("Limits must be positive")
    try:target=validate_output(a.output)
    except ValueError as exc:p.error(str(exc))
    processed=0;failures=[];formats=Counter()
    with a.manifest.open(newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("project_partition")!=a.partition or row.get("media_type") not in {"image","gif"}:continue
            if row.get("proposal_status")=="DRAFT_NOT_APPROVED":raise SystemExit("Draft proposals cannot be processed")
            if row.get("label") not in {"0","1"} or not row.get("group_id"):continue
            if not a.all_items and processed>=a.max_items:break
            processed+=1
            try:
                result=process(row,target/output_key(row),a.size,a.max_gif_frames);formats[result["source_format"]]+=1
                print(f"{processed}: {row['absolute_path']} -> {len(result['frames'])} frames",flush=True)
            except (OSError,ValueError,KeyError) as exc:
                failures.append({"source":row.get("absolute_path",""),"error":str(exc)});print(f"Failed: {exc}",flush=True)
    target.mkdir(parents=True,exist_ok=True)
    summary={"partition":a.partition,"processed":processed,"formats":dict(formats),"failures":failures}
    (target/"processing-report.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    (target/f"processing-report-{a.partition}.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    if failures:raise SystemExit(f"Some media failed; see processing-report-{a.partition}.json")
    if not processed and not a.allow_empty:raise SystemExit("No eligible still images or GIFs in selected partition")
    if not processed:print(f"No eligible still images or GIFs for {a.partition}; skipped")

if __name__=="__main__":main()
