"""Inspect a bounded, stratified selection of videos and produce a contact sheet."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from .extract import add_options, config_from_args, extract_video, make_detector, output_key, validate_output


def choose_representative(rows, limit):
    buckets={}
    for row in rows:
        if row.get("media_type") != "video": continue
        key=(row.get("dataset", "unknown"),row.get("label", "unknown"))
        buckets.setdefault(key,[]).append(row)
    selected=[]
    while buckets and len(selected)<limit:
        for key in sorted(list(buckets)):
            if len(selected)>=limit: break
            selected.append(buckets[key].pop(0))
            if not buckets[key]: del buckets[key]
    return selected


def contact_sheet(results, output):
    from PIL import Image, ImageDraw, ImageOps
    width=840; cell_height=260
    sheet=Image.new("RGB",(width,max(1,len(results))*cell_height),"white"); draw=ImageDraw.Draw(sheet)
    for i,(folder,frame,provenance) in enumerate(results):
        top=i*cell_height
        for j,name in enumerate((frame["original_file"],frame["file"])):
            with Image.open(folder/name) as source:
                thumb=ImageOps.contain(source.convert("RGB"),(405,220));sheet.paste(thumb,(j*420,top))
        text=f"{provenance.get('dataset','video')} {provenance.get('method','')} frame {frame['frame_index']} conf={frame['confidence']:.3f} sharp={frame['sharpness']:.1f} pad={frame['padding_fraction']:.1%}"
        draw.text((4,top+224),text,fill="black")
    sheet.save(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--manifest",type=Path);p.add_argument("--video",action="append",type=Path,default=[]);p.add_argument("--difficult-video",action="append",type=Path,default=[]);p.add_argument("--max-videos",type=int,default=4);p.add_argument("--output",required=True,type=Path);add_options(p);a=p.parse_args()
    if not a.manifest and not a.video and not a.difficult_video:p.error("Provide --manifest or at least one --video")
    if a.max_videos<1:p.error("max-videos must be positive")
    try: target=validate_output(a.output);config=config_from_args(a);config.validate()
    except ValueError as e:p.error(str(e))
    rows=[]
    for video in a.difficult_video+a.video:
        rows.append({"absolute_path":str(video.resolve()),"dataset":"manual","method":"difficult" if video in a.difficult_video else "manual","relative_path":str(video.resolve()),"media_type":"video"})
    if a.manifest:
        with a.manifest.open(newline="",encoding="utf-8") as source: rows.extend(choose_representative(csv.DictReader(source),a.max_videos))
    unique=[];seen={}
    for row in rows:
        key=str(Path(row["absolute_path"]).resolve()).casefold()
        if key in seen:
            existing=seen[key]
            if existing.get("method") == "difficult" and row.get("dataset") != "manual":
                existing.update(row); existing["inspection_note"]="difficult"
        else:
            if len(unique)>=a.max_videos: continue
            entry=dict(row);seen[key]=entry;unique.append(entry)
    if not unique:raise SystemExit("No videos found")
    detector=make_detector(a.device); samples=[]; summary=[]; reasons=Counter()
    for row in unique:
        folder=target/output_key(row);provenance={k:row.get(k,"") for k in ("dataset","method","relative_path","group_id","project_partition","label","inspection_note")}
        try:
            result=extract_video(Path(row["absolute_path"]),folder,detector,config=config,provenance=provenance,save_originals=True)
            summary.append({"video":row["absolute_path"],"selected":len(result["frames"]),"rejections":result["rejection_counts"],"zero_yield":result["zero_yield"]})
            reasons.update(result["rejection_counts"])
            samples.extend((folder,frame,provenance) for frame in result["frames"])
        except (OSError,ValueError,RuntimeError) as exc: summary.append({"video":row["absolute_path"],"error":str(exc)})
    target.mkdir(parents=True,exist_ok=True);contact_sheet(samples,target/"contact_sheet.jpg")
    report={"videos":summary,"selected":len(samples),"rejection_counts":dict(reasons)}
    (target/"quality_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if any("error" in r for r in summary):raise SystemExit("Some inspections failed")

if __name__ == "__main__":main()
