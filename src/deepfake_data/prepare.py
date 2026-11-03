"""Orchestrate the frozen Xception preparation release with resumable stages."""
from __future__ import annotations
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from .extract import ExtractionConfig, extract_video, make_detector, output_key, validate_output
from .process_stills import process
from .split import file_sha256
from .training_manifest import build_rows
from .validate_prepared import validate

RELEASE_SHA256='5aa0ce1f9c5f9db2b430f492ec672536dd98f63d375c17f0bb5bb644d4952aeb'
SEEK_EQUIVALENT_LEGACY_EXTRACT_SHA256='d23879f796ebd7afe5a5d714be38954dd9002948f6717608877d1dae7251bc51'
SEEK_MIGRATION_LEGACY_PREPARE_SHA256='ee483911b22fd47797ce53153c09a8670e76f93586d3ae53b176f1fa2878f49a'
PARTS={'train','val','calibration','test'}
CONFIG=ExtractionConfig(max_frames=10,candidates_per_segment=3,min_sharpness=15)


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def read_release(manifest, expected_sha=RELEASE_SHA256):
    if file_sha256(manifest)!=expected_sha:raise ValueError('Frozen release SHA-256 mismatch')
    rows=[];keys={};groups={};source_paths=set()
    with Path(manifest).open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('project_partition') not in PARTS:continue
            if row.get('label') not in {'0','1'} or not row.get('group_id') or row.get('proposal_status')=='DRAFT_NOT_APPROVED':raise ValueError('Invalid active release record')
            if row.get('media_type') not in {'video','image','gif'}:raise ValueError('Unsupported active media type')
            key=output_key(row);source=row.get('absolute_path','')
            if key in keys:raise ValueError(f'Duplicate or colliding source output key: {source}')
            canonical=str(Path(source).resolve()).casefold()
            if not source or canonical in source_paths:raise ValueError(f'Duplicate or missing source path: {source}')
            source_paths.add(canonical)
            keys[key]=source
            group=row['group_id'];part=row['project_partition']
            if group in groups and groups[group]!=part:raise ValueError('Release source group crosses partitions')
            groups[group]=part;rows.append(row)
    if not rows:raise ValueError('Release has no active sources')
    return rows


def compatible_preparation_state(old, current):
    if old==current:return True
    if (old.get('extract_code_sha256')!=SEEK_EQUIVALENT_LEGACY_EXTRACT_SHA256
        or old.get('prepare_code_sha256')!=SEEK_MIGRATION_LEGACY_PREPARE_SHA256):return False
    previous=dict(old);candidate=dict(current)
    for key in ('extract_code_sha256','prepare_code_sha256'):
        previous.pop(key,None);candidate.pop(key,None)
    previous_video=dict(previous.get('video',{}));candidate_video=dict(candidate.get('video',{}))
    if 'inference_batch_size' not in previous_video and candidate_video.get('inference_batch_size')==3:
        candidate_video.pop('inference_batch_size')
        candidate['video']=candidate_video
    return previous==candidate


def sample_rows(rows):
    """Bounded FF++, Celeb-DF and DF40 sample, never written into the release."""
    chosen=[]
    for dataset,media in [('ffpp','video'),('celebdf','video'),('df40','image')]:
        found=next((r for r in rows if r['dataset']==dataset and r['media_type']==media),None)
        if not found:raise ValueError(f'No {dataset} {media} sample in release')
        chosen.append(found)
    return chosen


def run(manifest, output, *, expected_sha=RELEASE_SHA256, smoke=False, detector=None, check_cuda=True, min_free=15*1024**3, config=CONFIG):
    output=validate_output(output);manifest=Path(manifest).resolve()
    if manifest.is_relative_to(output):raise ValueError('Output contains release manifest')
    config.validate();rows=read_release(manifest,expected_sha)
    if smoke:rows=sample_rows(rows)
    if check_cuda:
        try:
            import cv2, torch, PIL, facenet_pytorch  # noqa: F401
        except ImportError as exc:raise RuntimeError(f'Missing extraction dependency: {exc}') from exc
        if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    output.mkdir(parents=True,exist_ok=True);internal=output/'.state';internal.mkdir(exist_ok=True)
    versions={}
    for package in ('Pillow','opencv-python','facenet-pytorch','torch'):
        try:versions[package]=version(package)
        except PackageNotFoundError:versions[package]='not-installed'
    signature={'version':1,'manifest_sha256':expected_sha,'video':asdict(config),'still_size':299,'gif_frames':10,'device':'cuda' if check_cuda else 'injected','smoke':smoke,'prepare_code_sha256':file_sha256(__file__),'python':sys.version,'packages':versions,
               'extract_code_sha256':file_sha256(Path(__file__).with_name('extract.py')),'stills_code_sha256':file_sha256(Path(__file__).with_name('process_stills.py'))}
    state=internal/'config.json'
    if state.exists():
        old=json.loads(state.read_text(encoding='utf-8'))
        if not compatible_preparation_state(old,signature):raise ValueError('Incompatible preparation configuration; select a new output directory')
    else:
        if any((output/part).exists() for part in PARTS):raise ValueError('Unmanaged output directory already contains partition folders')
        atomic_json(state,signature)
    for part in sorted(PARTS):
        for label in ('real','fake'):(output/part/label).mkdir(parents=True,exist_ok=True)
    log=internal/'progress.log';errors=[];zero=[];read_errors=[];processed=Counter();result={'status':'running','config':signature,'sources_expected':len(rows)}
    atomic_json(internal/'status.json',result)
    def say(message):
        print(message,flush=True)
        with log.open('a',encoding='utf-8') as f:f.write(message+'\n');f.flush()
    try:
        if shutil.disk_usage(output).free < min_free:raise OSError('Insufficient free disk space')
        if detector is None and any(r['media_type']=='video' for r in rows):detector=make_detector('cuda' if check_cuda else 'cpu')
        for index,row in enumerate(rows,1):
            if shutil.disk_usage(output).free < min_free:raise OSError('Free disk space below reserve; resume after freeing space')
            part=row['project_partition'];label='real' if row['label']=='1' else 'fake';key=output_key(row)
            folder=internal/('videos' if row['media_type']=='video' else 'stills')/key
            image_dir=output/part/label/key
            if image_dir.exists() and any(image_dir.iterdir()) and not (folder/'state.json').is_file():raise ValueError(f'Unmanaged image output exists: {image_dir}')
            source=Path(row['absolute_path']).resolve()
            if source.is_relative_to(output):raise ValueError('Source resides inside output directory')
            say(f"[{index}/{len(rows)}] {row['dataset']} {part} {row['method']}: {source.name}")
            try:
                if row['media_type']=='video':
                    provenance={k:row.get(k,'') for k in ('dataset','method','relative_path','group_id','project_partition','label')}
                    item=extract_video(source,folder,detector,config=config,provenance=provenance,image_output=image_dir)
                    if item['zero_yield']:zero.append({'source':str(source),'rejection_counts':item['rejection_counts']})
                    if item['rejection_counts'].get('read_failed'):read_errors.append({'source':str(source),'failed_frames':item['rejection_counts']['read_failed']})
                else:item=process(row,folder,299,10,image_output=image_dir)
                processed[(part,row['dataset'],row['label'])]+=1
            except (OSError,ValueError,RuntimeError,KeyError) as exc:
                errors.append({'source':str(source),'error':str(exc)});say(f'ERROR: {exc}')
                # CUDA and disk failures should not cascade through every source.
                if isinstance(exc,(OSError,RuntimeError)):raise
        if errors:raise RuntimeError(f'{len(errors)} source processing errors')
        image_rows,skipped=build_rows(rows,internal/'videos',stills_root=internal/'stills')
        unexpected=[x for x in skipped if x.get('reason')!='zero_yield']
        if unexpected:raise RuntimeError(f'{len(unexpected)} sources have incomplete or incompatible output')
        if not image_rows:raise RuntimeError('No prepared images')
        fields=list(rows[0]);fields += [x for x in ('input_path','source_frame_index','output_sha256') if x not in fields]
        tmp=output/'images.csv.partial';seen=set()
        with tmp.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for row in image_rows:
                target=Path(row['input_path']).resolve();base=(output/row['project_partition']/('real' if row['label']=='1' else 'fake')).resolve()
                if not target.is_relative_to(base) or target in seen:raise ValueError('Output path collision or partition mismatch')
                seen.add(target);row['output_sha256']=file_sha256(target);writer.writerow(row)
            f.flush();os.fsync(f.fileno())
        checked=validate(tmp,decode=True,hash_outputs=True)
        blocking={k:v for k,v in checked['issues'].items() if k!='cross_dataset_source_independence_unverified' and not (smoke and k=='partition_class_coverage')}
        if set(checked['datasets'])!={'ffpp','celebdf','df40'}:blocking['missing_required_datasets']=checked['datasets']
        if blocking:raise RuntimeError('Prepared image validation failed: '+json.dumps(blocking))
        final=output/'images.csv'
        if final.exists() and file_sha256(final)!=file_sha256(tmp):raise ValueError('Existing image manifest differs; refuse overwrite')
        if final.exists():tmp.unlink()
        else:os.replace(tmp,final)
        result.update(status='complete',processed_sources=sum(processed.values()),image_count=len(image_rows),zero_yield=zero,read_errors=read_errors,errors=[],validation=checked,images_sha256=file_sha256(final))
        atomic_json(output/'preparation-report.json',result);atomic_json(internal/'status.json',result)
        say(f"COMPLETE: {len(image_rows)} images, {len(zero)} zero-yield videos")
        return result
    except BaseException as exc:
        result.update(status='failed',error=f'{type(exc).__name__}: {exc}',errors=errors,zero_yield=zero,read_errors=read_errors,processed_sources=sum(processed.values()))
        atomic_json(internal/'status.json',result);atomic_json(output/'preparation-report.json',result);say('FAILED: '+result['error']);raise
