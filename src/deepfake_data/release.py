"""Fail-closed promotion of a reviewed draft into an immutable release manifest.

Review evidence is a human attestation of source relationships, not proof of
person-identity independence. No release is written when any check fails.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import re
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from .extract import validate_output
from .split import file_sha256

ACTIVE={"train","val","calibration","test"}
HEX=re.compile(r"[0-9a-f]{64}")


def inspect(draft, proposal_report, components, component_report, reviews, hash_csv, hash_report, cross_media_review, practical=False):
    draft=Path(draft);manifest_hash=file_sha256(draft);blockers=[];counts=Counter();groups=defaultdict(set);labels=defaultdict(set);dataset_labels=defaultdict(set);methods=defaultdict(set);active_components=set();review_map={};component_map={};evidence_hashes={}
    def load_json(path,label):
        try:return json.loads(Path(path).read_text(encoding='utf-8'))
        except (OSError,ValueError,TypeError) as exc:blockers.append(f'{label} unavailable or invalid: {exc}');return {}
    proposal=load_json(proposal_report,'proposal report');comp_report=load_json(component_report,'component report');hash_data=load_json(hash_report,'hash report');cross={} if practical else load_json(cross_media_review,'cross-media review')
    if proposal.get('draft_manifest_sha256')!=manifest_hash or proposal.get('status')!='DRAFT_BLOCKED_NOT_FOR_EXTRACTION_OR_TRAINING':blockers.append('Proposal report does not match draft')
    if proposal.get('config',{}).get('components_used') is not True:blockers.append('Proposal was not generated with source-component safeguards')
    try:
        if comp_report.get('manifest_sha256')!=proposal.get('source_manifest_sha256') or comp_report.get('component_csv_sha256')!=file_sha256(components):blockers.append('Component evidence does not match source and mapping')
        with Path(components).open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                g=row.get('group_id','');c=row.get('component_id','')
                if not g or not c or g in component_map:blockers.append('Invalid or duplicate source component entry');break
                component_map[g]=c
    except OSError as exc:blockers.append(f'Component mapping unavailable: {exc}')
    try:
        if practical: raise StopIteration
        with Path(reviews).open(newline='',encoding='utf-8') as f:
            for row in csv.DictReader(f):
                component=row.get('component_id','');path=row.get('evidence_path','');digest=row.get('evidence_sha256','')
                if not component or component in review_map or row.get('decision')!='verified' or not row.get('reviewer','').strip() or not path or not HEX.fullmatch(digest):
                    blockers.append('Invalid, duplicate or unverified component review');break
                if path not in evidence_hashes:
                    try:evidence_hashes[path]=file_sha256(path)
                    except OSError:evidence_hashes[path]='missing'
                if evidence_hashes[path]!=digest:blockers.append('Component review evidence file is missing or changed');break
                review_map[component]=row
    except StopIteration:pass
    except OSError as exc:blockers.append(f'Component reviews unavailable: {exc}')
    try:
        valid_hash=hash_data.get('complete') is True and hash_data.get('proposal_mode') is True and hash_data.get('manifest_sha256')==manifest_hash and hash_data.get('hash_csv_sha256')==file_sha256(hash_csv) and not hash_data.get('errors') and not hash_data.get('cross_partition_exact_hash_overlaps')
    except OSError:valid_hash=False
    if not valid_hash:blockers.append('Complete matching source-byte hash audit without cross-partition duplicates is required')
    db=sqlite3.connect('');db.execute('CREATE TABLE hash_rows (path TEXT PRIMARY KEY, partition TEXT, digest TEXT, size INTEGER, mtime_ns INTEGER)');loaded=0
    if valid_hash:
        try:
            with Path(hash_csv).open(newline='',encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    if row.get('status')!='ok' or not HEX.fullmatch(row.get('sha256','')):raise ValueError('Invalid hash row')
                    db.execute('INSERT INTO hash_rows VALUES(?,?,?,?,?)',(str(Path(row['absolute_path']).resolve()).casefold(),row['partition'],row['sha256'],int(row['size']),int(row['mtime_ns'])));loaded+=1
        except (OSError,ValueError,KeyError,sqlite3.IntegrityError) as exc:blockers.append(f'Invalid or duplicate hash data: {exc}')
    hash_overlap_count=db.execute('SELECT COUNT(*) FROM (SELECT digest FROM hash_rows GROUP BY digest HAVING COUNT(DISTINCT partition)>1)').fetchone()[0]
    if hash_overlap_count:blockers.append(f'{hash_overlap_count} exact source hashes cross proposed partitions')
    expected=0;groups_missing=0;hash_missing=0;stat_changed=0;official_violation=0;invalid_labels=0;invalid_metadata=0;inconsistent=0;already_active=0
    with draft.open(newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('project_partition')!='quarantine' or row.get('proposal_status')!='DRAFT_NOT_APPROVED':already_active+=1
            part=row.get('proposed_partition','')
            if part not in ACTIVE:continue
            expected+=1;dataset=row.get('dataset','');label=row.get('label','');method=row.get('method','');group=row.get('group_id','');component=component_map.get(group,'')
            counts[(part,dataset,method,label)]+=1;labels[part].add(label);dataset_labels[(part,dataset)].add(label);methods[(part,dataset)].add(method)
            if label not in {'0','1'}:invalid_labels+=1
            if dataset=='df40' and row.get('metadata_status') not in {'matched','partition_conflict'}:invalid_metadata+=1
            if not component or (not practical and component not in review_map):groups_missing+=1
            elif row.get('proposed_source_group')!=component:inconsistent+=1
            if component:groups[component].add(part);active_components.add(component)
            if dataset=='celebdf' and row.get('original_partition')=='official_test' and part!='test':official_violation+=1
            if valid_hash:
                key=str(Path(row.get('absolute_path','')).resolve()).casefold();found=db.execute('SELECT partition,digest,size,mtime_ns FROM hash_rows WHERE path=?',(key,)).fetchone()
                if not found or found[0]!=part:hash_missing+=1
                else:
                    try:
                        stat=Path(row['absolute_path']).stat()
                        if (stat.st_size,stat.st_mtime_ns)!=(found[2],found[3]):stat_changed+=1
                    except OSError:stat_changed+=1
    overlaps={k:sorted(v) for k,v in groups.items() if len(v)>1}
    if already_active:blockers.append(f'{already_active} input records bypass draft quarantine')
    if invalid_labels:blockers.append(f'{invalid_labels} proposed records have invalid labels')
    if invalid_metadata:blockers.append(f'{invalid_metadata} proposed DF40 records lack reconciled label metadata')
    if official_violation:blockers.append(f'{official_violation} official Celeb-DF test records would be used outside test')
    if groups_missing or inconsistent:blockers.append(f'{groups_missing} proposed records lack reviewed components; {inconsistent} have mismatched components')
    if overlaps:blockers.append(f'{len(overlaps)} reviewed source components cross partitions')
    if valid_hash and (hash_missing or stat_changed or loaded!=expected):blockers.append(f'Hash coverage mismatch: missing={hash_missing}, changed={stat_changed}, rows={loaded}, expected={expected}')
    if any(labels[p]!={'0','1'} for p in ACTIVE):blockers.append('Both classes are required in all four partitions')
    if any(dataset_labels[('train',d)]!={'0','1'} for d in ('ffpp','celebdf','df40')):blockers.append('Training requires both labels from all three datasets')
    for dataset in ('celebdf','df40'):
        eval_labels=set().union(*(dataset_labels[(part,dataset)] for part in ('val','calibration','test')))
        if eval_labels!={'0','1'}:blockers.append(f'{dataset} evaluation requires real and fake coverage')
    if not any(method not in {'original','Deepfakes','Face2Face','FaceSwap','FaceShifter','NeuralTextures','DeepFakeDetection'} for part in ('val','calibration','test') for method in methods[(part,'df40')]):blockers.append('No modern DF40 evaluation method is included')
    active_dataset_partitions={(part,dataset) for part,dataset,_,_ in counts}
    required_dataset_pairs={tuple(sorted((left_dataset,right_dataset))) for left_part,left_dataset in active_dataset_partitions for right_part,right_dataset in active_dataset_partitions if left_part!=right_part and left_dataset!=right_dataset}
    pair_data=cross.get('covered_cross_dataset_pairs'); reviewed=cross.get('reviewed_components'); evidence_files=cross.get('evidence_files'); reviewer=cross.get('reviewer')
    provided_dataset_pairs={tuple(sorted(pair)) for pair in pair_data if isinstance(pair,list) and len(pair)==2 and all(isinstance(item,str) for item in pair)} if isinstance(pair_data,list) else set()
    valid_reviewed=isinstance(reviewed,list) and all(isinstance(item,str) for item in reviewed) and set(reviewed)>=active_components
    cross_ok=(cross.get('status')=='reviewed_complete' and cross.get('scope')=='all_active_cross_partition_source_pairs' and cross.get('manifest_sha256')==manifest_hash and cross.get('component_csv_sha256')==comp_report.get('component_csv_sha256') and type(cross.get('unresolved_count')) is int and cross.get('unresolved_count')==0 and isinstance(reviewer,str) and bool(reviewer.strip()) and valid_reviewed and provided_dataset_pairs>=required_dataset_pairs and isinstance(evidence_files,list) and bool(evidence_files) and all(isinstance(item,dict) for item in evidence_files))
    if cross_ok:
        for item in cross['evidence_files']:
            try:
                if not HEX.fullmatch(item.get('sha256','')) or file_sha256(item['path'])!=item['sha256']:cross_ok=False;break
                evidence_hashes[item['path']]=item['sha256']
            except (OSError,KeyError,TypeError):cross_ok=False;break
    if not practical and not cross_ok:blockers.append('Complete, manifest-bound cross-media/source review with hashed evidence is required')
    verified_inputs={}
    for path in (draft,proposal_report,components,component_report,reviews,hash_csv,hash_report,cross_media_review):
        if path is not None and Path(path).is_file():verified_inputs[str(Path(path).resolve())]=file_sha256(path)
    for path,digest in evidence_hashes.items():
        if digest!='missing':verified_inputs[str(Path(path).resolve())]=digest
    component_counts={p:len({component for component,partitions in groups.items() if p in partitions}) for p in ACTIVE}
    report={'ready':not blockers,'policy':'best_effort' if practical else 'reviewed','blockers':blockers,'verified_inputs':verified_inputs,'source_group_counts':component_counts,'draft_sha256':manifest_hash,'configuration':proposal.get('config',{}),'source_manifest_sha256':proposal.get('source_manifest_sha256',''),'input_hashes':{'components':file_sha256(components) if Path(components).is_file() else '', 'hash_csv':file_sha256(hash_csv) if Path(hash_csv).is_file() else '', 'reviews':file_sha256(reviews) if reviews and Path(reviews).is_file() else '', 'cross_media_review':file_sha256(cross_media_review) if cross_media_review and Path(cross_media_review).is_file() else ''},'active_records':expected,'required_cross_dataset_pairs':[list(pair) for pair in sorted(required_dataset_pairs)],'active_source_components':len(active_components),'coverage':[{'partition':p,'dataset':d,'method':m,'label':l,'count':n} for (p,d,m,l),n in sorted(counts.items())],'group_overlap':overlaps,'hash_rows':loaded,'independently_detected_cross_partition_hash_groups':hash_overlap_count,'limitations':['Best-effort source components include suspected filename links; unknown relationships remain unverified','Exact-byte hashing cannot detect all recompressed media or derived video frames','Person-identity independence is not established by this release gate','DF40 testing-release data are repurposed; results are not an official DF40 benchmark']}
    db.close();return report


def promote(draft,report,components,reviews,hash_csv,output):
    if not report.get('ready'):raise ValueError('Release checks failed')
    for path,digest in report.get('verified_inputs',{}).items():
        if file_sha256(path)!=digest:raise ValueError('Review input changed after validation: '+path)
    if file_sha256(draft)!=report['draft_sha256'] or file_sha256(components)!=report['input_hashes']['components'] or (reviews and file_sha256(reviews)!=report['input_hashes']['reviews']) or file_sha256(hash_csv)!=report['input_hashes']['hash_csv']:raise ValueError('Inputs changed after validation')
    output=Path(output);validate_output(output.parent)
    if output.exists():raise FileExistsError('Release manifest already exists; choose a new version')
    output.parent.mkdir(parents=True,exist_ok=True);mapping={}
    with Path(components).open(newline='',encoding='utf-8') as f:
        for r in csv.DictReader(f):mapping[r['group_id']]=r['component_id']
    hashes=sqlite3.connect('');hashes.execute('CREATE TABLE hashes (path TEXT PRIMARY KEY,digest TEXT,size INTEGER,mtime_ns INTEGER)')
    with Path(hash_csv).open(newline='',encoding='utf-8') as f:
        for item in csv.DictReader(f):hashes.execute('INSERT INTO hashes VALUES (?,?,?,?)',(str(Path(item['absolute_path']).resolve()).casefold(),item['sha256'],int(item['size']),int(item['mtime_ns'])))
    temporary=output.with_name(output.name+'.partial')
    with Path(draft).open(newline='',encoding='utf-8') as src,temporary.open('x',newline='',encoding='utf-8') as dest:
        reader=csv.DictReader(src);fields=list(reader.fieldnames or [])
        for field in ('pre_release_group_id','release_source_component','sha256'):
            if field not in fields:fields.append(field)
        writer=csv.DictWriter(dest,fieldnames=fields);writer.writeheader()
        for row in reader:
            part=row.get('proposed_partition','quarantine');row['project_partition']=part if part in ACTIVE else 'quarantine';row['proposal_status']='RELEASED_BEST_EFFORT' if report.get('policy')=='best_effort' else 'RELEASED_AFTER_REVIEW'
            if part in ACTIVE:
                row['pre_release_group_id']=row.get('group_id','');row['group_id']=mapping[row['group_id']];row['release_source_component']=row['group_id']
                found=hashes.execute('SELECT digest,size,mtime_ns FROM hashes WHERE path=?',(str(Path(row['absolute_path']).resolve()).casefold(),)).fetchone()
                if not found:raise ValueError('Source hash missing during promotion')
                stat=Path(row['absolute_path']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(found[1],found[2]):raise ValueError('Source changed during promotion')
                row['sha256']=found[0]
            writer.writerow(row)
        dest.flush();os.fsync(dest.fileno())
    hashes.close()
    # On Windows rename is atomic and refuses an existing destination. On POSIX,
    # hard-link creation gives the equivalent no-overwrite guarantee.
    if os.name=='nt':os.rename(temporary,output)
    else:os.link(temporary,output);temporary.unlink()
    return file_sha256(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('draft','proposal-report','components','component-report','hash-csv','hash-report','output','report'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--reviews',type=Path);p.add_argument('--cross-media-review',type=Path);p.add_argument('--best-effort',action='store_true')
    a=p.parse_args();validate_output(a.output.parent);validate_output(a.report.parent)
    if not a.best_effort and (not a.reviews or not a.cross_media_review):p.error('Strict policy requires --reviews and --cross-media-review')
    result=inspect(a.draft,a.proposal_report,a.components,a.component_report,a.reviews,a.hash_csv,a.hash_report,a.cross_media_review,practical=a.best_effort)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    if result['ready']:
        try:result['release_sha256']=promote(a.draft,result,a.components,a.reviews,a.hash_csv,a.output)
        except (OSError,ValueError) as exc:result['ready']=False;result['blockers'].append(str(exc))
    a.report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'ready':result['ready'],'blockers':result['blockers']},indent=2))
    if not result['ready']:raise SystemExit(2)

if __name__=='__main__':main()
