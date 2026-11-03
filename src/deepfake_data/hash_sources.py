"""Bounded, resumable exact-byte source hashing; never modifies source media."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter,defaultdict
from pathlib import Path
from .extract import validate_output

ACTIVE={"train","val","calibration","test"}


def run(manifest, cache, output, report, limit=100, use_proposals=False, stratified=False):
    if limit is not None and limit<1:raise ValueError("limit must be positive")
    from .split import file_sha256
    input_hash=file_sha256(manifest)
    for target in (Path(cache),Path(output),Path(report)):validate_output(target.parent)
    cache=Path(cache);output=Path(output);report=Path(report)
    for target in (cache,output,report):target.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(cache);db.execute("PRAGMA synchronous=NORMAL");db.execute("CREATE TABLE IF NOT EXISTS hashes (path TEXT PRIMARY KEY, size INTEGER, mtime_ns INTEGER, digest TEXT)")
    hashed=0;reused=0;attempted=0;eligible=0;errors=[];hash_parts=defaultdict(set);hash_paths=defaultdict(list); quotas=None;chosen=Counter()
    if stratified and limit is not None:
        available=Counter()
        with Path(manifest).open(newline="",encoding="utf-8") as src:
            for item in csv.DictReader(src):
                partition=item.get('proposed_partition','') if use_proposals else item.get('project_partition','')
                if partition in ACTIVE:available[(partition,item.get('dataset',''),item.get('label',''))]+=1
        quotas=Counter();remaining=limit
        while remaining and any(quotas[key]<available[key] for key in available):
            for key in sorted(available):
                if remaining and quotas[key]<available[key]:quotas[key]+=1;remaining-=1
    fields=["dataset","label","relative_path","absolute_path","group_id","partition","sha256","status","size","mtime_ns"]
    with Path(manifest).open(newline="",encoding="utf-8") as src,output.open('w',newline='',encoding='utf-8') as dst:
        writer=csv.DictWriter(dst,fieldnames=fields);writer.writeheader()
        for row in csv.DictReader(src):
            partition=row.get('proposed_partition','') if use_proposals else row.get('project_partition','')
            if partition not in ACTIVE:continue
            eligible+=1
            if limit is not None and attempted>=limit:continue
            bucket=(partition,row.get('dataset',''),row.get('label',''))
            if quotas is not None and chosen[bucket]>=quotas[bucket]:continue
            chosen[bucket]+=1;attempted+=1
            path=Path(row.get('absolute_path',''));record={'dataset':row.get('dataset',''),'label':row.get('label',''),'relative_path':row.get('relative_path',''),'absolute_path':str(path),'group_id':row.get('proposed_source_group' if use_proposals else 'group_id',''),'partition':partition,'sha256':'','status':'error','size':'','mtime_ns':''}
            try:
                stat=path.stat();record.update(size=stat.st_size,mtime_ns=stat.st_mtime_ns);cached=db.execute('SELECT size,mtime_ns,digest FROM hashes WHERE path=?',(str(path.resolve()),)).fetchone()
                if cached and cached[:2]==(stat.st_size,stat.st_mtime_ns):digest=cached[2];reused+=1
                else:
                    h=hashlib.sha256()
                    with path.open('rb') as f:
                        for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
                    after=path.stat()
                    if (after.st_size,after.st_mtime_ns)!=(stat.st_size,stat.st_mtime_ns):raise OSError('source changed while hashing')
                    digest=h.hexdigest();db.execute('INSERT OR REPLACE INTO hashes VALUES(?,?,?,?)',(str(path.resolve()),stat.st_size,stat.st_mtime_ns,digest));hashed+=1
                    if hashed % 50 == 0:db.commit()
                record.update(sha256=digest,status='ok');hash_parts[digest].add(partition);hash_paths[digest].append(str(path))
            except OSError as exc:record['status']='error: '+str(exc);errors.append({'path':str(path),'error':str(exc)})
            writer.writerow(record)
    db.commit();db.close();overlap={h:{'partitions':sorted(parts),'examples':hash_paths[h][:5]} for h,parts in hash_parts.items() if len(parts)>1}
    result={'proposal_mode':use_proposals,'stratified':stratified,'eligible_records':eligible,'newly_hashed':hashed,'cache_reused':reused,'attempted':attempted,'errors':errors,'complete':hashed+reused==eligible and not errors,'cross_partition_exact_hash_overlaps':overlap,'limitations':['Hashes compare exact file bytes, not visually similar media or video frames against still images','Cache reuse checks size and modification time, not protection from deliberate timestamp-preserving changes','A complete hash check does not prove source or identity independence']}
    final_hash=file_sha256(manifest)
    result['manifest_changed_during_scan']=input_hash!=final_hash
    if result['manifest_changed_during_scan']:result['complete']=False
    result['manifest_sha256']=input_hash;result['hash_csv_sha256']=file_sha256(output)
    report.write_text(json.dumps(result,indent=2),encoding='utf-8');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--cache',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--report',required=True,type=Path);p.add_argument('--max-files',type=int,default=100);p.add_argument('--all',action='store_true');p.add_argument('--use-proposals',action='store_true');p.add_argument('--stratified',action='store_true');a=p.parse_args()
    r=run(a.manifest,a.cache,a.output,a.report,None if a.all else a.max_files,a.use_proposals,a.stratified);print(json.dumps(r,indent=2))
    if r['errors'] or r['cross_partition_exact_hash_overlaps']:raise SystemExit(2)

if __name__=='__main__':main()
