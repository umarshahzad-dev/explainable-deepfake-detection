import csv
import json
import shutil
from pathlib import Path
import pytest
from deepfake_data.hash_sources import run
from deepfake_data.release import inspect,promote
from deepfake_data.split import file_sha256


def test_release_fails_closed_without_evidence(tmp_path):
    draft=tmp_path/'draft.csv'
    with draft.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','group_id','label','project_partition','proposed_partition','proposal_status']);w.writeheader();w.writerow({'dataset':'ffpp','group_id':'g','label':'0','project_partition':'quarantine','proposed_partition':'train','proposal_status':'DRAFT_NOT_APPROVED'})
    missing=tmp_path/'missing'
    report=inspect(draft,missing,missing,missing,missing,missing,missing,missing)
    assert not report['ready'] and any('hash' in x.lower() for x in report['blockers'])
    with pytest.raises(ValueError):promote(draft,report,missing,missing,missing,tmp_path/'final.csv')
    assert not (tmp_path/'final.csv').exists()


def test_reviewed_release_is_frozen_and_contains_source_hashes(tmp_path,monkeypatch):
    import deepfake_data.release as release
    rows=[]
    for part,dataset in [('train','ffpp'),('train','celebdf'),('train','df40'),('val','ffpp'),('calibration','ffpp'),('test','ffpp'),('test','celebdf'),('test','df40')]:
        for label in ['0','1']:
            name=f'{part}-{dataset}-{label}';source=tmp_path/(name+'.bin');source.write_bytes(name.encode())
            rows.append({'dataset':dataset,'method':'midjourney' if dataset=='df40' else 'original','label':label,'group_id':name,'proposed_source_group':name,'project_partition':'quarantine','proposed_partition':part,'proposal_status':'DRAFT_NOT_APPROVED','absolute_path':str(source),'relative_path':name,'original_partition':'','metadata_status':'matched' if dataset=='df40' else 'path_rule'})
    draft=tmp_path/'draft.csv';fields=list(rows[0])
    with draft.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps({'status':'DRAFT_BLOCKED_NOT_FOR_EXTRACTION_OR_TRAINING','draft_manifest_sha256':file_sha256(draft),'source_manifest_sha256':'source','config':{'seed':42,'components_used':True}}))
    components=tmp_path/'components.csv'
    with components.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['group_id','component_id']);w.writeheader();w.writerows({'group_id':r['group_id'],'component_id':r['group_id']} for r in rows)
    component_report=tmp_path/'component-report.json';component_report.write_text(json.dumps({'manifest_sha256':'source','component_csv_sha256':file_sha256(components)}))
    evidence=tmp_path/'evidence.txt';evidence.write_text('Reviewed synthetic fixture')
    reviews=tmp_path/'reviews.csv'
    with reviews.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['component_id','decision','reviewer','evidence_path','evidence_sha256']);w.writeheader();w.writerows({'component_id':r['group_id'],'decision':'verified','reviewer':'test','evidence_path':str(evidence),'evidence_sha256':file_sha256(evidence)} for r in rows)
    hashes=tmp_path/'hashes.csv';hash_report=tmp_path/'hashes.json';run(draft,tmp_path/'cache.sqlite',hashes,hash_report,limit=None,use_proposals=True)
    cross=tmp_path/'cross.json';cross.write_text(json.dumps({'status':'reviewed_complete','scope':'all_active_cross_partition_source_pairs','covered_cross_dataset_pairs':[['ffpp','celebdf'],['ffpp','df40'],['celebdf','df40']],'manifest_sha256':file_sha256(draft),'component_csv_sha256':file_sha256(components),'unresolved_count':0,'reviewer':'test','reviewed_components':[r['group_id'] for r in rows],'evidence_files':[{'path':str(evidence),'sha256':file_sha256(evidence)}]}))
    report=inspect(draft,proposal,components,component_report,reviews,hashes,hash_report,cross)
    assert report['ready'],report['blockers']
    best_effort=inspect(draft,proposal,components,component_report,None,hashes,hash_report,None,practical=True)
    assert best_effort['ready'],best_effort['blockers']
    assert any('identity independence' in item for item in best_effort['limitations'])
    original_hash_csv=hashes.read_bytes();original_hash_report=hash_report.read_text()
    with hashes.open(newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f);hash_fields=reader.fieldnames;hash_rows=list(reader)
    train=next(row for row in hash_rows if row['partition']=='train');test=next(row for row in hash_rows if row['partition']=='test');test['sha256']=train['sha256']
    with hashes.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=hash_fields);writer.writeheader();writer.writerows(hash_rows)
    changed_report=json.loads(original_hash_report);changed_report['hash_csv_sha256']=file_sha256(hashes);changed_report['cross_partition_exact_hash_overlaps']={};hash_report.write_text(json.dumps(changed_report))
    detected=inspect(draft,proposal,components,component_report,reviews,hashes,hash_report,cross)
    assert not detected['ready'] and any('exact source hashes cross' in reason for reason in detected['blockers'])
    detected_best_effort=inspect(draft,proposal,components,component_report,None,hashes,hash_report,None,practical=True)
    assert not detected_best_effort['ready'] and any('exact source hashes cross' in reason for reason in detected_best_effort['blockers'])
    hashes.write_bytes(original_hash_csv);hash_report.write_text(original_hash_report)
    cross_data=json.loads(cross.read_text());cross_data['scope']='partial';cross.write_text(json.dumps(cross_data))
    incomplete=inspect(draft,proposal,components,component_report,reviews,hashes,hash_report,cross)
    assert not incomplete['ready'] and any('cross-media' in reason for reason in incomplete['blockers'])
    cross_data['scope']='all_active_cross_partition_source_pairs';cross.write_text(json.dumps(cross_data))
    report=inspect(draft,proposal,components,component_report,reviews,hashes,hash_report,cross)
    evidence.write_text('Modified evidence')
    with pytest.raises(ValueError):promote(draft,report,components,reviews,hashes,tmp_path/'invalid.csv')
    assert not (tmp_path/'invalid.csv').exists()
    evidence.write_text('Reviewed synthetic fixture')
    monkeypatch.setattr(release.os,'rename',lambda src,dst:shutil.copyfile(src,dst))
    result=tmp_path/'final.csv';sha=promote(draft,report,components,reviews,hashes,result)
    assert sha==file_sha256(result)
    with result.open(newline='',encoding='utf-8') as f:out=list(csv.DictReader(f))
    assert all(len(row['sha256'])==64 and row['proposal_status']=='RELEASED_AFTER_REVIEW' for row in out)
    with pytest.raises(FileExistsError):promote(draft,report,components,reviews,hashes,result)


def test_review_template_cannot_overwrite_existing_decisions(tmp_path):
    from deepfake_data.release_review_template import generate
    source=tmp_path/'draft.csv'
    with source.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['group_id','dataset','proposed_partition']);w.writeheader();w.writerow({'group_id':'g','dataset':'ffpp','proposed_partition':'train'})
    components=tmp_path/'components.csv'
    with components.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['group_id','component_id']);w.writeheader();w.writerow({'group_id':'g','component_id':'c'})
    report=tmp_path/'components.json';report.write_text(json.dumps({'component_csv_sha256':file_sha256(components)}))
    output=tmp_path/'review';generate(source,components,report,output)
    target=output/'component-reviews.csv';before=target.read_bytes()
    with pytest.raises(FileExistsError):generate(source,components,report,output)
    assert target.read_bytes()==before
