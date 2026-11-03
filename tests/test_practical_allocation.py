import csv
import json
from deepfake_data.hash_sources import run
from deepfake_data.practical_allocation import build, supplement, quarantine_conflicts
from deepfake_data.split import file_sha256


def test_practical_allocation_preserves_quarantine_and_blocks_hash_leakage(tmp_path):
    rows = []
    for name, dataset, part, contents in [('ff','ffpp','train',b'same'), ('test','celebdf','test',b'same'), ('other','df40','train',b'other'), ('excluded','df40','quarantine',b'ignored')]:
        media=tmp_path/name; media.write_bytes(contents)
        rows.append({'dataset':dataset,'group_id':name,'proposed_source_group':name,'absolute_path':str(media),'project_partition':'quarantine','proposed_partition':part,'proposal_status':'DRAFT_NOT_APPROVED','original_partition':'official_test' if name=='test' else '', 'method':'m','label':'0'})
    draft=tmp_path/'draft.csv'
    with draft.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps({'draft_manifest_sha256':file_sha256(draft),'source_manifest_sha256':'source','config':{'components_used':True}}))
    components=tmp_path/'components.csv'
    with components.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['group_id','component_id']);w.writeheader();w.writerows({'group_id':r['group_id'],'component_id':r['group_id']} for r in rows)
    component_report=tmp_path/'components.json';component_report.write_text(json.dumps({'manifest_sha256':'source','component_csv_sha256':file_sha256(components)}))
    hashes=tmp_path/'hashes.csv';hash_report=tmp_path/'hashes.json'
    run(draft,tmp_path/'cache.sqlite',hashes,hash_report,limit=None,use_proposals=True)
    output=tmp_path/'out.csv';report=tmp_path/'report.json'
    result=build(draft,proposal,components,component_report,hashes,hash_report,output,report)
    assert result['exact_conflict_components_quarantined']==2
    with output.open(newline='',encoding='utf-8') as f: output_rows={r['group_id']:r for r in csv.DictReader(f)}
    assert output_rows['ff']['proposed_partition']=='quarantine'
    assert output_rows['test']['proposed_partition']=='quarantine'
    assert output_rows['excluded']['proposed_partition']=='quarantine'
    assert all(r['project_partition']=='quarantine' for r in output_rows.values())
    resolved=tmp_path/'resolved.csv'
    result=quarantine_conflicts(draft,proposal,hashes,hash_report,resolved,tmp_path/'resolved.json')
    assert result['exact_conflict_hashes_quarantined']==1
    with resolved.open(newline='',encoding='utf-8') as f: resolved_rows={r['group_id']:r for r in csv.DictReader(f)}
    assert resolved_rows['ff']['proposed_partition']=='quarantine'
    assert resolved_rows['test']['proposed_partition']=='quarantine'
    assert resolved_rows['other']['proposed_partition']=='train'


def test_supplement_only_colocates_reviewable_ff_sources(tmp_path):
    fields=['dataset','group_id','proposed_source_group','project_partition','proposed_partition','proposal_reason','metadata_status','label','method']
    rows=[dict(zip(fields,values)) for values in [
        ('ffpp','ff','component','quarantine','test','source','matched','1','original'),
        ('df40','df','component','quarantine','quarantine','suspected_ffpp_heldout_overlap','matched','0','DiT'),
        ('df40','unknown','unknown','quarantine','quarantine','unknown_label','matched','','DiT'),
    ]]
    draft=tmp_path/'draft.csv'
    with draft.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    proposal=tmp_path/'proposal.json';proposal.write_text(json.dumps({'draft_manifest_sha256':file_sha256(draft),'config':{}}))
    components=tmp_path/'components.csv'
    with components.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['group_id','component_id']);w.writeheader();w.writerows([{'group_id':'ff','component_id':'component'},{'group_id':'df','component_id':'component'},{'group_id':'unknown','component_id':'unknown'}])
    output=tmp_path/'out.csv';result=supplement(draft,proposal,components,output,tmp_path/'report.json')
    assert result['supplemented_records']==1
    with output.open(newline='',encoding='utf-8') as f: out={r['group_id']:r for r in csv.DictReader(f)}
    assert out['df']['proposed_partition']=='test' and out['df']['project_partition']=='quarantine'
    assert out['unknown']['proposed_partition']=='quarantine'
