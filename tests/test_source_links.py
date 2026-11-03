import csv
import json
from pathlib import Path
from deepfake_data.source_links import dfd_components,enrich
from deepfake_data.readiness import audit_manifest
from deepfake_data.training_manifest import build_rows


def test_actor_components_and_missing_target(tmp_path):
    root=tmp_path/'ffpp'
    for name in ['01__scene.mp4','02__scene.mp4','03__other.mp4']:
        p=root/'original_sequences/actors/c23/videos'/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    for name in ['01_02__scene__ABCDEFGH.mp4','03_02__missing__ABCDEFGH.mp4']:
        p=root/'manipulated_sequences/DeepFakeDetection/c23/videos'/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    mapping,report=dfd_components(root)
    assert mapping[('original','01__scene')]==mapping[('original','02__scene')]==mapping[('DeepFakeDetection','01_02__scene__ABCDEFGH')]
    assert ('DeepFakeDetection','03_02__missing__ABCDEFGH') not in mapping
    assert report['issues']['target_original_missing']==1


def test_calibration_leakage_and_training_manifest(tmp_path):
    source=tmp_path/'split.csv'
    with source.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['dataset','media_type','label','group_id','project_partition']);w.writeheader();w.writerows([{'dataset':'ffpp','media_type':'video','label':'0','group_id':'g','project_partition':'train'},{'dataset':'ffpp','media_type':'video','label':'1','group_id':'g','project_partition':'calibration'}])
    assert 'g' in audit_manifest(source)['group_overlap']
    image={'dataset':'df40','media_type':'image','label':'0','absolute_path':'x.png','project_partition':'calibration'}
    output,skipped=build_rows([image],tmp_path)
    assert not skipped and output[0]['project_partition']=='calibration'


def test_combined_draft_has_no_active_partitions_and_preserves_official_holdout(tmp_path):
    from deepfake_data.combined_draft import build,celeb_source_key
    assert celeb_source_key('id0_id1_0000')=='id0_0000'
    root=tmp_path/'ffpp';video=root/'original_sequences/youtube/c23/videos/001.mp4';video.parent.mkdir(parents=True);video.touch()
    fields=['dataset','method','label','group_id','group_status','sample_id','source_domain','relative_path','absolute_path','media_type','original_partition','project_partition','metadata_status']
    rows=[{'dataset':'ffpp','method':'original','label':'1','group_id':'ffpp:youtube:001','group_status':'verified','sample_id':'001','relative_path':'001.mp4','media_type':'video'}, {'dataset':'celebdf','method':'original','label':'1','group_id':'a','sample_id':'id0_0000','relative_path':'Celeb-real/id0_0000.mp4','media_type':'video','original_partition':'official_test'}, {'dataset':'celebdf','method':'fake','label':'0','group_id':'b','sample_id':'id0_id1_0000','relative_path':'Celeb-synthesis/id0_id1_0000.mp4','media_type':'video'}, {'dataset':'df40','method':'x','label':'0','group_id':'d','sample_id':'z','source_domain':'unknown','relative_path':'x/a.png','media_type':'image','metadata_status':'partition_conflict'}, {'dataset':'df40','method':'x','label':'','group_id':'e','sample_id':'z','source_domain':'unknown','relative_path':'x/b.png','media_type':'image'}]
    source=tmp_path/'source.csv'
    with source.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    output=tmp_path/'draft.csv';report=build(source,output,tmp_path/'report.json',root,per_group=1,per_method_label=2)
    with output.open(newline='',encoding='utf-8') as f:result=list(csv.DictReader(f))
    assert all(row['project_partition']=='quarantine' for row in result)
    assert result[1]['proposal_reason']=='official_celebdf_test_candidate_unverified' and result[1]['proposed_partition']=='test'
    assert result[2]['proposal_reason']=='suspected_shared_official_test_source'
    assert result[3]['proposed_partition']=='train' and result[4]['proposal_reason']=='unknown_label'
    assert report['status'].startswith('DRAFT_BLOCKED')


def test_train_sampling_weights_balance_datasets_and_exclude_test(tmp_path):
    from deepfake_data.sampling_weights import create
    manifest=tmp_path/'images.csv';fields=['project_partition','dataset','method','label','group_id','input_path']
    rows=[]
    for dataset in ['ffpp','celebdf','df40']:
        for label in ['0','1']:
            rows.append({'project_partition':'train','dataset':dataset,'method':'m'+label,'label':label,'group_id':'g'+dataset+label,'input_path':dataset+label+'.png'})
    rows.append({'project_partition':'test','dataset':'ffpp','method':'m0','label':'0','group_id':'test','input_path':'test.png'})
    with manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    result=create(manifest,tmp_path/'weights.csv',tmp_path/'weights.json')
    assert result['training_rows']==6 and abs(result['weight_total']-1)<1e-9
    assert all(abs(x-1/3)<1e-9 for x in result['dataset_weight_totals'].values())
    with (tmp_path/'weights.csv').open(newline='',encoding='utf-8') as f:assert len(list(csv.DictReader(f)))==6


def test_draft_rows_cannot_be_extracted_or_enter_image_manifest(tmp_path):
    import pytest
    from deepfake_data.extract import select_videos
    from deepfake_data.training_manifest import build_rows
    video={'project_partition':'train','media_type':'video','proposal_status':'DRAFT_NOT_APPROVED'}
    with pytest.raises(ValueError):select_videos([video],'train')
    image={'project_partition':'train','media_type':'image','proposal_status':'DRAFT_NOT_APPROVED','label':'0','absolute_path':'x.png'}
    rows,skipped=build_rows([image],tmp_path)
    assert rows==[] and skipped[0]['reason']=='draft_not_approved'


def test_both_celeb_identity_candidates_protect_official_sources():
    from deepfake_data.combined_draft import celeb_source_keys
    assert celeb_source_keys('id0_id1_0000')=={'id0_0000','id1_0000'}
    assert celeb_source_keys('id1_0000')=={'id1_0000'}


def test_draft_quota_is_deterministic_and_not_active(tmp_path):
    from deepfake_data.combined_draft import build
    root=tmp_path/'ffpp';root.mkdir();source=tmp_path/'source.csv';fields=['dataset','method','label','group_id','group_status','sample_id','source_domain','relative_path','absolute_path','media_type','metadata_status','original_partition','project_partition']
    rows=[{'dataset':'df40','method':'m','label':'0','group_id':f'g{i//2}','sample_id':str(i),'source_domain':'unknown','relative_path':f'm/{i}.png','media_type':'image','metadata_status':'partition_conflict'} for i in range(6)]
    with source.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    out1=tmp_path/'a.csv';out2=tmp_path/'b.csv'
    build(source,out1,tmp_path/'a.json',root,per_group=1,per_method_label=2)
    build(source,out2,tmp_path/'b.json',root,per_group=1,per_method_label=2)
    assert out1.read_bytes()==out2.read_bytes()
    with out1.open(newline='',encoding='utf-8') as f:r=list(csv.DictReader(f))
    selected=[x for x in r if x['proposed_partition']=='train']
    assert len(selected)==2 and len({x['group_id'] for x in selected})==2
    assert all(x['project_partition']=='quarantine' for x in r)


def test_relationship_audit_identifies_duplicate_test_val_records(tmp_path):
    from deepfake_data.relationship_audit import audit
    celeb=tmp_path/'celebdf/Celeb-DF-v2';real=celeb/'Celeb-real';fake=celeb/'Celeb-synthesis';real.mkdir(parents=True);fake.mkdir()
    (real/'id0_0000.mp4').touch();(real/'id1_0000.mp4').touch();(fake/'id0_id1_0000.mp4').touch()
    (celeb/'List_of_testing_videos.txt').write_text('1 Celeb-real/id0_0000.mp4\n')
    (tmp_path/'df40/extracted/e4s').mkdir(parents=True);meta=tmp_path/'df40/metadata/dataset_json';meta.mkdir(parents=True)
    sample={'clip':{'label':'e4s_Fake','frames':['x']}}
    (meta/'e4s_cdf.json').write_text(json.dumps({'e4s_cdf':{'e4s_Fake':{'train':{},'test':sample,'val':sample}}}))
    report=audit(tmp_path)
    assert report['df40_test_val_comparison']['same_complete_records']==1
    assert report['celebdf_filename_evidence']['non_official_withheld_by_either_candidate_id']==1


def test_bounded_source_hashing_detects_cross_partition_duplicate_and_resumes(tmp_path):
    from deepfake_data.hash_sources import run
    fields=['dataset','relative_path','absolute_path','group_id','project_partition']
    source=tmp_path/'manifest.csv'; rows=[]
    for index,part in enumerate(['train','test']):
        path=tmp_path/f'{index}.png';path.write_bytes(b'same');rows.append({'dataset':'df40','relative_path':path.name,'absolute_path':str(path),'group_id':f'g{index}','project_partition':part})
    with source.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    report=run(source,tmp_path/'cache.sqlite',tmp_path/'hashes.csv',tmp_path/'report.json',limit=None)
    assert report['complete'] and len(report['cross_partition_exact_hash_overlaps'])==1
    again=run(source,tmp_path/'cache.sqlite',tmp_path/'hashes.csv',tmp_path/'report.json',limit=1)
    assert not again['complete'] and again['cache_reused']==1 and again['attempted']==1


def test_stratified_hashing_covers_train_and_test(tmp_path):
    from deepfake_data.hash_sources import run
    fields=['dataset','label','relative_path','absolute_path','group_id','project_partition']
    source=tmp_path/'manifest.csv';rows=[]
    for index,part in enumerate(['train','train','test','test']):
        path=tmp_path/f'sample{index}.png';path.write_bytes(str(index).encode());rows.append({'dataset':'df40','label':'0','relative_path':path.name,'absolute_path':str(path),'group_id':f'g{index}','project_partition':part})
    with source.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    output=tmp_path/'hashes.csv';result=run(source,tmp_path/'cache.sqlite',output,tmp_path/'report.json',limit=2,stratified=True)
    with output.open(newline='',encoding='utf-8') as f:parts={row['partition'] for row in csv.DictReader(f)}
    assert parts=={'train','test'} and result['attempted']==2 and not result['complete']


def test_midjourney_and_actor_components_reserved_not_forced_to_test(tmp_path):
    from deepfake_data.combined_draft import build
    root=tmp_path/'ffpp';root.mkdir();source=tmp_path/'input.csv'
    fields=['dataset','method','label','group_id','group_status','sample_id','source_domain','relative_path','absolute_path','media_type','metadata_status','original_partition','project_partition']
    rows=[{'dataset':'df40','method':'MidJourney','label':'0','group_id':'df40:m:one','sample_id':'one','relative_path':'m/a.png','media_type':'image','metadata_status':'partition_conflict'},{'dataset':'ffpp','method':'DeepFakeDetection','label':'0','group_id':'ffpp:dfd:actors:05','relative_path':'dfd/a.mp4','media_type':'video'},{'dataset':'ffpp','method':'DeepFakeDetection','label':'1','group_id':'ffpp:dfd:actors:10','relative_path':'dfd/b.mp4','media_type':'video'}]
    with source.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    output=tmp_path/'draft.csv';build(source,output,tmp_path/'report.json',root)
    with output.open(newline='',encoding='utf-8') as f:result=list(csv.DictReader(f))
    assert all(row['proposed_partition']=='quarantine' and row['project_partition']=='quarantine' for row in result)
    assert result[0]['proposal_reason']=='df40_modern_evaluation_candidate_unverified'
    assert all(row['proposal_reason']=='dfd_actor_evaluation_candidate_unverified' for row in result[1:])
