import csv
import json
from deepfake_data.split import file_sha256
from deepfake_data.source_components import build


def test_cross_dataset_components_are_explicitly_suspected(tmp_path):
    root=tmp_path/'ffpp';p=root/'original_sequences/youtube/c23/videos/001.mp4';p.parent.mkdir(parents=True);p.touch()
    source=tmp_path/'manifest.csv';fields=['group_id','dataset','method','source_domain','sample_id','group_status','relationship_status']
    rows=[{'group_id':'celebdf:id0_0000','dataset':'celebdf','sample_id':'id0_0000'}, {'group_id':'celebdf:id0_id1_0000','dataset':'celebdf','sample_id':'id0_id1_0000'}, {'group_id':'df40:e4s:cdf:a','dataset':'df40','method':'e4s','source_domain':'cdf','sample_id':'id0_id1_0000'}, {'group_id':'ffpp:youtube:001','dataset':'ffpp','sample_id':'001'}, {'group_id':'df40:x:ff:b','dataset':'df40','method':'x','source_domain':'ff','sample_id':'001'}]
    with source.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    report=build(source,root,tmp_path/'out')
    assert report['components_with_multiple_datasets']==2
    with (tmp_path/'out/source-edges.csv').open(newline='',encoding='utf-8') as f:edges=list(csv.DictReader(f))
    assert edges and all(row['status']=='suspected' for row in edges)
    with (tmp_path/'out/source-components.csv').open(newline='',encoding='utf-8') as f:components={r['group_id']:r for r in csv.DictReader(f)}
    assert components['celebdf:id0_0000']['component_id']==components['df40:e4s:cdf:a']['component_id']
    assert components['ffpp:youtube:001']['component_id']==components['df40:x:ff:b']['component_id']


def test_perceptual_hash_is_deterministic_and_not_identity_proof():
    import numpy as np
    from deepfake_data.cross_media_probe import dhash
    a=np.zeros((20,20,3),dtype=np.uint8);b=a.copy();b[:,:10]=255
    assert dhash(a)==dhash(a.copy())
    assert (dhash(a)^dhash(b)).bit_count()>0


def test_candidate_components_quarantine_suspected_train_to_official_test(tmp_path):
    import json,pytest
    from deepfake_data.combined_draft import build
    from deepfake_data.split import file_sha256
    source=tmp_path/'source.csv';fields=['dataset','label','group_id','sample_id','relative_path','original_partition','media_type']
    rows=[{'dataset':'celebdf','label':'1','group_id':'a','sample_id':'id0_0000','relative_path':'Celeb-real/id0_0000.mp4','original_partition':'official_test','media_type':'video'}, {'dataset':'celebdf','label':'0','group_id':'b','sample_id':'id9_id8_0005','relative_path':'Celeb-synthesis/id9_id8_0005.mp4','media_type':'video'}]
    with source.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    components=tmp_path/'components.csv'
    with components.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['group_id','component_id']);w.writeheader();w.writerows([{'group_id':'a','component_id':'c'},{'group_id':'b','component_id':'c'}])
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({'manifest_sha256':file_sha256(source),'component_csv_sha256':file_sha256(components)}))
    root=tmp_path/'ffpp';root.mkdir();out=tmp_path/'out.csv'
    build(source,out,tmp_path/'report.json',root,components=components,components_report=proof)
    with out.open(newline='',encoding='utf-8') as f:output=list(csv.DictReader(f))
    assert output[0]['proposed_partition']=='test' and output[1]['proposal_reason']=='suspected_component_reaches_evaluation'
    components.write_text('tampered')
    with pytest.raises(ValueError):build(source,tmp_path/'other.csv',tmp_path/'other.json',root,components=components,components_report=proof)


def test_peripheral_feature_correspondence_detects_same_image():
    import numpy as np
    from deepfake_data.cross_media_probe import peripheral_feature_support
    image=np.random.default_rng(42).integers(0,256,(256,256,3),dtype=np.uint8)
    result=peripheral_feature_support(image,image.copy())
    assert result['peripheral_matches']>=4 and result['geometric_inliers']>=4
