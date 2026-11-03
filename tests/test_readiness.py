import csv
import json
from deepfake_data.readiness import audit_manifest,assess
from deepfake_data.split import file_sha256,stream_split


def write_manifest(path,rows):
    fields=["dataset","method","media_type","label","source_domain","group_id","group_status","relationship_status","project_partition","sha256","absolute_path"]
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def test_gate_rejects_preview_and_detects_overlap(tmp_path):
    manifest=tmp_path/"manifest.csv"
    write_manifest(manifest,[{"dataset":"ffpp","media_type":"video","label":"0","group_id":"g","project_partition":"train","sha256":"same"},{"dataset":"ffpp","media_type":"video","label":"1","group_id":"g","project_partition":"test","sha256":"same"}])
    inventory=tmp_path/"inventory.json";inventory.write_text(json.dumps({"scan_limited":True,"counts":{"indexed:ffpp":2}}))
    report=assess(manifest,inventory,None,tmp_path,batch_videos=1)
    assert not report["extraction_ready"] and not report["training_evaluation_ready"]
    assert report["manifest"]["group_overlap"]["g"] == ["test","train"]
    assert "same" in report["manifest"]["hash_overlap"]
    assert "Not authorized" in report["overnight_extraction"]


def test_stream_split_is_deterministic_and_keeps_full_rows(tmp_path):
    source=tmp_path/"source.csv"
    write_manifest(source,[{"dataset":"ffpp","media_type":"video","label":"0","group_id":"g","group_status":"verified"},{"dataset":"ffpp","media_type":"video","label":"1","group_id":"g","group_status":"verified"},{"dataset":"df40","media_type":"image","label":"","group_id":"","group_status":"unresolved"}])
    one=tmp_path/"one.csv";two=tmp_path/"two.csv"
    a=stream_split(source,one);b=stream_split(source,two)
    assert one.read_bytes()==two.read_bytes() and a["source_manifest_sha256"]==file_sha256(source)
    with one.open(newline="",encoding="utf-8") as f: rows=list(csv.DictReader(f))
    assert rows[0]["project_partition"]==rows[1]["project_partition"]
    assert rows[2]["quarantine_reason"]=="unknown_label"


def test_readiness_does_not_trust_mismatched_report_hash(tmp_path,monkeypatch):
    import deepfake_data.readiness as readiness
    source=tmp_path/"source.csv";write_manifest(source,[{"dataset":"ffpp","media_type":"video","label":"0","group_id":"a","group_status":"verified"}])
    output=tmp_path/"split.csv";stream_split(source,output)
    inventory=tmp_path/"inventory.json";inventory.write_text(json.dumps({"scan_limited":False,"counts":{"indexed:ffpp":1},"manifest_sha256":"wrong"}))
    monkeypatch.setattr(readiness,"sample_audit",lambda root:{"images":1,"videos":1,"errors":[],"labels":{"ffpp:0":1},"zero_yield":0})
    report=assess(output,inventory,None,tmp_path,batch_videos=1,split_report=output.with_suffix(".report.json"))
    assert next(c for c in report["checks"] if c["check"]=="full_inventory")["status"]=="FAIL"


def test_reconciliation_queues_missing_and_unlisted(tmp_path):
    from deepfake_data.inventory import generate
    from deepfake_data.reconcile_details import generate as details
    configs=tmp_path/"configs";configs.mkdir();root=tmp_path/"data/raw/df40/extracted/e4s/cdf/frames/a";root.mkdir(parents=True);(root/"extra.png").touch()
    metadata=tmp_path/"data/raw/df40/metadata/dataset_json";metadata.mkdir(parents=True)
    payload={"e4s_cdf":{"e4s_Fake":{"test":{"a":{"label":"e4s_Fake","frames":["deepfakes_detection_datasets/DF40/e4s/cdf/frames/a/missing.png"]}}}}}
    (metadata/"e4s_cdf.json").write_text(json.dumps(payload));config=configs/"datasets.json";config.write_text(json.dumps({"raw_root":"data/raw","output_dir":"artifacts/inventory","datasets":{"df40":{"path":"df40/extracted","metadata_path":"df40/metadata/dataset_json"}}}))
    manifest,_=generate(config);report=details(manifest,config,tmp_path/"issues")
    assert report["counts"]["local:unlisted"]==1 and report["counts"]["json:no_matching_local_file"]==1


def test_finalize_reports_partition_conflicts(tmp_path,monkeypatch):
    import shutil
    import deepfake_data.finalize_inventory as module
    manifest=tmp_path/"inventory.csv"
    with manifest.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['dataset','method','media_type','label','source_domain','metadata_status','original_partition']);writer.writeheader();writer.writerow({'dataset':'df40','method':'e4s','media_type':'image','label':'0','metadata_status':'matched','original_partition':'test|val'})
    report=tmp_path/"report.json";report.write_text(json.dumps({'scan_limited':False,'errors':[],'counts':{'indexed:df40':1}}))
    monkeypatch.setattr(module.os,'replace',lambda src,dst:shutil.copyfile(src,dst))
    result=module.finalize(manifest,report)
    assert result['partition_conflicts']==1
    with manifest.open(newline='',encoding='utf-8') as f: assert next(csv.DictReader(f))['metadata_status']=='partition_conflict'
    assert json.loads(report.read_text())['manifest_sha256']==file_sha256(manifest)


def test_partition_conflict_quarantined_in_holdout():
    from deepfake_data.split import split_rows
    row={"dataset":"df40","label":"0","group_id":"g","group_status":"sample_verified","source_domain":"unknown","metadata_status":"partition_conflict","overlap_status":"reviewed_independent"}
    assert split_rows([row],holdout="ffpp",protocol="experimental")[0]["project_partition"]=="quarantine"


def test_stream_hash_detects_exact_cross_partition_duplicate(tmp_path):
    from deepfake_data.split import split_rows
    groups=[]
    for index in range(100):
        row={"group_id":f"g{index}","dataset":"ffpp","label":"0","group_status":"verified"}
        partition=split_rows([row])[0]["project_partition"]
        if not groups or partition != groups[0][1]:groups.append((row,partition))
        if len(groups)==2:break
    rows=[]
    for index,(row,_) in enumerate(groups):
        path=tmp_path/f"file{index}.mp4";path.write_bytes(b"same bytes");rows.append({**row,"media_type":"video","absolute_path":str(path)})
    source=tmp_path/"source.csv";write_manifest(source,rows)
    report=stream_split(source,tmp_path/"split.csv",hash_media=True)
    assert len(report["hash_overlap"])==1


def test_readiness_accepts_hashed_source_enrichment_chain(tmp_path,monkeypatch):
    import deepfake_data.readiness as readiness
    source=tmp_path/'source.csv';write_manifest(source,[{'dataset':'ffpp','media_type':'video','label':'0','group_id':'a','group_status':'verified'}])
    output=tmp_path/'split.csv';stream_split(source,output)
    inventory=tmp_path/'inventory.json';inventory.write_text(json.dumps({'scan_limited':False,'counts':{'indexed:ffpp':1},'manifest_sha256':'originalhash'}))
    links=tmp_path/'links.json';links.write_text(json.dumps({'source_manifest_sha256':'originalhash','output_manifest_sha256':file_sha256(source)}))
    monkeypatch.setattr(readiness,'sample_audit',lambda root:{'images':1,'videos':1,'errors':[],'labels':{'ffpp:0':1},'zero_yield':0})
    result=assess(output,inventory,None,tmp_path,batch_videos=1,split_report=output.with_suffix('.report.json'),source_links_report=links)
    assert next(c for c in result['checks'] if c['check']=='full_inventory')['status']=='PASS'
    assert not result['training_evaluation_ready']
