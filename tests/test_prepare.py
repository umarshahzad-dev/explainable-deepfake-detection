import csv
import json
import shutil
from pathlib import Path
import pytest
from PIL import Image
from deepfake_data.extract import ExtractionConfig
from deepfake_data.prepare import read_release, run
from deepfake_data.split import file_sha256


def fixture_release(tmp_path):
    rows=[]
    for dataset,media in [('ffpp','video'),('celebdf','video'),('df40','image')]:
        source=tmp_path/(dataset+'.bin');source.write_bytes(dataset.encode())
        rows.append({'dataset':dataset,'method':'method','media_type':media,'absolute_path':str(source),'relative_path':dataset+'.bin','group_id':dataset,'project_partition':'train','label':'0','proposal_status':'RELEASED_BEST_EFFORT'})
    manifest=tmp_path/'release.csv'
    with manifest.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return manifest,rows


def test_preparation_resume_organization_and_failure(tmp_path,monkeypatch):
    import deepfake_data.prepare as module
    monkeypatch.setattr(module.os,'replace',lambda src,dst:shutil.copyfile(src,dst))
    manifest,rows=fixture_release(tmp_path); output=tmp_path/'out'; calls=[]; interrupted=[True]
    def fake_extract(source,folder,detector,*,config,provenance,image_output):
        calls.append(provenance['dataset']);folder.mkdir(parents=True,exist_ok=True);image_output.mkdir(parents=True,exist_ok=True)
        if interrupted[0] and provenance['dataset']=='celebdf':interrupted[0]=False;raise RuntimeError('interrupted')
        config_data={'version':2,'video':str(source),'parameters':{'size':299},'runtime':{},'provenance':provenance}
        result={'config':config_data,'frames':[{'file':str((image_output/'frame.png').resolve()),'frame_index':1}],'zero_yield':False,'rejection_counts':{}}
        Image.new('RGB',(299,299),'red' if provenance['dataset']=='ffpp' else 'blue').save(image_output/'frame.png')
        (folder/'state.json').write_text('{}');(folder/'complete.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(module,'extract_video',fake_extract)
    with pytest.raises(RuntimeError):run(manifest,output,expected_sha=file_sha256(manifest),smoke=True,detector=object(),check_cuda=False,min_free=0)
    assert json.loads((output/'.state/status.json').read_text())['status']=='failed'
    # Still fixture needs a valid encoded source for the actual still processor.
    Image.new('RGB',(20,20),'green').save(rows[2]['absolute_path'],format='PNG')
    result=run(manifest,output,expected_sha=file_sha256(manifest),smoke=True,detector=object(),check_cuda=False,min_free=0)
    assert result['status']=='complete' and result['image_count']==3
    with (output/'images.csv').open(newline='',encoding='utf-8') as f:images=list(csv.DictReader(f))
    assert {r['dataset'] for r in images}=={'ffpp','celebdf','df40'}
    assert all(Path(r['input_path']).is_relative_to(output/'train'/'fake') and len(r['output_sha256'])==64 for r in images)
    assert all(r['group_id']==r['dataset'] and r['project_partition']=='train' for r in images)
    with pytest.raises(ValueError,match='Incompatible preparation configuration'):
        run(manifest,output,expected_sha=file_sha256(manifest),smoke=True,detector=object(),check_cuda=False,min_free=0,config=ExtractionConfig(max_frames=1))


def test_release_rejects_collision_and_tampering(tmp_path):
    manifest,rows=fixture_release(tmp_path)
    with pytest.raises(ValueError,match='SHA-256'):read_release(manifest,'0'*64)
    with manifest.open('a',newline='',encoding='utf-8') as f:csv.DictWriter(f,fieldnames=list(rows[0])).writerow(rows[0])
    with pytest.raises(ValueError,match='colliding'):read_release(manifest,file_sha256(manifest))
