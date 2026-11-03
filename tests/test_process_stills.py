import shutil
from pathlib import Path
from PIL import Image
import pytest
from deepfake_data.process_stills import process
from deepfake_data.training_manifest import build_rows
from deepfake_data.extract import output_key


def test_jpeg_named_png_normalized_and_provenance(tmp_path,monkeypatch):
    import deepfake_data.process_stills as module
    monkeypatch.setattr(module.os,'replace',lambda src,dst:shutil.copyfile(src,dst))
    source=tmp_path/'sample.png';Image.new('RGB',(40,20),'red').save(source,format='JPEG')
    row={'dataset':'df40','method':'x','relative_path':'x/sample.png','group_id':'group','project_partition':'train','label':'0','media_type':'image','absolute_path':str(source)}
    root=tmp_path/'out';out=root/output_key(row)
    result=process(row,out)
    assert result['source_format']=='JPEG' and result['config']['provenance']['group_id']=='group'
    with Image.open(out/result['frames'][0]['file']) as image:assert image.size==(299,299)
    assert process(row,out)==result
    with pytest.raises(ValueError):process(row,out,size=256)
    inputs,skips=build_rows([row],tmp_path,stills_root=root)
    assert not skips and inputs[0]['project_partition']=='train' and inputs[0]['group_id']=='group'


def test_gif_frames_keep_one_source_and_calibration(tmp_path,monkeypatch):
    import deepfake_data.process_stills as module
    monkeypatch.setattr(module.os,'replace',lambda src,dst:shutil.copyfile(src,dst))
    source=tmp_path/'sample.gif';a=Image.new('RGB',(20,20),'red');b=Image.new('RGB',(20,20),'blue');a.save(source,save_all=True,append_images=[b],duration=100,loop=0)
    row={'dataset':'df40','method':'x','relative_path':'x/sample.gif','group_id':'one-gif','project_partition':'calibration','label':'0','media_type':'gif','absolute_path':str(source)}
    root=tmp_path/'out';result=process(row,root/output_key(row));inputs,skips=build_rows([row],tmp_path,stills_root=root)
    assert result['source_frame_count']==2 and len(inputs)==2 and not skips
    assert all(x['group_id']=='one-gif' and x['project_partition']=='calibration' for x in inputs)


def test_interrupted_gif_reuses_completed_frame(tmp_path,monkeypatch):
    import deepfake_data.process_stills as module
    monkeypatch.setattr(module.os,'replace',lambda src,dst:shutil.copyfile(src,dst))
    source=tmp_path/'source.gif';a=Image.new('RGB',(20,20),'red');b=Image.new('RGB',(20,20),'blue');a.save(source,save_all=True,append_images=[b],duration=100,loop=0)
    row={'dataset':'df40','method':'x','relative_path':'source.gif','group_id':'g','project_partition':'train','label':'0','absolute_path':str(source)}
    original=Image.Image.save;calls=[];interrupt=[True]
    def save(image,fp,*args,**kwargs):
        calls.append(1)
        if len(calls)==2 and interrupt[0]:interrupt[0]=False;raise RuntimeError('interrupted')
        return original(image,fp,*args,**kwargs)
    monkeypatch.setattr(Image.Image,'save',save)
    with pytest.raises(RuntimeError):process(row,tmp_path/'checkpoint')
    result=process(row,tmp_path/'checkpoint')
    assert len(result['frames'])==2 and len(calls)==3


def test_prepared_validation_detects_cross_partition_content(tmp_path):
    import csv
    from deepfake_data.validate_prepared import validate
    src=tmp_path/'source.png';img=Image.new('RGB',(299,299),'red');img.save(src)
    target1=tmp_path/'a.png';target2=tmp_path/'b.png';img.save(target1);img.save(target2)
    path=tmp_path/'manifest.csv';fields=['dataset','method','label','group_id','project_partition','absolute_path','input_path','source_frame_index']
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows([{'dataset':'df40','method':'x','label':'0','group_id':'g','project_partition':'train','absolute_path':str(src),'input_path':str(target1),'source_frame_index':'0'},{'dataset':'df40','method':'x','label':'1','group_id':'g','project_partition':'test','absolute_path':str(src),'input_path':str(target2),'source_frame_index':'0'}])
    report=validate(path,decode=True,hash_outputs=True)
    assert not report['ready'] and report['issues']['duplicate_manifest_rows']==1
    assert report['issues']['group_leakage']==1 and report['issues']['exact_content_leakage']==1
