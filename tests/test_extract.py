import json
import shutil
import numpy as np
import pytest
from deepfake_data.extract import ExtractionConfig, best_candidate, candidate_segments, evaluate_candidate, extract_video, read_candidate_frames, square_crop, square_geometry


def test_square_geometry_and_boundary_padding():
    box, fraction=square_geometry([80,60,140,100],300,200,1.5)
    assert box[2]-box[0] == box[3]-box[1] == 90 and fraction == 0
    box,fraction=square_geometry([-10,10,50,70],100,100,1.5)
    assert box[2]-box[0] == box[3]-box[1] and 0 < fraction < 1
    frame=np.zeros((200,300,3),dtype=np.uint8);frame[60:100,80:140]=255
    crop,padding=square_crop(frame,[80,60,140,100],ExtractionConfig())
    assert crop.shape == (299,299,3) and padding == 0
    white=np.where(crop[:,:,0]>128); assert 1.4 < (np.ptp(white[1])+1)/(np.ptp(white[0])+1) < 1.6


def test_segments_and_ranking_are_deterministic():
    segments=candidate_segments(100,4,3)
    assert len(segments)==4 and all(len(x)==3 for x in segments)
    assert all(all(i//25==s for i in indices) for s,indices in enumerate(segments))
    candidates=[({"reason":"eligible","score":2,"frame_index":8},None),({"reason":"eligible","score":2,"frame_index":3},None)]
    assert best_candidate(candidates)[0]["frame_index"]==3


def test_candidate_frames_use_one_seek_per_segment_and_keep_indices():
    class SequentialCapture:
        def __init__(self):self.cursor=0;self.seeks=0;self.grabs=0
        def set(self,prop,index):self.cursor=index;self.seeks+=1
        def grab(self):self.cursor+=1;self.grabs+=1;return True
        def read(self):
            value=self.cursor;self.cursor+=1
            return True,np.full((1,1,3),value,dtype=np.uint8)
    capture=SequentialCapture();frames=read_candidate_frames(capture,[5,8,11])
    assert [x[0] for x in frames]==[5,8,11]
    assert [int(x[2][0,0,0]) for x in frames]==[5,8,11]
    assert capture.seeks==1 and capture.grabs==4


def test_bounded_batch_inference_preserves_temporal_selection(tmp_path,monkeypatch):
    import cv2
    import deepfake_data.extract as module
    video=tmp_path/'batch.mp4';video.write_bytes(b'video');monkeypatch.setattr(cv2,'VideoCapture',type('Capture',(object,),{'__init__':lambda self,*a:setattr(self,'index',0),'isOpened':lambda self:True,'get':lambda self,p:6,'set':lambda self,p,v:setattr(self,'index',v),'grab':lambda self:(setattr(self,'index',self.index+1) or True),'read':lambda self:(True,np.tile((np.indices((160,160)).sum(axis=0)%2)[:,:,None],(1,1,3)).astype(np.uint8)),'release':lambda self:None}))
    monkeypatch.setattr(module,'_write_image',lambda path,image:path.write_bytes(b'png'))
    monkeypatch.setattr(module.os,'replace',lambda src,dst:shutil.copyfile(src,dst))
    class Detector:
        def __init__(self):self.batch_sizes=[]
        def detect(self,images,landmarks=True):
            assert isinstance(images,list);self.batch_sizes.append(len(images))
            boxes=[np.array([[20,20,120,120]],dtype=float) for _ in images]
            probs=[np.array([.99]) for _ in images]
            points=[None for _ in images]
            return boxes,probs,points
    detector=Detector();cfg=ExtractionConfig(max_frames=2,candidates_per_segment=3,min_sharpness=0,inference_batch_size=3)
    result=extract_video(video,tmp_path/'batch-out',detector,config=cfg)
    assert detector.batch_sizes==[3,3]
    assert [x['frame_index'] for x in result['frames']]==[0,3]


def test_confidence_and_size_rejection():
    frame=np.zeros((200,200,3),dtype=np.uint8); config=ExtractionConfig(min_sharpness=0)
    assert evaluate_candidate(frame,[[10,10,110,110]],[0],None,0,config)[0]["reason"]=="low_confidence_or_invalid_box"
    assert evaluate_candidate(frame,[[10,10,60,60]],[.99],None,0,config)[0]["reason"]=="face_too_small"


def test_visible_low_contrast_face_passes_revised_blur_threshold():
    grid=(np.indices((160,160)).sum(axis=0)%2).astype(np.uint8)
    frame=np.repeat(grid[:,:,None],3,axis=2)
    box=[[20,20,120,120]]
    before,_=evaluate_candidate(frame,box,[.99],None,0,ExtractionConfig(min_sharpness=30))
    after,crop=evaluate_candidate(frame,box,[.99],None,0,ExtractionConfig())
    assert before['reason']=='blurred' and 15 <= after['sharpness'] < 30
    assert after['reason']=='eligible' and crop.shape==(299,299,3)
    flat=np.zeros_like(frame)
    assert evaluate_candidate(flat,box,[.99],None,0,ExtractionConfig())[0]['reason']=='blurred'


def test_resume_configuration_provenance_and_zero_yield(tmp_path,monkeypatch):
    import cv2
    import deepfake_data.extract as module
    video=tmp_path/"v.mp4";video.write_bytes(b"video")
    class Capture:
        def __init__(self,path):self.index=0
        def isOpened(self):return True
        def get(self,p):return 3
        def set(self,p,v):self.index=v
        def read(self):
            image=np.zeros((160,160,3),dtype=np.uint8);image[::2]=255;image[0,0,0]=self.index
            return True,image
        def release(self):pass
    class Detector:
        def __init__(self,fail=False,empty=False):self.calls=[];self.fail=fail;self.empty=empty
        def detect(self,image,landmarks=True):
            index=int(image[0,0,2]);self.calls.append(index)
            if self.fail and index==1:raise RuntimeError("interrupted")
            if self.empty:return None,None,None
            return np.array([[20,20,120,120]]),np.array([.99]),None
    monkeypatch.setattr(cv2,"VideoCapture",Capture)
    monkeypatch.setattr(module,"_write_image",lambda path,image:path.write_bytes(b"png"))
    monkeypatch.setattr(module.os,"replace",lambda src,dst:shutil.copyfile(src,dst))
    config=ExtractionConfig(max_frames=3,candidates_per_segment=1,min_sharpness=0)
    provenance={"dataset":"ffpp","method":"Deepfakes","group_id":"g","project_partition":"train"}
    with pytest.raises(RuntimeError):extract_video(video,tmp_path/"output",Detector(fail=True),config=config,provenance=provenance)
    detector=Detector();result=extract_video(video,tmp_path/"output",detector,config=config,provenance=provenance)
    assert detector.calls==[1,2] and len(result["frames"])==3
    assert result["config"]["provenance"]==provenance
    legacy=json.loads((tmp_path/'output'/'state.json').read_text());legacy['parameters'].pop('inference_batch_size')
    (tmp_path/'output'/'state.json').write_text(json.dumps(legacy))
    complete=json.loads((tmp_path/'output'/'complete.json').read_text());complete['config']['parameters'].pop('inference_batch_size')
    (tmp_path/'output'/'complete.json').write_text(json.dumps(complete))
    assert extract_video(video,tmp_path/'output',Detector(fail=True),config=config,provenance=provenance)['frames']==result['frames']
    with pytest.raises(ValueError):extract_video(video,tmp_path/"output",detector,config=ExtractionConfig(max_frames=3,candidates_per_segment=1,margin=2),provenance=provenance)
    empty=extract_video(video,tmp_path/"empty",Detector(empty=True),config=config)
    assert empty["zero_yield"] and empty["rejection_counts"]["no_face"]==3


def test_training_manifest_rejects_changed_partition(tmp_path):
    from deepfake_data.extract import output_key
    from deepfake_data.training_manifest import build_rows
    row={"dataset":"ffpp","method":"Deepfakes","relative_path":"f.mp4","absolute_path":str(tmp_path/"f.mp4"),"group_id":"source-1","project_partition":"test","media_type":"video","label":"0"}
    folder=tmp_path/output_key(row);folder.mkdir();(folder/"f.png").write_bytes(b"crop")
    provenance={key:row.get(key,"") for key in ("dataset","method","relative_path","group_id","project_partition")};provenance["project_partition"]="train"
    (folder/"complete.json").write_text(json.dumps({"config":{"version":2,"video":row["absolute_path"],"parameters":{},"provenance":provenance},"frames":[{"file":"f.png","frame_index":1}]}))
    output,skipped=build_rows([row],tmp_path)
    assert not output and skipped[0]["reason"]=="provenance_mismatch"
