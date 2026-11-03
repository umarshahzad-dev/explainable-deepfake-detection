"""Bounded image-to-video perceptual evidence; never makes linkage decisions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from .extract import ExtractionConfig,choose_face,make_detector,square_crop,validate_output


def dhash(image):
    import cv2
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
    value=0
    for bit in (small[:,1:]>=small[:,:-1]).flat:value=(value<<1)|int(bit)
    return value


def peripheral_feature_support(left,right):
    """Count geometrically consistent peripheral SIFT matches; never decides identity."""
    import cv2
    import numpy as np
    if not hasattr(cv2,'SIFT_create'):return {'feature_error':'SIFT unavailable'}
    detector=cv2.SIFT_create();height,width=left.shape[:2]
    a,da=detector.detectAndCompute(cv2.cvtColor(left,cv2.COLOR_BGR2GRAY),None)
    b,db=detector.detectAndCompute(cv2.cvtColor(right,cv2.COLOR_BGR2GRAY),None)
    if da is None or db is None or len(db)<2:return {'peripheral_matches':0,'geometric_inliers':0}
    matches=[]
    for pair in cv2.BFMatcher().knnMatch(da,db,k=2):
        if len(pair)<2:continue
        first,second=pair;x,y=a[first.queryIdx].pt
        if first.distance<.7*second.distance and (x<width*.2 or x>width*.8 or y<height*.15 or y>height*.85):matches.append(first)
    inliers=0
    if len(matches)>=4:
        src=np.float32([a[m.queryIdx].pt for m in matches]);dst=np.float32([b[m.trainIdx].pt for m in matches]);cv2.setRNGSeed(42)
        _,mask=cv2.findHomography(src,dst,cv2.RANSAC,3.0)
        if mask is not None:inliers=int(mask.sum())
    return {'peripheral_matches':len(matches),'geometric_inliers':inliers}

def probe(pairs,output,detector,samples_per_video=3,max_pairs=5):
    import cv2
    from PIL import Image,ImageDraw,ImageOps
    output=validate_output(output);output.mkdir(parents=True,exist_ok=True);results=[];panels=[]
    config=ExtractionConfig();
    for i,row in enumerate(pairs):
        if i>=max_pairs:break
        image_path=Path(row['image_path']);video_path=Path(row['video_path']);result={'image_path':str(image_path),'video_path':str(video_path),'relationship':'suspected','evidence':row.get('evidence',''),'frames':[]}
        image=cv2.imread(str(image_path));capture=cv2.VideoCapture(str(video_path))
        if image is None or not capture.isOpened():
            result['error']='image_or_video_unreadable';capture.release();results.append(result);continue
        try:
            count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if count<=0:result['error']='missing_frame_count';results.append(result);continue
            indices=sorted({0 if samples_per_video==1 else j*(count-1)//(samples_per_video-1) for j in range(samples_per_video)})
            source_hash=dhash(cv2.resize(image,(299,299),interpolation=cv2.INTER_AREA));best=None
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES,index);ok,frame=capture.read()
                if not ok:result['frames'].append({'frame_index':index,'error':'read_failed'});continue
                boxes,probs,points=detector.detect(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),landmarks=True)
                face=choose_face(boxes,probs,points,.90) if boxes is not None and probs is not None else None
                if face is None:result['frames'].append({'frame_index':index,'error':'no_face'});continue
                box,_=face;crop,padding=square_crop(frame,box,config)
                if crop is None:result['frames'].append({'frame_index':index,'error':'invalid_crop'});continue
                distance=(source_hash^dhash(crop)).bit_count();entry={'frame_index':index,'dhash_distance':distance,'padding_fraction':padding,**peripheral_feature_support(image,crop)};result['frames'].append(entry)
                if best is None or (distance,index)<(best[0],best[1]):best=(distance,index,crop)
            if best is not None:
                result['best_distance']=best[0];result['best_frame_index']=best[1];panels.append((image,best[2],result))
        finally:capture.release()
        results.append(result)
    sheet=Image.new('RGB',(620,max(1,len(panels))*250),'white');draw=ImageDraw.Draw(sheet)
    for i,(left,right,info) in enumerate(panels):
        for j,img in enumerate((left,right)):
            rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB);thumb=ImageOps.contain(Image.fromarray(rgb),(300,220));sheet.paste(thumb,(j*310,i*250))
        draw.text((3,i*250+223),f"pair {i+1}: distance={info['best_distance']}, frame={info['best_frame_index']}",fill='black')
    sheet.save(output/'contact-sheet.jpg')
    report={'pairs':results,'limitations':['Perceptual distance and geometric feature matches are uncalibrated review aids, not automatic proof or disproof of source or identity','A small frame sample can miss true overlap','No linkage or split assignment is changed']}
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--max-pairs',type=int,default=5);p.add_argument('--samples-per-video',type=int,default=3);p.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=p.parse_args()
    if a.max_pairs<1 or a.samples_per_video<1:p.error('Limits must be positive')
    with a.pairs.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))[:a.max_pairs]
    print(json.dumps(probe(rows,a.output,make_detector(a.device),a.samples_per_video,a.max_pairs),indent=2))

if __name__=='__main__':main()
