"""Explicit permitted research stubs; never silently substitute another method."""


def clip_detection(*args,**kwargs):
    raise NotImplementedError('CLIP detection requires open_clip or transformers, local compatible weights and1–2weeks separate validated implementation')


def lora_adaptation(*args,**kwargs):
    raise NotImplementedError('LoRA requires model-specific adapter insertion, local backbone weights, adapter checkpoint admission and a separate validation study; estimated3–7days')


def neural_architecture_search(*args,**kwargs):
    raise NotImplementedError('NAS requires a bounded search space, validation-only selection and many independent training runs; not executed on this laptop implementation')


def population_based_training(*args,**kwargs):
    raise NotImplementedError('PBT requires multiple simultaneous population runs, checkpoint exploitation/mutation and a validation-only scheduler; separate compute study, no population launched')


def spsl_detection(*args,**kwargs):
    raise NotImplementedError('SPSL is a separate spatial/phase architecture and training reproduction. Reference: Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain (CVPR2021). The FFT branch is not a claimed SPSL reproduction; estimated1–2weeks')


def recce_detection(*args,**kwargs):
    raise NotImplementedError('RECCE requires a separately trained reconstruction/classification architecture. Reference: End-to-End Reconstruction-Classification Learning for Face Forgery Detection (CVPR2022). Estimated2–3weeks; no pretrained asset or reconstruction model supplied')


def face_xray_detection(*args,**kwargs):
    raise NotImplementedError('Face X-ray requires synthetic blend-boundary supervision, a boundary-prediction head and separate training. Reference: Face X-Ray for More General Face Forgery Detection (CVPR2020); estimated1–2weeks. Basic SBI does not implement this boundary detector')


def master_landmarks(row,size):
    import numpy as np,math
    if row.get('crop_landmarks') is not None: points=np.asarray(row['crop_landmarks'],dtype=float)
    else:
        points=np.asarray(row.get('landmarks'),dtype=float); geometry=row.get('geometry',{})
        if points.shape!=(5,2) or not np.isfinite(points).all() or not all(key in geometry for key in ('x','y','side','source_width','source_height')):
            raise ValueError('SBI requires saved finite landmarks and complete crop geometry')
        values=[geometry[key] for key in ('x','y','side','source_width','source_height')]+[geometry.get('rotation_degrees',0)]
        if not all(math.isfinite(value) for value in values) or geometry['side']<=0: raise ValueError('Invalid SBI landmark geometry')
        center=np.asarray([geometry['source_width']/2,geometry['source_height']/2]); angle=math.radians(geometry.get('rotation_degrees',0))
        matrix=np.asarray([[math.cos(angle),math.sin(angle)],[-math.sin(angle),math.cos(angle)]])
        points=((points-center)@matrix.T+center-np.asarray([geometry['x'],geometry['y']]))*size/geometry['side']
    if points.shape!=(5,2) or not np.isfinite(points).all() or (points<0).any() or (points>=size).any(): raise ValueError('SBI landmarks must lie inside the saved master crop')
    return points


def self_blended_images(image,prob=.3,*,landmarks=None,rng=None,return_applied=False):
    """Basic five-landmark hull self-blending proxy, not a full SBI reproduction."""
    import random,math,cv2
    import numpy as np
    from PIL import Image,ImageDraw,ImageFilter,ImageEnhance
    if not math.isfinite(prob) or not 0<=prob<=1: raise ValueError('SBI probability must be in[0,1]')
    points=np.asarray(landmarks,dtype=float)
    if prob and (points.shape!=(5,2) or not np.isfinite(points).all()): raise ValueError('SBI requires five finite crop-coordinate landmarks')
    rng=rng or random
    if not prob or rng.random()>=prob: return (image.copy(),False) if return_applied else image.copy()
    if (points<0).any() or (points[:,0]>=image.width).any() or (points[:,1]>=image.height).any(): raise ValueError('SBI landmarks lie outside the image')
    hull=cv2.convexHull(points.astype(np.float32)).reshape(-1,2)
    if len(hull)<3 or cv2.contourArea(hull)<1: raise ValueError('SBI landmarks form a degenerate face region')
    mask=Image.new('L',image.size,0); ImageDraw.Draw(mask).polygon([tuple(point) for point in hull],fill=255)
    mask=mask.filter(ImageFilter.GaussianBlur(max(1,min(image.size)*.025)))
    strength=rng.uniform(.5,1.); mask=mask.point(lambda value:int(value*strength))
    source=image.convert('RGB').rotate(rng.uniform(-5,5),resample=Image.Resampling.BICUBIC,fillcolor=(0,0,0))
    source=source.transform(image.size,Image.Transform.AFFINE,(1,0,rng.uniform(-.02,.02)*image.width,0,1,rng.uniform(-.02,.02)*image.height),resample=Image.Resampling.BICUBIC)
    source=ImageEnhance.Brightness(source).enhance(rng.uniform(.9,1.1))
    blended=Image.composite(source,image.convert('RGB'),mask)
    return (blended,True) if return_applied else blended
