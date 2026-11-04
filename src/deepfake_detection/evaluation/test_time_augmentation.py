"""Lazy deterministic crop-level TTA, bounded sequential model inference."""
from PIL import Image,ImageOps,ImageEnhance
import math


def tta_views(image,cfg):
    inference=cfg.get('inference',{})
    sizes=inference.get('tta_multi_scale') or [cfg['model']['input_size']]
    if not isinstance(sizes,(tuple,list)) or not sizes or len(sizes)>8 or any(not isinstance(size,int) or isinstance(size,bool) or not 1<=size<=1024 for size in sizes):
        raise ValueError('TTA scales must be a bounded list of positive integer sizes up to1024')
    rotations=inference.get('tta_rotation') or [0]
    if not isinstance(rotations,(tuple,list)) or len(rotations)>16 or any(not isinstance(angle,(int,float)) or not math.isfinite(angle) or abs(angle)>180 for angle in rotations):
        raise ValueError('TTA rotations must be a bounded finite angle list')
    width,height=image.size
    boxes=[(0,0,width,height)]
    if inference.get('tta_multi_crop'):
        cropped_width=max(1,int(width*.875)); cropped_height=max(1,int(height*.875))
        right=width-cropped_width; bottom=height-cropped_height
        origins=[(right//2,bottom//2),(0,0),(right,0),(0,bottom),(right,bottom)]
        boxes=[(x,y,x+cropped_width,y+cropped_height) for x,y in origins]
    for box in boxes:
        cropped=image.crop(box)
        for horizontal in ([False,True] if inference.get('tta_hflip') else [False]):
            for vertical in ([False,True] if inference.get('tta_vflip') else [False]):
                view=ImageOps.mirror(cropped) if horizontal else cropped
                if vertical: view=ImageOps.flip(view)
                for angle in rotations:
                    rotated=view.rotate(angle,resample=Image.Resampling.BICUBIC,expand=False,fillcolor=(128,128,128)) if angle else view
                    for color in ([False,True] if inference.get('tta_color_jitter') else [False]):
                        colored=ImageEnhance.Color(ImageEnhance.Brightness(rotated).enhance(1.05)).enhance(.95) if color else rotated
                        for size in sizes:
                            yield {'image':colored,'input_size':size,'hflip':horizontal,'vflip':vertical,'crop_box':box,'rotation':angle,'color_jitter':color}


def tta_enabled(cfg):
    inference=cfg.get('inference',{})
    return any(inference.get(key,False) for key in ('tta_hflip','tta_vflip','tta_multi_crop','tta_color_jitter')) or bool(inference.get('tta_multi_scale')) or any(inference.get('tta_rotation') or [0])


def inference_augmented(cfg):
    return tta_enabled(cfg) or bool(cfg.get('inference',{}).get('use_mc_dropout'))


def combined_tta(model,image,cfg,*,device=None,temperature=1,dropout=False):
    """Average raw view probabilities, then calibrate the effective pooled logit."""
    import torch
    from deepfake_detection.data.preprocessing import model_tensor
    from deepfake_detection.evaluation.metrics import real_probability
    if not math.isfinite(temperature) or temperature<=0: raise ValueError('Temperature must be positive and finite')
    parameter=next(model.parameters(),None)
    device=torch.device(device or (parameter.device if parameter is not None else 'cpu'))
    modes=[module.training for module in model.modules()]; total=squared=0.; count=0
    try:
        model.eval()
        if dropout:
            for module in model.modules():
                if isinstance(module,torch.nn.modules.dropout._DropoutNd): module.train()
        with torch.no_grad():
            for view in tta_views(image,cfg):
                transform={**cfg['model'],'input_size':view['input_size']}
                tensor=model_tensor(view['image'],transform).unsqueeze(0).to(device)
                output=model(tensor).float()
                if output.shape!=(1,1) or not bool(torch.isfinite(output).all()): raise ValueError('TTA requires one finite binary logit per view')
                probability=float(real_probability([float(output.item())])[0])
                total+=probability; squared+=probability**2; count+=1
    finally:
        for module,mode in zip(model.modules(),modes): module.training=mode
    if not count: raise ValueError('TTA generated no views')
    raw=total/count; bounded=min(1-1e-12,max(1e-12,raw)); logit=math.log(bounded)-math.log1p(-bounded)
    return {'real_probability':float(real_probability([logit],temperature)[0]),'logit':logit,'views':count,'view_probability_std':math.sqrt(max(0,squared/count-raw**2))}


def mc_dropout_predict(model,image,samples=20,*,cfg=None,device=None,seed=42):
    """Seeded repeatable MC probability mean/std; never updates BN buffers."""
    import torch
    import numpy as np
    if not isinstance(samples,int) or not 2<=samples<=2000: raise ValueError('MC dropout requires2..2000 samples')
    if not any(isinstance(module,torch.nn.modules.dropout._DropoutNd) and module.p>0 for module in model.modules()):
        raise ValueError('MC dropout requires an active dropout layer')
    parameter=next(model.parameters(),None); device=torch.device(device or (parameter.device if parameter is not None else 'cpu'))
    modes=[module.training for module in model.modules()]; predictions=[]; views=1
    devices=list(range(torch.cuda.device_count())) if device.type=='cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if device.type=='cuda': torch.cuda.manual_seed_all(seed)
            model.eval()
            for module in model.modules():
                if isinstance(module,torch.nn.modules.dropout._DropoutNd): module.train()
            with torch.no_grad():
                for _ in range(samples):
                    if isinstance(image,torch.Tensor):
                        output=model(image.to(device)).float()
                        if output.ndim!=2 or output.shape[1]!=1 or not bool(torch.isfinite(output).all()): raise ValueError('MC dropout requires finite binary logits')
                        from deepfake_detection.evaluation.metrics import real_probability
                        predictions.append(real_probability(output.cpu().reshape(-1).numpy()))
                    else:
                        if cfg is None: raise ValueError('PIL MC dropout requires model preprocessing config')
                        packet=combined_tta(model,image,cfg,device=device,dropout=True)
                        predictions.append(np.asarray([packet['real_probability']])); views=packet['views']
    finally:
        for module,mode in zip(model.modules(),modes): module.training=mode
    array=np.asarray(predictions)
    return {'real_mean':array.mean(0).tolist(),'real_std':array.std(0).tolist(),'samples':samples,'views':views,'uncertainty_note':'Dropout predictive variation; not an OOD guarantee'}


def inference_score(model,image,cfg,*,device='cpu'):
    inference=cfg.get('inference',{})
    if not inference.get('use_mc_dropout'): return combined_tta(model,image,cfg,device=device)
    result=mc_dropout_predict(model,image,inference.get('mc_dropout_samples',20),cfg=cfg,device=device,seed=cfg.get('seed',42))
    real=result['real_mean'][0]; bounded=min(1-1e-12,max(1e-12,real))
    return {'real_probability':real,'logit':math.log(bounded)-math.log1p(-bounded),'views':result['views'],'mc_samples':result['samples'],'mc_dropout_real_std':result['real_std'][0]}


def predict_tta_dataset(model,dataset,cfg,*,device='cpu',progress=None):
    import torch
    logits=[]; labels=[]
    for index in range(len(dataset)):
        image,label=dataset.read_master(index)
        image=dataset.augment(image)
        predicted=inference_score(model,image,cfg,device=device)
        logits.append(predicted['logit']); labels.append(label)
        if progress is not None: progress.update(index+1)
    if not labels: raise ValueError('Empty TTA dataset')
    if progress is not None: progress.finish(len(labels))
    loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(logits,dtype=torch.float64),torch.tensor(labels,dtype=torch.float64))
    return {'labels':labels,'logits':logits,'loss':float(loss),'tta_applied':True}
