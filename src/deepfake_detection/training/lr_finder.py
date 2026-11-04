"""Bounded geometric LR range diagnostic; never publishes training weights."""
import io
import math
import time
import torch
from torch.utils.data import DataLoader
from deepfake_detection.training.averaging import cpu_state
from deepfake_detection.data.datasets import CropDataset
from deepfake_detection.data.integrity import preflight_crops
from deepfake_detection.models.losses import objective_for_model
from deepfake_detection.data.manifests import check_splits
from deepfake_detection.models.model_loader import initialize_model
from deepfake_detection.common.paths import Workspace,atomic_bytes,write_json
from deepfake_detection.common.progress import Progress
from deepfake_detection.common.reproducibility import seed_all,rng_state,restore_rng
from deepfake_detection.training.trainer import _optimizer,train_epoch,NoOptimizerUpdates


def _plot(curve):
    """Log-LR plot using installed Pillow; no optional plotting dependency."""
    from PIL import Image,ImageDraw
    image=Image.new('RGB',(800,480),'white'); draw=ImageDraw.Draw(image)
    draw.line([(65,35),(65,415),(765,415)],fill='black',width=2)
    draw.text((70,10),'Learning rate range test (one batch per update)',fill='black')
    draw.text((350,450),'Learning rate (log scale)',fill='black'); draw.text((5,20),'Loss',fill='black')
    valid=[row for row in curve if row['updated']]
    if valid:
        xmin=min(math.log10(row['lr']) for row in valid); xmax=max(math.log10(row['lr']) for row in valid)
        ymin=min(row['smoothed_loss'] for row in valid); ymax=max(row['smoothed_loss'] for row in valid)
        points=[(65+700*(math.log10(row['lr'])-xmin)/max(xmax-xmin,1e-12),415-370*(row['smoothed_loss']-ymin)/max(ymax-ymin,1e-12)) for row in valid]
        if len(points)>1: draw.line(points,fill='blue',width=2)
        for point in points: draw.ellipse((point[0]-2,point[1]-2,point[0]+2,point[1]+2),fill='blue')
        draw.text((65,420),f"{10**xmin:.1e}",fill='black'); draw.text((700,420),f"{10**xmax:.1e}",fill='black')
    output=io.BytesIO(); image.save(output,format='PNG'); return output.getvalue()


def find_lr(name,cfg,rows,workspace=None,*,device=None,steps=100,start_lr=1e-7,end_lr=.01):
    if not 2<=steps<=500 or not 0<start_lr<end_lr or not math.isfinite(end_lr): raise ValueError('LR finder needs 2..500 steps and positive increasing finite LRs')
    if any(row.get('dataset')=='dfd' or row.get('split')=='external' for row in rows): raise ValueError('OOD data cannot tune learning rates')
    audit=check_splits(rows)
    if not audit['ok']: raise ValueError('LR finder input split audit failed')
    training_rows=[row for row in rows if row.get('split')=='train']
    if not training_rows or {row['label'] for row in training_rows}!={0,1}: raise ValueError('LR finder needs both training classes')
    ws=workspace or Workspace(); run=ws.run(name)
    if run.exists() and any(path.name!='run.log' for path in run.iterdir()): raise FileExistsError('LR finder requires a fresh diagnostic run name')
    preflight_crops(training_rows,cfg,ws)
    device=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    original_rng=rng_state(); seed_all(cfg['seed']); model=None; saved=None; flags=[]
    started=time.perf_counter(); curve=[]; updates=0; best=float('inf'); smooth=0
    progress=Progress('LR range batch',steps)
    try:
        model=initialize_model(cfg,device=device); saved=cpu_state(model); flags=[(m,m.training) for m in model.modules()]
        requires=[(p,p.requires_grad) for p in model.parameters()]
        objective=objective_for_model(model,cfg['training']).to(device)
        optimizer=_optimizer(model,cfg,'finetune',objective.parameters())
        base_lrs=[group['lr'] for group in optimizer.param_groups]; reference=cfg['training']['backbone_lr']
        loader=DataLoader(CropDataset(training_rows,cfg,training=True,workspace=ws),batch_size=cfg['training']['batch_size'],shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(cfg['seed']))
        iterator=iter(loader)
        from deepfake_detection.training.precision import amp_settings
        _,_,scale_enabled=amp_settings(cfg['training']['amp'],device,cfg['training'])
        scaler=torch.amp.GradScaler('cuda',enabled=scale_enabled,init_scale=1024)
        for index in range(steps):
            try: batch=next(iterator)
            except StopIteration: iterator=iter(loader); batch=next(iterator)
            lr=start_lr*(end_lr/start_lr)**(index/(steps-1))
            for group,base_lr in zip(optimizer.param_groups,base_lrs): group['lr']=lr*base_lr/reference
            try:
                outcome=train_epoch(model,[batch],optimizer,device=device,accumulation=1,amp=cfg['training']['amp'],scaler=scaler,clip_norm=cfg['training']['clip_norm'],freeze_statistics=cfg['training']['freeze_bn'],recipe={**cfg['training'],'attack_mean':cfg['model']['mean'],'attack_std':cfg['model']['std']},objective=objective)
            except NoOptimizerUpdates as exc:
                if exc.reason!='scaler_skipped': raise
                curve.append({'step':index+1,'lr':lr,'loss':None,'smoothed_loss':None,'updated':False}); continue
            updates+=outcome['optimizer_updates']; smooth=.98*smooth+.02*outcome['loss']; corrected=smooth/(1-.98**updates)
            curve.append({'step':index+1,'lr':lr,'loss':outcome['loss'],'smoothed_loss':corrected,'updated':True})
            best=min(best,corrected); progress.update(index+1)
            if updates>10 and corrected>4*best: break
        valid=[row for row in curve if row['updated']]
        if len(valid)<2: raise ValueError('LR finder performed fewer than two successful updates')
        interior=valid[max(0,len(valid)//10):max(2,len(valid)-len(valid)//10)]
        slopes=[((b['smoothed_loss']-a['smoothed_loss'])/(math.log(b['lr'])-math.log(a['lr'])),a['lr']) for a,b in zip(interior,interior[1:])]
        recommended=min(slopes)[1] if slopes else valid[0]['lr']
        result={'pilot':True,'diagnostic':'lr_range','seed':cfg['seed'],'optimizer_updates':updates,'curve':curve,'recommended_lr':recommended,'elapsed_seconds':time.perf_counter()-started,'report_path':str(run),'note':'Geometric backbone LR; head LR keeps its configured ratio. One microbatch per update. Recommendation needs validation; no trained weights exported.'}
        atomic_bytes(run/'learning_rate_finder.png',_plot(curve),ws); write_json(run/'lr_finder.json',result,ws)
        progress.finish(len(curve)); return result
    finally:
        if model is not None and saved is not None:
            model.load_state_dict(saved)
            for module,flag in flags: module.training=flag
            for parameter,flag in requires: parameter.requires_grad_(flag)
            model.zero_grad(set_to_none=True)
        restore_rng(original_rng)
