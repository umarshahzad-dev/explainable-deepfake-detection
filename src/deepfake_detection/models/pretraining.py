"""Explicit offline source admission; face checkpoints must match the backbone."""
from pathlib import Path
import math
import torch
from torch import nn
from torch.nn import functional as F
from deepfake_detection.common.paths import ROOT
from deepfake_detection.common.paths import read_json,fingerprint
from deepfake_detection.data.preprocessing import preprocessing_hash
from deepfake_detection.models.model_loader import artifact_metadata,load_run_model


def self_supervised_pretraining(*args,**kwargs):
    raise NotImplementedError('Self-supervised pretraining requires 1-2 weeks of separate work: select SSL objective, approved unlabeled development data, a pretraining runner and validated local backbone checkpoint; no automatic downloads')


def synthetic_face_pretraining(*args,**kwargs):
    raise NotImplementedError('Synthetic face pretraining requires an approved local generator/checkpoint, provenance-labelled synthetic dataset, identity-safe splits, disk/VRAM budget and a separate pretraining runner; no generator download or inference is automatic')


def source_weights(cfg):
    pretraining=cfg.get('pretraining',{})
    if pretraining.get('self_supervised'): self_supervised_pretraining()
    if pretraining.get('pretrain_synthetic'): synthetic_face_pretraining()
    model_source=cfg['model'].get('pretrained_source','imagenet'); section_source=pretraining.get('source','imagenet')
    if model_source!='imagenet' and section_source!='imagenet' and model_source!=section_source: raise ValueError('Conflicting pretrained sources in model/pretraining')
    source=model_source if model_source!='imagenet' else section_source
    if source not in ('imagenet','vggface2','ms1m','webface'): raise ValueError(f'Unknown pretrained source: {source}')
    value=cfg['weights'] if source=='imagenet' else pretraining.get(source+'_weights_path')
    if not value: raise FileNotFoundError(f'{source} requires explicit pretraining.{source}_weights_path; provide compatible local raw backbone weights and verified source normalization; downloads disabled')
    path=Path(value); path=path if path.is_absolute() else ROOT/path
    if not path.is_file(): raise FileNotFoundError(f'{source} local pretrained file missing: {path}; downloads disabled')
    return source,path


class DistillationLoss(nn.Module):
    def __init__(self,teacher_model,temperature=4.,alpha=.5,*,student_config=None,teacher_config=None):
        super().__init__()
        if temperature<=0 or not math.isfinite(temperature) or not 0<=alpha<=1: raise ValueError('Invalid distillation temperature/alpha')
        self.teacher=teacher_model.eval(); self.temperature=temperature; self.alpha=alpha
        self.student_config=student_config; self.teacher_config=teacher_config
        for parameter in self.teacher.parameters(): parameter.requires_grad_(False)
    def distill(self,student_logits,teacher_logits):
        if student_logits.shape!=teacher_logits.shape or student_logits.shape[-1]!=1: raise ValueError('Distillation requires matching REAL logits')
        t=self.temperature; logits=student_logits.float()/t; teacher=teacher_logits.detach().float()/t
        log_probs=torch.cat([F.logsigmoid(-logits),F.logsigmoid(logits)],-1)
        probability=torch.sigmoid(teacher); reference=torch.cat([1-probability,probability],-1)
        return F.kl_div(log_probs,reference,reduction='batchmean')*t*t
    def combine(self,supervised_loss,student_logits,teacher_logits): return (1-self.alpha)*supervised_loss+self.alpha*self.distill(student_logits,teacher_logits)
    def forward(self,student_logits,teacher_logits,labels):
        return self.combine(F.binary_cross_entropy_with_logits(student_logits,labels.to(student_logits).reshape_as(student_logits)),student_logits,teacher_logits)
    @torch.no_grad()
    def teacher_logits(self,inputs):
        self.teacher.eval(); value=inputs.float()
        if self.student_config is not None and self.teacher_config is not None:
            student=self.student_config['model']; teacher=self.teacher_config['model']
            mean=torch.tensor(student['mean'],device=value.device)[None,:,None,None]; std=torch.tensor(student['std'],device=value.device)[None,:,None,None]
            value=value*std+mean
            if value.shape[-2:]!=(teacher['input_size'],teacher['input_size']): value=F.interpolate(value,size=(teacher['input_size'],teacher['input_size']),mode='bicubic',align_corners=False,antialias=True)
            mean=torch.tensor(teacher['mean'],device=value.device)[None,:,None,None]; std=torch.tensor(teacher['std'],device=value.device)[None,:,None,None]
            value=(value-mean)/std
        return self.teacher(value)


def assert_teacher_separation(exposure,held,*,message='Teacher exposure'):
    """Fit and checkpoint-selection information cannot enter final heldouts."""
    for field in ('parents','known_person_ids'):
        used={token for row in exposure for token in row.get(field,[])}
        evaluated={token for row in held for token in row.get(field,[])}
        if used&evaluated: raise ValueError(f'{message} overlaps student held-out {field}')
    for field in ('sample_id','video_id','sha256','crop_sha256'):
        used={row[field] for row in exposure if row.get(field)}
        evaluated={row[field] for row in held if row.get(field)}
        if used&evaluated: raise ValueError(f'{message} overlaps student held-out {field}')


def distillation_spec(cfg,rows,workspace):
    recipe=cfg['training']; section=cfg.get('pretraining',{})
    if not (recipe.get('knowledge_distillation') or section.get('knowledge_distillation')): return None
    selected=recipe.get('teacher_run') or section.get('teacher_run')
    if not selected: raise ValueError('Knowledge distillation requires an explicit local teacher_run; downloads disabled')
    if recipe.get('teacher_run') and section.get('teacher_run') and recipe['teacher_run']!=section['teacher_run']: raise ValueError('Conflicting teacher run IDs')
    from deepfake_detection.models.registry import resolve_model_path
    run=resolve_model_path(selected,workspace); teacher_cfg=read_json(run/'config.json')
    if preprocessing_hash(cfg)!=preprocessing_hash(teacher_cfg): raise ValueError('Teacher/student preprocessing contracts differ; re-extract/retrain a compatible teacher before distillation')
    metadata=artifact_metadata(run,teacher_cfg,workspace)
    manifest=read_json(run/'input_manifest.json'); teacher_rows=manifest.get('rows')
    if not teacher_rows: raise ValueError('Teacher training provenance is missing')
    if any(row.get('dataset')=='dfd' or row.get('split')=='external' for row in teacher_rows): raise ValueError('OOD teacher data cannot enter distillation')
    fit=[row for row in teacher_rows if row.get('split')=='train']; held=[row for row in rows if row.get('split') in ('val','calibration','test','external')]
    if not fit: raise ValueError('Teacher training provenance has no fit records')
    inherited=manifest.get('teacher_exposure',[])
    if (teacher_cfg.get('training',{}).get('knowledge_distillation') or teacher_cfg.get('pretraining',{}).get('knowledge_distillation')) and teacher_cfg.get('training',{}).get('teacher_weights_sha256'):
        if not inherited or fingerprint(inherited)!=teacher_cfg['training'].get('teacher_lineage_sha256'):
            raise ValueError('Distilled teacher exposure lineage missing or changed; rebuild a provenance-complete teacher')
    if not isinstance(inherited,list): raise ValueError('Teacher exposure lineage must be a list')
    exposure=[{**row,'exposure_kind':'train' if row['split']=='train' else 'selection',
               'exposure_run':selected,'exposure_weights_sha256':metadata['weights_sha256']}
              for row in teacher_rows if row.get('split') in {'train','val'}]+inherited
    assert_teacher_separation([r for r in exposure if r.get('exposure_kind')=='train'],held,message='Teacher training')
    final_held=[r for r in rows if r.get('split') in {'calibration','test','external'}]
    assert_teacher_separation(exposure,final_held,message='Teacher selection/exposure')
    recipe['teacher_run']=selected; recipe['teacher_weights_sha256']=metadata['weights_sha256']
    recipe['teacher_lineage_sha256']=fingerprint(exposure)
    cfg['pilot']=bool(cfg.get('pilot') or metadata.get('pilot') or any(r.get('pilot') for r in exposure))
    return {'run':run,'weights_sha256':metadata['weights_sha256'],'exposure_rows':exposure}


def load_distillation(cfg,spec,workspace,device):
    if spec is None: return None
    teacher,teacher_cfg,metadata=load_run_model(spec['run'],workspace=workspace,device=device)
    if metadata['weights_sha256']!=spec['weights_sha256']: raise ValueError('Teacher weights changed during run admission')
    return DistillationLoss(teacher,cfg['training'].get('distillation_temperature',4),cfg['training'].get('distillation_alpha',.5),student_config=cfg,teacher_config=teacher_cfg)
