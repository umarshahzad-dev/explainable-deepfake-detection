"""Explicit two-phase training. No training is executed by import or other CLI commands."""
import csv
from io import StringIO
import math
import time
import copy
import re
from contextlib import nullcontext
import torch
from torch import nn
from torch.utils.data import DataLoader
from deepfake_detection.training.checkpoints import load_checkpoint, resume_signature, save_checkpoint
from deepfake_detection.data.datasets import CropDataset, HierarchicalSampler,mixup_batch,cutmix_batch,fouriermix_batch,fft_amplitude_mix_batch,dct_mask_batch,add_gaussian_noise
from deepfake_detection.data.manifests import check_splits
from deepfake_detection.evaluation.metrics import compute_metrics, real_probability
from deepfake_detection.models.model_loader import architecture, export_weights, initialize_model, publish_best, artifact_metadata
from deepfake_detection.common.paths import Workspace, atomic_bytes, fingerprint, read_json, write_json
from deepfake_detection.common.reproducibility import seed_all, seed_worker, rng_state, restore_rng
from deepfake_detection.common.progress import Progress, console
from deepfake_detection.training.optimizers import make_optimizer,OPTIMIZER_REGISTRY
from deepfake_detection.models.losses import objective_for_model,objective_forward,LOSS_REGISTRY
from deepfake_detection.training.averaging import WeightEMA,StochasticWeightAveraging,WeightAveraging
from deepfake_detection.training.adversarial import fgsm_attack,pgd_attack
from deepfake_detection.models.pretraining import distillation_spec,load_distillation
from deepfake_detection.training.precision import amp_settings


def freeze_bn(model):
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def regularization_loss(model,l1=0,l2=0):
    """Summed penalties over trainable model parameters, including biases."""
    if not math.isfinite(l1) or not math.isfinite(l2) or l1<0 or l2<0: raise ValueError('Regularization coefficients must be finite and nonnegative')
    parameters=[p for p in model.parameters() if p.requires_grad]
    if not parameters: raise ValueError('Regularization requires trainable parameters')
    loss=parameters[0].sum()*0
    for parameter in parameters:
        values=parameter.float()
        if l1: loss=loss+l1*values.abs().sum()
        if l2: loss=loss+l2*values.square().sum()
    return loss


@torch.no_grad()
def add_weight_noise(model,std=.01):
    """Seeded additive optimizer-update noise on trainable weight tensors."""
    if std<0 or not math.isfinite(std): raise ValueError('Weight noise std must be finite and nonnegative')
    if not std: return
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.ndim>1: parameter.add_(torch.randn_like(parameter)*std)


class NoOptimizerUpdates(ValueError):
    def __init__(self, reason, message):
        self.reason = reason
        super().__init__(message)


def train_epoch(model, loader, optimizer, *, device="cpu", accumulation=4, amp=True,
                clip_norm=1.0, scaler=None, scheduler=None, freeze_statistics=True, progress=None,recipe=None,objective=None,update_observers=(),distillation=None):
    device = torch.device(device)
    recipe=recipe or {}
    if recipe.get('mixup_alpha',0)>0 and recipe.get('cutmix_alpha',0)>0:
        raise ValueError('Select either MixUp or CutMix for a batch, not both')
    enabled,amp_dtype,scale_enabled=amp_settings(amp,device,recipe)
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=scale_enabled)
    if amp_dtype==torch.bfloat16 and hasattr(scaler,'is_enabled') and scaler.is_enabled(): raise ValueError('BF16 AMP does not use GradScaler')
    model.train()
    objective=objective or objective_for_model(model,recipe).to(device)
    objective.train()
    optimized={id(p) for group in optimizer.param_groups for p in group['params']}
    if any(p.requires_grad and id(p) not in optimized for p in objective.parameters()):
        raise ValueError('Learned auxiliary loss parameters must be included in the optimizer; use run_training or explicitly add them')
    if freeze_statistics:
        freeze_bn(model)
    optimizer.zero_grad(set_to_none=True)
    samples, microbatches, updates, skipped = 0, 0, 0, 0
    group_samples = 0
    nominal_group = None
    loss_sum = 0.0
    labels_all, logits_all = [], []
    replay=[]
    sharpness=bool(getattr(optimizer,'is_sam',False))
    if accumulation < 1:
        raise ValueError("Accumulation must be positive")
    def forward_loss(inputs,labels,domains=None):
        logits,loss=objective_forward(model,inputs,labels,objective,domains)
        if recipe.get('use_adversarial'):
            kind=recipe.get('adversarial_method','fgsm')
            kwargs={'eps':recipe.get('adversarial_eps',.01),'mean':recipe.get('attack_mean'),'std':recipe.get('attack_std')}
            if kind=='fgsm': attacked=fgsm_attack(model,inputs,labels,**kwargs)
            elif kind=='pgd': attacked=pgd_attack(model,inputs,labels,**kwargs,alpha=recipe.get('adversarial_alpha',.002),steps=recipe.get('adversarial_steps',3))
            else: raise ValueError('Adversarial method must be fgsm or pgd')
            _,other_loss=objective_forward(model,attacked,labels,objective,domains)
            weight=recipe.get('adversarial_weight',1)
            if weight<0 or not math.isfinite(weight): raise ValueError('Adversarial weight must be finite and nonnegative')
            loss=(loss+weight*other_loss)/(1+weight)
        if distillation is not None: loss=distillation.combine(loss,logits,distillation.teacher_logits(inputs))
        if recipe.get('regularization_l1',0) or recipe.get('regularization_l2',0):
            loss=loss+regularization_loss(model,recipe.get('regularization_l1',0),recipe.get('regularization_l2',0))
        return logits,loss
    def update():
        nonlocal updates, skipped, group_samples
        parameters=[p for group in optimizer.param_groups for p in group['params']]
        old_scale=scaler.get_scale()
        if not sharpness: scaler.unscale_(optimizer)
        for parameter in parameters:
            if parameter.grad is not None:
                if sharpness: parameter.grad.div_(old_scale)
                parameter.grad.mul_(nominal_group / group_samples)
        if amp_dtype==torch.bfloat16 and any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
            raise FloatingPointError('Nonfinite BF16 gradients; optimizer update refused')
        if clip_norm is not None:
            nn.utils.clip_grad_norm_(parameters, clip_norm)
        if sharpness:
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                if not enabled or amp_dtype==torch.bfloat16: raise FloatingPointError('Nonfinite SAM first-pass gradients')
                scaler.update(new_scale=old_scale/2);skipped+=1
                optimizer.zero_grad(set_to_none=True);group_samples=0;replay.clear();return
            optimizer.first_step(zero_grad=True)
            end_rng=rng_state()
            batch_norms=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
            tracking=[m.track_running_stats for m in batch_norms]
            try:
                for m in batch_norms: m.track_running_stats=False
                for replay_inputs,replay_labels,replay_domains,replay_rng in replay:
                    restore_rng(replay_rng)
                    with torch.amp.autocast(device.type,enabled=enabled,dtype=amp_dtype):
                        _,second_loss=forward_loss(replay_inputs,replay_labels,replay_domains)
                    if not bool(torch.isfinite(second_loss)): raise FloatingPointError('Nonfinite SAM second-pass loss')
                    scaler.scale(second_loss*(len(replay_labels)/nominal_group)).backward()
                scaler.unscale_(optimizer)
                for parameter in parameters:
                    if parameter.grad is not None: parameter.grad.mul_(nominal_group/group_samples)
                if clip_norm is not None: nn.utils.clip_grad_norm_(parameters,clip_norm)
            finally:
                optimizer.restore_weights()
                for m,value in zip(batch_norms,tracking): m.track_running_stats=value
                restore_rng(end_rng)
            if amp_dtype==torch.bfloat16 and any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in parameters):
                raise FloatingPointError('Nonfinite BF16 SAM gradients; optimizer update refused')
            scaler.step(optimizer,_sam_second_pass=True)
        else:
            scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            skipped += 1
        else:
            updates += 1
            if recipe.get('use_weight_noise'): add_weight_noise(model,recipe.get('weight_noise_std',.01))
            for observer in update_observers: observer.update(model)
            if scheduler is not None:
                scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        group_samples = 0
        replay.clear()
    for batch in loader:
        inputs,labels=batch[:2]
        domains=batch[2].to(device).long() if len(batch)==3 else None
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda").float().reshape(-1, 1)
        original_labels=labels.detach().clone()
        if recipe.get('gaussian_noise_std',0)>0:
            inputs=add_gaussian_noise(inputs,recipe['gaussian_noise_std'])
        if recipe.get('dct_mask_prob',0)>0:
            inputs=dct_mask_batch(inputs,recipe['dct_mask_prob'],recipe.get('dct_num_masks',8),recipe.get('dct_mask_size',8))
        if recipe.get('fouriermix_prob',0)>0:
            inputs=fouriermix_batch(inputs,recipe['fouriermix_prob'],recipe.get('fouriermix_alpha',.1))
        def mix_within_domain(mixer,*args):
            nonlocal inputs,labels
            if domains is None: inputs,labels=mixer(inputs,labels,*args)
            else:
                mixed_inputs,mixed_labels=inputs.clone(),labels.clone()
                for domain in domains.unique(sorted=True):
                    mask=domains==domain
                    mixed_inputs[mask],mixed_labels[mask]=mixer(inputs[mask],labels[mask],*args)
                inputs,labels=mixed_inputs,mixed_labels
        if recipe.get('fft_amplitude_mix_prob',0)>0:
            mix_within_domain(fft_amplitude_mix_batch,recipe['fft_amplitude_mix_prob'],recipe.get('fft_amplitude_mix_alpha',.5))
        if recipe.get('cutmix_alpha',0)>0:
            mix_within_domain(cutmix_batch,recipe['cutmix_alpha'])
        elif recipe.get('mixup_alpha',0)>0:
            mix_within_domain(mixup_batch,recipe['mixup_alpha'])
        batch_size = len(labels)
        if nominal_group is None:
            nominal_group = batch_size * accumulation
        if sharpness: replay.append((inputs.detach(),labels.detach(),domains,rng_state()))
        with torch.amp.autocast(device.type, enabled=enabled,dtype=amp_dtype):
            logits,loss=forward_loss(inputs,labels,domains)
            scaled_loss = loss * (batch_size / nominal_group)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss")
        if not scaled_loss.requires_grad:
            raise NoOptimizerUpdates("no_gradient", "No gradient flowed: loss is detached from trainable parameters")
        scaler.scale(scaled_loss).backward()
        microbatches += 1
        group_samples += batch_size
        samples += batch_size
        loss_sum += float(loss.detach()) * batch_size
        labels_all.extend(original_labels.cpu().reshape(-1).int().tolist())
        logits_all.extend(logits.detach().float().cpu().reshape(-1).tolist())
        if microbatches % accumulation == 0:
            update()
        if progress is not None:
            progress.update(microbatches, f"loss {loss_sum/samples:.4f}")
    if group_samples:
        update()
    if not samples or updates == 0:
        if skipped:
            raise NoOptimizerUpdates("scaler_skipped", f"AMP scaler skipped all {skipped} attempted optimizer updates; gradients flowed but overflow protection prevented stepping")
        raise NoOptimizerUpdates("empty", "Training epoch performed no successful optimizer updates: no input batches")
    if progress is not None:
        progress.finish(microbatches, f"loss {loss_sum/samples:.4f}")
    return {"loss": loss_sum / samples, "samples": samples, "optimizer_updates": updates,
            "skipped_amp_updates": skipped, "labels": labels_all, "logits": logits_all}


def predict_loader(model, loader, *, device="cpu", amp=True, progress=None,amp_dtype='float16'):
    device = torch.device(device)
    enabled,dtype,_=amp_settings(amp,device,{'amp_dtype':amp_dtype})
    model.eval()
    labels_all, logits_all = [], []
    total_loss, samples = 0.0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device).float().reshape(-1, 1)
            with torch.amp.autocast(device.type, enabled=enabled,dtype=dtype):
                logits = model(inputs)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite validation output")
            total_loss += float(loss) * len(labels)
            samples += len(labels)
            labels_all.extend(labels.cpu().reshape(-1).int().tolist())
            logits_all.extend(logits.float().cpu().reshape(-1).tolist())
            if progress is not None:
                progress.update(samples)
    if not samples:
        raise ValueError("Empty validation/evaluation loader")
    if progress is not None:
        progress.finish(samples)
    return {"labels": labels_all, "logits": logits_all, "loss": total_loss / samples}


def build_param_groups(model,base_lr,layer_decay):
    if not math.isfinite(base_lr) or base_lr<=0 or not math.isfinite(layer_decay) or not 0<layer_decay<=1:
        raise ValueError('Base LR must be positive and layer decay must be in(0,1]')
    classifier=list(model.head_parameters() if hasattr(model,'head_parameters') else model.get_classifier().parameters())
    classifier_ids={id(p) for p in classifier}
    base=getattr(model,'base',model); prefix='base.' if base is not model else ''
    stages=[]
    for name,child in base.named_children():
        if name=='blocks' and isinstance(child,(nn.Sequential,nn.ModuleList)):
            stages.extend((prefix+f'blocks.{key}',block) for key,block in child.named_children())
        else: stages.append((prefix+name,child))
    covered=set(classifier_ids); groups=[]
    # Root-owned tensors (e.g. positional embeddings) precede stages.
    root=[p for p in base.parameters(recurse=False) if id(p) not in covered]
    if root: groups.append({'params':root,'lr':base_lr,'layer_depth':0,'backbone_layer':'root'}); covered.update(map(id,root))
    for name,stage in stages:
        parameters=[p for p in stage.parameters() if id(p) not in covered]
        if not parameters: continue
        depth=len(groups)
        groups.append({'params':parameters,'lr':base_lr*layer_decay**depth,'layer_depth':depth,'backbone_layer':name})
        covered.update(map(id,parameters))
    if covered!={id(p) for p in model.parameters()}: raise ValueError('Layer LR groups did not cover every model parameter exactly once')
    groups.append({'params':classifier,'lr':base_lr})
    return groups


def _optimizer(model, cfg, phase,auxiliary_parameters=()):
    recipe = cfg["training"]
    classifier = list(model.head_parameters() if hasattr(model,'head_parameters') else model.get_classifier().parameters())
    classifier_ids = {id(p) for p in classifier}
    for parameter in model.parameters():
        parameter.requires_grad_(phase == "finetune" or id(parameter) in classifier_ids)
    if phase == "head":
        groups = [{"params": classifier, "lr": recipe["head_lr"]}]
    else:
        decay=recipe.get('layer_lr_decay',1)
        if not 0<decay<=1: raise ValueError('Layer LR decay must be in (0,1]')
        if decay==1 and not recipe.get('layer_warmup_epochs',0): groups=[{'params':[p for p in model.parameters() if id(p) not in classifier_ids],'lr':recipe['backbone_lr']}]
        else:
            groups=build_param_groups(model,recipe['backbone_lr'],decay)[:-1]
        groups.append({'params':classifier,'lr':recipe['finetune_head_lr']})
    auxiliary=[p for p in auxiliary_parameters if p.requires_grad]
    if auxiliary: groups.append({'params':auxiliary,'lr':recipe['head_lr'] if phase=='head' else recipe['finetune_head_lr']})
    return make_optimizer(groups,recipe)


def backbone_blocks(model):
    base=getattr(model,'base',model)
    xception=[(name,module) for name,module in base.named_children() if re.fullmatch(r'block\d+',name)]
    if xception: return xception
    blocks=getattr(base,'blocks',None)
    if isinstance(blocks,(nn.Sequential,nn.ModuleList)): return [(f'blocks.{name}',module) for name,module in blocks.named_children()]
    raise ValueError('Gradual unfreezing needs identifiable backbone blocks')


def apply_unfreeze_schedule(model,cfg,phase,epoch):
    schedule=cfg['training'].get('unfreeze_schedule','all')
    if schedule not in ('all','gradual','cascade'): raise ValueError('Unfreeze schedule must be all, gradual or cascade')
    if phase not in ('head','finetune') or epoch<0: raise ValueError('Invalid unfreeze phase/epoch')
    head=list(model.head_parameters() if hasattr(model,'head_parameters') else model.get_classifier().parameters())
    trainable={id(p) for p in head}
    if phase=='finetune':
        if schedule=='all' or (schedule=='gradual' and epoch>=3): trainable.update(id(p) for p in model.parameters())
        elif schedule=='cascade':
            interval=cfg['training'].get('cascade_epochs_per_stage',1)
            if interval<1: raise ValueError('Cascade interval must be positive')
            blocks=backbone_blocks(model); count=min(len(blocks),epoch//interval+1)
            if count==len(blocks): trainable.update(id(p) for p in model.parameters())
            else:
                for _,block in blocks[-count:]: trainable.update(id(p) for p in block.parameters())
        else:
            for _,block in backbone_blocks(model)[-3:]: trainable.update(id(p) for p in block.parameters())
    for parameter in model.parameters(): parameter.requires_grad_(id(parameter) in trainable)
    return [name for name,parameter in model.named_parameters() if parameter.requires_grad]


def progressive_size(cfg,epoch):
    recipe=cfg['training']
    if not (recipe.get('progressive_resizing') or cfg.get('pretraining',{}).get('progressive_resizing')): return cfg['model']['input_size']
    sizes=recipe.get('progressive_sizes',[224,256,299]); interval=recipe.get('progressive_interval',6)
    if epoch<0 or interval<1 or not sizes or any(not isinstance(size,int) or size<1 for size in sizes) or sizes!=sorted(set(sizes)) or sizes[-1]!=cfg['model']['input_size']:
        raise ValueError('Progressive sizes must increase to the final model input size, with a positive interval')
    return sizes[min(epoch//interval,len(sizes)-1)]


def _scheduler(optimizer,cfg,steps_per_epoch):
    from deepfake_detection.training.schedulers import make_scheduler
    return make_scheduler(optimizer,cfg['training'],steps_per_epoch)


def _history(path, rows, ws):
    stream = StringIO(newline="")
    fields = ["phase", "epoch", "train_loss", "validation_loss", "validation_macro_auc", "optimizer_updates", "seconds"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: row[k] for k in fields} for row in rows)
    atomic_bytes(path, stream.getvalue().encode("utf-8"), ws)


def run_training(name, cfg, rows, workspace=None, *, device=None, resume=None, max_batches=None, skip_integrity_check=False):
    run_started=time.perf_counter()
    cfg = copy.deepcopy(cfg)
    cfg["pilot"] = bool(cfg.get("pilot") or max_batches is not None or any(r.get("pilot") for r in rows))
    cfg["integrity_check_skipped"] = bool(skip_integrity_check)
    ws = workspace or Workspace()
    run = ws.run(name)
    recipe = cfg["training"]
    if recipe['loss'] not in LOSS_REGISTRY: raise ValueError(f"Unknown training loss {recipe['loss']}")
    if any(r.get("split") == "external" or r.get("dataset") == "dfd" for r in rows):
        raise ValueError("External/OOD manifests cannot enter a training run")
    audit = check_splits(rows)
    if not audit["ok"]:
        raise ValueError(f"Split check failed: {audit}")
    training_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "val"]
    if not training_rows or not validation_rows:
        raise ValueError("Training and early-stopping validation rows are required")
    for domain in {r["dataset"] for r in validation_rows}:
        if {r["label"] for r in validation_rows if r["dataset"] == domain} != {0, 1}:
            raise ValueError(f"Validation domain {domain} needs both classes for macro AUC")
    manifest_sha = fingerprint(sorted(rows, key=lambda r: r["sample_id"]))
    domain_map=None
    adaptation=cfg.get('domain_adaptation',{})
    if adaptation.get('test_time_adaptation'):
        raise ValueError('Test-time adaptation requires the explicit development-only tent_adaptation API; it cannot run implicitly during training')
    if adaptation.get('coral_weight',0)!=0:
        raise ValueError('Use domain_adaptation.method=coral with training.domain_adaptation_weight; coral_weight is not an active selector')
    domain_weight=recipe.get('domain_adaptation_weight',recipe.get('domain_loss_weight',0))
    if 'domain_adaptation_weight' in recipe and 'domain_loss_weight' in recipe and recipe['domain_loss_weight']!=domain_weight:
        raise ValueError('Conflicting domain adaptation weight selectors')
    if domain_weight>0: recipe['domain_loss_weight']=domain_weight
    if recipe.get('domain_loss_weight',0)>0:
        adaptation=cfg.get('domain_adaptation',{})
        recipe['domain_method']=adaptation.get('method','none')
        recipe['grl_lambda']=adaptation.get('grl_lambda',.1)
        recipe['mmd_kernel']=adaptation.get('mmd_kernel','rbf')
        domain_map={name:index for index,name in enumerate(sorted({r['dataset'] for r in training_rows}))}
        recipe['domain_map']=domain_map
        if recipe['domain_method'] not in {'dann','mmd','coral'} or len(domain_map)<2:
            raise ValueError('Enabled domain adaptation requires a supported method and at least two training domains')
    teacher_spec=distillation_spec(cfg,rows,ws)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    _,_,scale_enabled=amp_settings(recipe['amp'],device,recipe)
    if run.exists() and any(p.name != "run.log" for p in run.iterdir()) and resume is None:
        raise FileExistsError("Run exists; choose a new run ID or explicitly resume a phase")
    if resume not in {None, "head", "finetune"}:
        raise ValueError("Resume phase must be head or finetune")
    if resume:
        if not (run / "config.json").exists() or read_json(run / "config.json") != cfg:
            raise ValueError("Saved run configuration does not match requested configuration")
        if resume == "head" and (run / "finetune_latest.pt").exists():
            raise ValueError("Run already entered fine-tuning; resume finetune explicitly")
    resume_state = None
    if resume:
        resume_state = load_checkpoint(run / f"{resume}_latest.pt", resume_signature(cfg, manifest_sha, resume, device), workspace=ws, device="cpu")
        if resume_state.get("pilot_max_batches") != max_batches:
            raise ValueError("Pilot truncation cannot change on resume; use a new run")
        artifact_metadata(run,cfg,ws)
        artifact_metadata(run,cfg,ws,phase=resume)
        if resume=="finetune": artifact_metadata(run,cfg,ws,phase="head")
        if resume == "finetune" and (resume_state["epoch"] + 1 >= recipe["finetune_epochs"] or resume_state["bad_epochs"] >= recipe["patience"]):
            raise ValueError("Fine-tuning is already complete; no epochs remain")
    from deepfake_detection.data.integrity import preflight_crops
    integrity = {"skipped":True,"reason":"explicit_emergency_bypass"} if skip_integrity_check else preflight_crops(rows,cfg,ws)
    seed_all(cfg["seed"])
    train_set = CropDataset(training_rows, cfg, training=True, workspace=ws,domain_map=domain_map)
    val_set = CropDataset(validation_rows, cfg, workspace=ws)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "integrity_report.json",integrity,ws)
    if not resume:
        write_json(run / "config.json", cfg, ws)
        write_json(run / "split_audit.json", audit, ws)
        exposure={"hash":manifest_sha,"rows":rows}
        if teacher_spec is not None: exposure['teacher_exposure']=teacher_spec['exposure_rows']
        write_json(run / "input_manifest.json", exposure, ws)
    # Resume never depends on the original ImageNet initialization file being present.
    model = architecture(cfg["model"]).to(device) if resume else initialize_model(cfg, device=device)
    objective=objective_for_model(model,recipe).to(device)
    distillation=load_distillation(cfg,teacher_spec,ws,device)
    ema=WeightEMA(model,recipe.get('ema_decay',.999)) if recipe.get('use_ema') else None
    swa=StochasticWeightAveraging(model,recipe.get('swa_start_epoch',15)) if recipe.get('use_swa') else None
    window_average=WeightAveraging(model,recipe.get('weight_average_window',5)) if recipe.get('use_weight_averaging') else None
    global_best = float("-inf")
    history = []
    total_updates = 0
    phases = ["finetune"] if resume == "finetune" else ["head", "finetune"]
    try:
        for phase in phases:
            optimizer = _optimizer(model, cfg, phase,objective.parameters())
            count = recipe["head_epochs"] if phase == "head" else recipe["finetune_epochs"]
            expected_steps = math.ceil(math.ceil(recipe["epoch_samples"] / recipe["batch_size"]) / recipe["accumulation"])
            scheduler = _scheduler(optimizer, cfg, expected_steps) if phase == "finetune" else None
            scaler = torch.amp.GradScaler("cuda", enabled=scale_enabled)
            signature = resume_signature(cfg, manifest_sha, phase, device)
            start_epoch, phase_best, bad_epochs = 0, float("-inf"), 0
            if resume == phase:
                state = resume_state
                model.load_state_dict(state["model_state_dict"])
                objective.load_state_dict(state.get('loss_state_dict',{}),strict=True)
                if ema is not None: ema.load_state_dict(state['ema_state_dict'])
                if swa is not None: swa.load_state_dict(state['swa_state_dict'])
                if window_average is not None: window_average.load_state_dict(state['weight_average_state_dict'])
                optimizer.load_state_dict(state["optimizer_state_dict"])
                scaler.load_state_dict(state["scaler_state_dict"])
                if scheduler is not None:
                    scheduler.load_state_dict(state["scheduler_state_dict"])
                start_epoch = state["epoch"] + 1
                phase_best, global_best = state["best_val_auc"], state["global_best_auc"]
                committed = run / "best_metadata.json"
                if committed.exists():
                    # Validate the committed generation; preserve a best published before a crash.
                    best = artifact_metadata(run,cfg,ws)
                    global_best = max(global_best, best["validation_macro_auc"])
                    if best["phase"] == phase:
                        phase_best = max(phase_best, best["validation_macro_auc"])
                phase_best=max(phase_best,artifact_metadata(run,cfg,ws,phase=phase)["validation_macro_auc"])
                bad_epochs, history, total_updates = state["bad_epochs"], state["history"], state["total_optimizer_updates"]
                restore_rng(state["rng"])
                if phase == "finetune" and (start_epoch >= count or bad_epochs >= recipe["patience"]):
                    raise ValueError("Fine-tuning is already complete; no epochs remain")
            for epoch in range(start_epoch, count):
                started = time.perf_counter()
                apply_unfreeze_schedule(model,cfg,phase,epoch)
                if phase=='finetune' and recipe.get('layer_warmup_epochs',0): scheduler.set_layer_warmup(epoch,recipe['layer_warmup_epochs'])
                ordinal = epoch + 1 + (recipe["head_epochs"] if phase == "finetune" else 0)
                input_size=progressive_size(cfg,ordinal-1)
                if input_size!=train_set.cfg['model']['input_size']:
                    train_cfg=copy.deepcopy(cfg); train_cfg['model']['input_size']=input_size
                    train_set=CropDataset(training_rows,train_cfg,training=True,workspace=ws,domain_map=domain_map)
                sampler = HierarchicalSampler(training_rows, samples=recipe["epoch_samples"], seed=cfg["seed"])
                sampler.set_epoch(epoch + (10000 if phase == "finetune" else 0))
                generator = torch.Generator().manual_seed(cfg["seed"] + epoch + (10000 if phase == "finetune" else 0))
                loader = DataLoader(train_set, batch_size=recipe["batch_size"], sampler=sampler, num_workers=recipe["workers"],
                                    pin_memory=bool(recipe["pin_memory"] and device.type == "cuda"), worker_init_fn=seed_worker, generator=generator)
                if max_batches is not None:
                    import itertools
                    loader = itertools.islice(loader, max_batches)
                total_batches = math.ceil(recipe["epoch_samples"] / recipe["batch_size"])
                if max_batches is not None:
                    total_batches = min(total_batches, max_batches)
                previous_auc = f"{history[-1]['validation_macro_auc']:.4f}" if history else "pending"
                batch_progress = Progress(f"Epoch {ordinal}/{recipe['head_epochs']+recipe['finetune_epochs']} · batch", total_batches, suffix=f" · val AUC {previous_auc} (previous)")
                epoch_recipe={**recipe,'attack_mean':cfg['model']['mean'],'attack_std':cfg['model']['std']}
                training = train_epoch(model, loader, optimizer, device=device, accumulation=recipe["accumulation"],
                                       amp=recipe["amp"], scaler=scaler, scheduler=scheduler, clip_norm=recipe["clip_norm"], freeze_statistics=recipe["freeze_bn"], progress=batch_progress,recipe=epoch_recipe,objective=objective,update_observers=[ema] if ema else (),distillation=distillation)
                total_updates += training["optimizer_updates"]
                validation_loader = DataLoader(val_set, batch_size=recipe["batch_size"], shuffle=False, num_workers=0)
                if swa is not None: swa.update(model,epoch=ordinal)
                if window_average is not None: window_average.update(model,ordinal)
                candidates={}
                averages={'raw':None,**({'ema':ema} if ema else {}),**({'swa':swa} if swa is not None and swa.updates else {}),**({'window':window_average} if window_average else {})}
                for variant,average in averages.items():
                    with average.average_parameters(model) if average else nullcontext():
                        if variant=='swa' and recipe.get('use_swa_final_bn_update',False):
                            bn_set=CropDataset(training_rows,cfg,training=False,workspace=ws)
                            bn_loader=DataLoader(bn_set,batch_size=recipe['batch_size'],shuffle=False,num_workers=0)
                            if max_batches is not None:
                                source_bn_loader=bn_loader
                                class BoundedLoader:
                                    dataset=bn_set
                                    def __iter__(self):
                                        import itertools
                                        return itertools.islice(source_bn_loader,max_batches)
                                bn_loader=BoundedLoader()
                            bn_rng=rng_state()
                            try: swa.bn_update(bn_loader,device,model=model)
                            finally: restore_rng(bn_rng)
                        validation=predict_loader(model,validation_loader,device=device,amp=recipe['amp'],amp_dtype=recipe.get('amp_dtype','float16'))
                    scores=real_probability(validation['logits']); domains={}
                    for domain in sorted({r['dataset'] for r in validation_rows}):
                        indices=[i for i,r in enumerate(validation_rows) if r['dataset']==domain]
                        domains[domain]=compute_metrics([validation['labels'][i] for i in indices],scores[indices])['roc_auc']
                    candidates[variant]={'validation':validation,'domains':domains,'score':sum(domains.values())/len(domains)}
                selected=max(candidates,key=lambda key:candidates[key]['score'])
                validation=candidates[selected]['validation']; domains=candidates[selected]['domains']; score=candidates[selected]['score']
                if scheduler is not None: scheduler.step_metric(score)
                entry = {"phase": phase, "epoch": epoch + 1, "train_loss": training["loss"], "validation_loss": validation["loss"],
                         "validation_macro_auc": score, "validation_domain_auc": domains, "optimizer_updates": training["optimizer_updates"],
                         "seconds": time.perf_counter() - started}
                entry['weight_variant']=selected; entry['candidate_validation_auc']={key:value['score'] for key,value in candidates.items()}
                entry['training_input_size']=input_size
                history.append(entry)
                snapshot_interval=recipe.get('snapshot_interval_epochs',0)
                if snapshot_interval and phase=='finetune' and (epoch+1)%snapshot_interval==0:
                    from deepfake_detection.models.ensembles import publish_snapshot
                    publish_snapshot(model,run,cfg,epoch=epoch+1,workspace=ws)
                if score > phase_best:
                    phase_best, bad_epochs = score, 0
                    with averages[selected].average_parameters(model) if averages[selected] else nullcontext():
                        publish_best(model,run,cfg,{"phase":phase,"epoch":epoch+1,"validation_macro_auc":score,
                                     "manifest_sha256":manifest_sha,"pilot":cfg["pilot"],'weight_variant':selected},ws,phase=phase)
                else:
                    bad_epochs += 1
                if score > global_best:
                    global_best = score
                    with averages[selected].average_parameters(model) if averages[selected] else nullcontext():
                        publish_best(model, run, cfg, {"run_id": name, "phase": phase, "epoch": epoch + 1,
                                    "validation_macro_auc": score, "manifest_sha256": manifest_sha,
                                    "preprocessing_hash": signature["preprocessing_hash"], "seed": cfg["seed"],
                                    "class_map": cfg["model"]["class_map"], "pilot": cfg["pilot"],'weight_variant':selected}, ws)
                # Save AFTER both best scores change; the checkpoint never contains the stale floor bug.
                state = {"signature": signature, "epoch": epoch, "model_state_dict": model.state_dict(),
                         'loss_state_dict':objective.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                         "scaler_state_dict": scaler.state_dict(), "rng": rng_state(), "best_val_auc": phase_best,
                         "global_best_auc": global_best, "bad_epochs": bad_epochs, "history": history,
                         "total_optimizer_updates": total_updates, "pilot_max_batches": max_batches}
                if ema is not None: state['ema_state_dict']=ema.state_dict()
                if swa is not None: state['swa_state_dict']=swa.state_dict()
                if window_average is not None: state['weight_average_state_dict']=window_average.state_dict()
                save_checkpoint(run / f"{phase}_latest.pt", state, ws)
                _history(run / "history.csv", history, ws)
                write_json(run / "history.json", history, ws)
                console(f"{name}: {phase} epoch {epoch + 1}/{count}; loss={training['loss']:.5f}; validation macro AUC={score:.5f}")
                if phase == "finetune" and bad_epochs >= recipe["patience"]:
                    break
            if phase == "head":
                from safetensors.torch import load_file
                head=artifact_metadata(run,cfg,ws,phase="head")
                model.load_state_dict(load_file(str(run / head["weights_file"]), device=str(device)))
                if ema is not None: ema=WeightEMA(model,recipe.get('ema_decay',.999))
        result = {"status": "completed", "run_id": name, "best_validation_macro_auc": global_best,
                  "elapsed_seconds":time.perf_counter()-run_started,
                  "best_checkpoint_path":str(run/artifact_metadata(run,cfg,ws)["weights_file"]),
                  "optimizer_updates": total_updates, "epochs": len(history), "pilot": cfg["pilot"]}
        if cfg.get('evaluation',{}).get('save_learning_curves'):
            from deepfake_detection.evaluation.analysis import save_learning_curves
            result['learning_curves']=save_learning_curves(run/'history.json',run,workspace=ws)
        write_json(run / "status.json", result, ws)
        return result
    except Exception as exc:
        write_json(run / "status.json", {"status": "failed", "run_id": name,"pilot":cfg["pilot"], "error": f"{type(exc).__name__}: {exc}"}, ws)
        raise
