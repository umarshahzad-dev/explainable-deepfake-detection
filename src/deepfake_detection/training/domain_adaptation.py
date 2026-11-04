"""Development-domain adaptation; never implicitly consumes held-out data."""
import math
from contextlib import contextmanager
import torch
from torch import nn


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lambda_): ctx.lambda_=lambda_; return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient): return -ctx.lambda_*gradient,None


def grad_reverse(x,lambda_=.1):
    if not math.isfinite(lambda_) or lambda_<0: raise ValueError('Gradient reversal coefficient must be finite and nonnegative')
    return GradientReversalLayer.apply(x,lambda_)


class DANNModel(nn.Module):
    """Binary classifier and adversarial domain head sharing embeddings."""
    def __init__(self,backbone,classifier,domain_classifier):
        super().__init__()
        self.backbone=backbone; self.classifier=classifier; self.domain_classifier=domain_classifier

    def forward(self,x,alpha=.1):
        features=self.backbone(x)
        if features.ndim>2: features=features.mean(tuple(range(2,features.ndim)))
        return self.classifier(features),self.domain_classifier(grad_reverse(features,alpha))


def _validate_features(source,target):
    if source.ndim!=2 or target.ndim!=2 or source.shape[1]!=target.shape[1] or not len(source) or not len(target):
        raise ValueError('Domain features must be nonempty NxD matrices with matching dimensions')


def mmd_loss(source_features,target_features,kernel='rbf'):
    """Biased nonnegative MMD estimator; fixed multiscale RBF bandwidths."""
    _validate_features(source_features,target_features)
    source=source_features.float(); target=target_features.float()
    if kernel=='linear': return (source.mean(0)-target.mean(0)).square().sum()
    if kernel!='rbf': raise ValueError('MMD kernel must be rbf or linear')
    def gram(a,b):
        distance=torch.cdist(a,b).square()
        return sum(torch.exp(-distance/(2*bandwidth**2)) for bandwidth in (.5,1.,2.,4.))/4
    return (gram(source,source).mean()+gram(target,target).mean()-2*gram(source,target).mean()).clamp_min(0)


def domain_alignment(features,domains,method,kernel='rbf'):
    if method not in {'mmd','coral'}: raise ValueError('Unknown feature alignment method')
    groups=[features[domains==domain] for domain in domains.unique(sorted=True)]
    penalty=(lambda a,b:mmd_loss(a,b,kernel)) if method=='mmd' else coral_loss
    values=[penalty(a,b) for index,a in enumerate(groups) for b in groups[index+1:]]
    return torch.stack(values).mean() if values else features.sum()*0


def coral_loss(source_features,target_features):
    """Squared sample covariance discrepancy, normalized by 4D²."""
    _validate_features(source_features,target_features)
    if min(len(source_features),len(target_features))<2:
        return (source_features.sum()+target_features.sum())*0
    def covariance(features):
        centered=features.float()-features.float().mean(0)
        return centered.T@centered/(len(features)-1)
    return (covariance(source_features)-covariance(target_features)).square().sum()/(4*source_features.shape[1]**2)


def _development_rows(dataset):
    rows=getattr(dataset,'rows',None)
    if not rows or len(rows)!=len(dataset): raise ValueError('Adaptation requires complete explicit development-data provenance')
    if any(row.get('split') not in {'dev','adaptation'} or row.get('dataset')=='dfd' for row in rows):
        raise ValueError('Adaptation accepts dedicated development data only; held-out evaluation/DFD cannot be tuned')
    return rows


@contextmanager
def _adapt_parameters(model,parameters):
    selected={id(p) for p in parameters}
    flags=[p.requires_grad for p in model.parameters()]
    modes=[m.training for m in model.modules()]
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    try:
        model.eval()
        for p in model.parameters(): p.requires_grad_(id(p) in selected)
        yield
    except BaseException:
        model.load_state_dict(before)
        raise
    finally:
        for p,flag in zip(model.parameters(),flags): p.requires_grad_(flag); p.grad=None
        for m,mode in zip(model.modules(),modes): m.training=mode


def _head_parameters(model):
    if not hasattr(model,'get_classifier'): raise ValueError('Adaptation requires an exposed classifier head')
    params=list(model.get_classifier().parameters())
    if not params: raise ValueError('Classifier head has no adaptable parameters')
    return params


def few_shot_anchor(model,target_samples,num_anchor=50):
    """One head-only pass on explicitly labeled target development anchors."""
    rows=_development_rows(target_samples)
    if not isinstance(num_anchor,int) or num_anchor<1: raise ValueError('Anchor count must be positive')
    count=min(num_anchor,len(target_samples)); parameters=_head_parameters(model)
    device=next(model.parameters()).device; losses=[]
    with _adapt_parameters(model,parameters):
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(target_samples,range(count)),batch_size=min(8,count),shuffle=False,num_workers=0)
        for batch in loader:
            inputs,targets=batch[:2]; inputs=inputs.to(device); targets=targets.to(device).float().reshape(-1,1)
            if not bool(((targets==0)|(targets==1)).all()): raise ValueError('Anchors require binary labels fake=0 real=1')
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(inputs),targets)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite anchor loss')
            loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
    return {'method':'few_shot_anchor','samples':count,'loss':sum(losses)/len(losses),'fit_sample_ids':[r.get('sample_id') for r in rows[:count]],'evaluation_claim':'target-development adapted; not zero-shot'}


def tent_adaptation(model,target_loader,steps=10,lr=1e-4):
    """Bounded BN-affine entropy adaptation on a dedicated development set."""
    rows=_development_rows(target_loader.dataset)
    if not isinstance(steps,int) or steps<1 or steps>10000 or not math.isfinite(lr) or lr<=0:
        raise ValueError('TENT requires 1..10000 steps and positive finite LR')
    norms=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm) and m.affine]
    if not norms: raise ValueError('TENT requires affine BatchNorm layers')
    parameters=[p for norm in norms for p in (norm.weight,norm.bias)]
    tracking=[norm.track_running_stats for norm in norms]; device=next(model.parameters()).device
    losses=[]; updates=0
    with _adapt_parameters(model,parameters):
        try:
            for norm in norms: norm.train(); norm.track_running_stats=False
            optimizer=torch.optim.Adam(parameters,lr=lr)
            iterator=iter(target_loader)
            for _ in range(steps):
                try: batch=next(iterator)
                except StopIteration:
                    iterator=iter(target_loader)
                    try: batch=next(iterator)
                    except StopIteration: raise ValueError('Adaptation loader is empty')
                inputs=(batch[0] if isinstance(batch,(tuple,list)) else batch).to(device)
                if len(inputs)<2: continue
                optimizer.zero_grad(set_to_none=True)
                logits=model(inputs).float()
                if logits.shape!=(len(inputs),1): raise ValueError('TENT requires one binary logit per input')
                probabilities=logits.sigmoid()
                entropy=(torch.nn.functional.softplus(logits)-probabilities*logits).mean()
                if not bool(torch.isfinite(entropy)): raise FloatingPointError('Nonfinite adaptation entropy')
                entropy.backward(); optimizer.step(); losses.append(float(entropy.detach())); updates+=1
            if not updates: raise ValueError('TENT requires batches with at least two samples')
        finally:
            for norm,value in zip(norms,tracking): norm.track_running_stats=value
    return {'method':'tent','updates':updates,'entropy':sum(losses)/len(losses),'fit_sample_ids':[r.get('sample_id') for r in rows],'evaluation_claim':'target-development adapted; not zero-shot'}


def self_train(model,unlabeled_loader,threshold=.9):
    """One bounded head-only pseudo-label pass on explicit development data."""
    rows=_development_rows(unlabeled_loader.dataset)
    if not math.isfinite(threshold) or not .5<threshold<1: raise ValueError('Pseudo-label confidence must be strictly between .5 and 1')
    device=next(model.parameters()).device; parameters=_head_parameters(model)
    accepted=updates=0; counts={'fake':0,'real':0}
    with _adapt_parameters(model,parameters):
        optimizer=torch.optim.AdamW(parameters,lr=1e-4,weight_decay=0)
        for batch in unlabeled_loader:
            inputs=(batch[0] if isinstance(batch,(tuple,list)) else batch).to(device)
            with torch.no_grad():
                logits=model(inputs).float()
                if logits.shape!=(len(inputs),1): raise ValueError('Self-training requires one binary logit per input')
                if not bool(torch.isfinite(logits).all()): raise FloatingPointError('Nonfinite pseudo-label logits')
                probability=logits.sigmoid()
                mask=(torch.maximum(probability,1-probability)>=threshold).reshape(-1)
                targets=(probability[mask]>=.5).float()
            if not mask.any(): continue
            optimizer.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(inputs[mask]),targets)
            if not bool(torch.isfinite(loss)): raise FloatingPointError('Nonfinite self-training loss')
            loss.backward(); optimizer.step(); updates+=1; accepted+=len(targets)
            counts['real']+=int(targets.sum()); counts['fake']+=len(targets)-int(targets.sum())
    return {'method':'self_training','accepted':accepted,'updates':updates,'pseudo_class_counts':counts,'fit_sample_ids':[r.get('sample_id') for r in rows],'evaluation_claim':'target-development adapted; not zero-shot'}


def maml_adaptation(model,tasks):
    raise NotImplementedError('MAML requires a separately designed episodic meta-training dataset and higher-order optimization; no implicit target-test adaptation')
