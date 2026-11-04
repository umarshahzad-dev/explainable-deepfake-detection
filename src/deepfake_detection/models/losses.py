"""Binary, embedding and paired-context objectives; REAL logit and fake=0 labels.

Embedding losses take NxD features, not scalar classification logits. Frequency
loss takes spatial feature maps. Their training adapters retain supervised BCE.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def _targets(logits,targets):
    targets=torch.as_tensor(targets,device=logits.device,dtype=logits.dtype).reshape_as(logits)
    if not torch.isfinite(targets).all() or (targets<0).any() or (targets>1).any(): raise ValueError('Binary targets must be finite in [0,1]')
    return targets


class LabelSmoothingBCE(nn.Module):
    def __init__(self,eps=.05):
        super().__init__()
        if not 0<=eps<1: raise ValueError('Label smoothing epsilon must be in [0,1)')
        self.eps=eps
    def forward(self,logits,targets):
        targets=_targets(logits,targets)
        return F.binary_cross_entropy_with_logits(logits,(1-self.eps)*targets+self.eps*.5)


class FocalLoss(nn.Module):
    """Alpha weights REAL=1;1-alpha weights FAKE=0. Soft targets supported."""
    def __init__(self,alpha=.25,gamma=2):
        super().__init__()
        if not 0<=alpha<=1 or gamma<0: raise ValueError('Invalid focal alpha/gamma')
        self.alpha,self.gamma=alpha,gamma
    def forward(self,logits,targets):
        targets=_targets(logits,targets)
        bce=F.binary_cross_entropy_with_logits(logits,targets,reduction='none')
        probabilities=torch.sigmoid(logits)
        pt=targets*probabilities+(1-targets)*(1-probabilities)
        weights=targets*self.alpha+(1-targets)*(1-self.alpha)
        return (weights*(1-pt).pow(self.gamma)*bce).mean()


class WeightedBCE(nn.Module):
    def __init__(self,fake_weight=1,real_weight=1):
        super().__init__()
        if fake_weight<=0 or real_weight<=0: raise ValueError('Class weights must be positive')
        self.fake_weight,self.real_weight=fake_weight,real_weight
    def forward(self,logits,targets):
        targets=_targets(logits,targets)
        weight=targets*self.real_weight+(1-targets)*self.fake_weight
        return (F.binary_cross_entropy_with_logits(logits,targets,reduction='none')*weight).mean()


class LovaszHingeLoss(nn.Module):
    supports_soft_labels=False
    def forward(self,logits,targets):
        targets=_targets(logits,targets).flatten()
        if not ((targets==0)|(targets==1)).all(): raise ValueError('Lovasz hinge requires hard binary targets; disable batch mixing')
        target_fake=1-targets;fake_logits=-logits.flatten()
        errors=1-fake_logits*(2*target_fake-1)
        sorted_errors,order=errors.sort(descending=True)
        truth=target_fake[order]
        positives=truth.sum()
        intersection=positives-truth.cumsum(0);union=positives+(1-truth).cumsum(0)
        gradient=1-intersection/union.clamp_min(1)
        if len(gradient)>1: gradient=torch.cat([gradient[:1],gradient[1:]-gradient[:-1]])
        return torch.dot(F.relu(sorted_errors),gradient)


class DiceLoss(nn.Module):
    def __init__(self,eps=1e-6): super().__init__();self.eps=eps
    def forward(self,logits,targets):
        targets=1-_targets(logits,targets);probability=torch.sigmoid(-logits)
        return 1-(2*(probability*targets).sum()+self.eps)/(probability.sum()+targets.sum()+self.eps)


class TverskyLoss(nn.Module):
    def __init__(self,alpha=.3,beta=.7,eps=1e-6):
        super().__init__()
        if alpha<0 or beta<0 or alpha+beta==0: raise ValueError('Tversky weights must be nonnegative and nonzero')
        self.alpha,self.beta,self.eps=alpha,beta,eps
    def forward(self,logits,targets):
        truth=1-_targets(logits,targets);probability=torch.sigmoid(-logits)
        tp=(probability*truth).sum();fp=(probability*(1-truth)).sum();fn=((1-probability)*truth).sum()
        return 1-(tp+self.eps)/(tp+self.alpha*fp+self.beta*fn+self.eps)


def _embedding(features,targets):
    if features.ndim!=2 or features.shape[1]<2: raise ValueError('Metric loss requires NxD embeddings with D>=2, not scalar binary logits')
    labels=torch.as_tensor(targets,device=features.device).flatten()
    if len(labels)!=len(features) or not ((labels==0)|(labels==1)).all(): raise ValueError('Metric losses require matching hard class labels')
    return features.float(),labels.long()


class ContrastiveLoss(nn.Module):
    kind='embedding'
    def __init__(self,margin=1): super().__init__();self.margin=margin
    def forward(self,logits,targets):
        features,labels=_embedding(logits,targets)
        if len(features)<2: return features.sum()*0
        distance=torch.cdist(features,features)
        same=labels[:,None]==labels[None,:]
        pairs=torch.triu(torch.ones_like(same),diagonal=1).bool()
        losses=torch.where(same,distance.square(),F.relu(self.margin-distance).square())
        return losses[pairs].mean()


class TripletLoss(nn.Module):
    kind='embedding'
    def __init__(self,margin=1): super().__init__();self.margin=margin
    def forward(self,logits,targets):
        features,labels=_embedding(logits,targets);terms=[]
        distance=torch.cdist(features,features)
        for i in range(len(features)):
            positives=(labels==labels[i]).nonzero().flatten();positives=positives[positives!=i]
            negatives=(labels!=labels[i]).nonzero().flatten()
            if len(positives) and len(negatives):
                terms.append(F.relu(distance[i,positives].max()-distance[i,negatives].min()+self.margin))
        return torch.stack(terms).mean() if terms else features.sum()*0


class ArcFaceLoss(nn.Module):
    kind='embedding'
    def __init__(self,embedding_dim,num_classes=2,margin=.5,scale=32):
        super().__init__();self.margin,self.scale=margin,scale
        self.weight=nn.Parameter(torch.empty(num_classes,embedding_dim));nn.init.xavier_uniform_(self.weight)
    def forward(self,logits,targets):
        features,labels=_embedding(logits,targets)
        with torch.amp.autocast(features.device.type,enabled=False):
            cosine=F.linear(F.normalize(features,dim=1),F.normalize(self.weight.float(),dim=1)).clamp(-1+1e-6,1-1e-6)
            sine=torch.sqrt((1-cosine.square()).clamp_min(1e-6))
            phi=cosine*math.cos(self.margin)-sine*math.sin(self.margin)
            phi=torch.where(cosine>math.cos(math.pi-self.margin),phi,cosine-math.sin(math.pi-self.margin)*self.margin)
            mask=F.one_hot(labels,num_classes=self.weight.shape[0]).bool()
            return F.cross_entropy(torch.where(mask,phi,cosine)*self.scale,labels)


class CosFaceLoss(ArcFaceLoss):
    def __init__(self,embedding_dim,num_classes=2,margin=.35,scale=32): super().__init__(embedding_dim,num_classes,margin,scale)
    def forward(self,logits,targets):
        features,labels=_embedding(logits,targets)
        with torch.amp.autocast(features.device.type,enabled=False):
            cosine=F.linear(F.normalize(features,dim=1),F.normalize(self.weight.float(),dim=1))
            margin=F.one_hot(labels,num_classes=self.weight.shape[0])*self.margin
            return F.cross_entropy((cosine-margin)*self.scale,labels)


class CenterLoss(nn.Module):
    kind='embedding'
    def __init__(self,embedding_dim,num_classes=2):
        super().__init__();self.centers=nn.Parameter(torch.zeros(num_classes,embedding_dim))
    def forward(self,logits,targets):
        features,labels=_embedding(logits,targets)
        return .5*(features-self.centers[labels]).square().sum(1).mean()


class SupConLoss(nn.Module):
    kind='embedding'
    def __init__(self,temperature=.1):
        super().__init__()
        if temperature<=0: raise ValueError('SupCon temperature must be positive')
        self.temperature=temperature
    def forward(self,logits,targets):
        features,labels=_embedding(logits,targets)
        if len(features)<2: return features.sum()*0
        features=F.normalize(features,dim=1)
        similarity=features@features.T/self.temperature
        diagonal=torch.eye(len(features),dtype=torch.bool,device=features.device)
        denominator=torch.logsumexp(similarity.masked_fill(diagonal,float('-inf')),dim=1)
        positive=(labels[:,None]==labels[None,:])&~diagonal
        counts=positive.sum(1);valid=counts>0
        terms=-((similarity-denominator[:,None])*positive).sum(1)/counts.clamp_min(1)
        return terms[valid].mean() if valid.any() else features.sum()*0


class KLDivLoss(nn.Module):
    def forward(self,logits,targets):
        if logits.shape[-1]==1:
            probability=_targets(logits,targets).float()
            teacher=torch.cat([1-probability,probability],dim=-1)
            student=torch.cat([F.logsigmoid(-logits.float()),F.logsigmoid(logits.float())],dim=-1)
        else:
            teacher=targets.to(device=logits.device,dtype=torch.float32)
            if teacher.shape!=logits.shape: raise ValueError('KL target probabilities must match class logits')
            teacher=teacher/teacher.sum(-1,keepdim=True).clamp_min(1e-12)
            student=F.log_softmax(logits.float(),dim=-1)
        return F.kl_div(student,teacher,reduction='batchmean')


class FrequencyPenaltyLoss(nn.Module):
    kind='spatial'
    def __init__(self,cutoff=.25): super().__init__();self.cutoff=cutoff
    def forward(self,logits,targets=None):
        if logits.ndim<3: raise ValueError('Frequency penalty requires spatial feature maps, not binary logits')
        inputs=logits.float()
        spectrum=torch.fft.rfft2(inputs,norm='ortho')
        if targets is not None:
            if targets.shape!=inputs.shape: raise ValueError('Frequency reference must match spatial features')
            spectrum=spectrum-torch.fft.rfft2(targets.detach().float(),norm='ortho')
        h,w=inputs.shape[-2:]
        fy=torch.fft.fftfreq(h,device=inputs.device)[:,None];fx=torch.fft.rfftfreq(w,device=inputs.device)[None,:]
        mask=(torch.sqrt(fy.square()+fx.square())>=self.cutoff)
        return spectrum.abs().square()[...,mask].mean()


class ConsistencyLoss(nn.Module):
    kind='paired_logits'
    def forward(self,logits,targets):
        if logits.shape!=targets.shape: raise ValueError('Consistency requires paired logits with matching shape')
        return F.mse_loss(torch.sigmoid(logits.float()),torch.sigmoid(targets.detach().float()))


class AdversarialLoss(nn.BCEWithLogitsLoss):
    """Logistic discriminator objective. Adversarial image attacks are separate."""


class MSELoss(nn.Module):
    def forward(self,logits,targets): return F.mse_loss(torch.sigmoid(logits),_targets(logits,targets))


class HingeLoss(nn.Module):
    def forward(self,logits,targets): return F.relu(1-logits*(2*_targets(logits,targets)-1)).mean()


class CombinedLoss(nn.Module):
    def __init__(self,losses,weights=None):
        super().__init__();self.losses=nn.ModuleList(losses);self.weights=list(weights or [1]*len(losses))
        if not losses or len(losses)!=len(self.weights) or any(w<0 or not math.isfinite(w) for w in self.weights) or not any(self.weights):
            raise ValueError('Combined loss requires matching nonnegative, nonzero weights')
    def forward(self,logits,targets): return sum(weight*loss(logits,targets) for loss,weight in zip(self.losses,self.weights))


LOSS_REGISTRY={
    'bce':lambda c,d:nn.BCEWithLogitsLoss(),
    'label_smoothing_bce':lambda c,d:LabelSmoothingBCE(c.get('label_smoothing',.05)),
    'focal':lambda c,d:FocalLoss(c.get('focal_alpha',.25),c.get('focal_gamma',2)),
    'weighted_bce':lambda c,d:WeightedBCE(c.get('class_weight_fake',1),c.get('class_weight_real',1)),
    'lovasz':lambda c,d:LovaszHingeLoss(),
    'dice':lambda c,d:DiceLoss(),
    'tversky':lambda c,d:TverskyLoss(c.get('tversky_alpha',.3),c.get('tversky_beta',.7)),
    'contrastive':lambda c,d:ContrastiveLoss(c.get('contrastive_margin',1)),
    'triplet':lambda c,d:TripletLoss(c.get('triplet_margin',1)),
    'arcface':lambda c,d:ArcFaceLoss(d,margin=c.get('arcface_margin',.5),scale=c.get('metric_scale',32)),
    'cosface':lambda c,d:CosFaceLoss(d,margin=c.get('cosface_margin',.35),scale=c.get('metric_scale',32)),
    'center':lambda c,d:CenterLoss(d),
    'supcon':lambda c,d:SupConLoss(c.get('supcon_temperature',.1)),
    'kl':lambda c,d:KLDivLoss(),
    'frequency_penalty':lambda c,d:FrequencyPenaltyLoss(c.get('frequency_cutoff',.25)),
    'consistency':lambda c,d:ConsistencyLoss(),
    'adversarial':lambda c,d:AdversarialLoss(),
    'mse':lambda c,d:MSELoss(),
    'hinge':lambda c,d:HingeLoss(),
}


def make_loss(name,config=None,embedding_dim=512):
    config=config or {}
    if name=='combined':
        components=config.get('combined_losses',[{'name':'bce','weight':1},{'name':'dice','weight':.1}])
        if any(item['name']=='combined' for item in components): raise ValueError('Recursive combined objectives are invalid')
        return CombinedLoss([make_loss(item['name'],config,embedding_dim) for item in components],[item.get('weight',1) for item in components])
    if name not in LOSS_REGISTRY: raise ValueError(f'Unknown loss {name}')
    return LOSS_REGISTRY[name](config,embedding_dim)


LOSS_REGISTRY['combined']=lambda c,d:make_loss('combined',c,d)


class TrainingObjective(nn.Module):
    """Classification plus explicit feature/context auxiliaries.

    Mixed soft-target examples are excluded from hard-label metric auxiliaries;
    BCE still trains them. Learned prototypes are separate checkpointed state.
    """
    def __init__(self,config,embedding_dim):
        super().__init__();self.config=dict(config)
        self.primary=make_loss(config.get('loss','bce'),config,embedding_dim)
        self.primary_weight=config.get('auxiliary_loss_weight',.1)
        self.frequency=FrequencyPenaltyLoss(config.get('frequency_cutoff',.25))
        self.consistency=ConsistencyLoss()
        parts=[p for p,w in zip(self.primary.losses,self.primary.weights) if w>0] if isinstance(self.primary,CombinedLoss) else ([self.primary] if self.primary_weight>0 or getattr(self.primary,'kind','binary')=='binary' else [])
        kinds=[getattr(p,'kind','binary') for p in parts]
        self.needs_features='embedding' in kinds
        self.domain_classifier=None
        self.domain_method=config.get('domain_method','none')
        if config.get('domain_loss_weight',0)>0:
            if self.domain_method not in {'dann','mmd','coral'}: raise ValueError('Unknown enabled domain adaptation method')
            count=len(config.get('domain_map',{}))
            if count<2: raise ValueError('Domain adaptation requires at least two training domains')
            if self.domain_method=='dann': self.domain_classifier=nn.Linear(embedding_dim,count)
            self.needs_features=True
        self.needs_spatial='spatial' in kinds or config.get('frequency_loss_weight',0)>0
        self.needs_alternative='paired_logits' in kinds or config.get('consistency_loss_weight',0)>0
        if any(not getattr(p,'supports_soft_labels',True) for p in parts) and any(config.get(key,0)>0 for key in ('mixup_alpha','cutmix_alpha','fft_amplitude_mix_prob')):
            raise ValueError('Selected hard-label objective requires disabling MixUp/CutMix/phase mixing')

    def _component(self,module,logits,targets,context):
        kind=getattr(module,'kind','binary')
        if kind=='embedding':
            valid=((targets==0)|(targets==1)).reshape(-1)
            if not valid.any(): return logits.sum()*0
            features=context.get('features')
            if features is None: raise ValueError('Feature loss requires model embeddings, not scalar logits')
            return module(features[valid],targets[valid].reshape(-1))
        if kind=='spatial':
            features=context.get('spatial_features')
            if features is None: raise ValueError('Frequency penalty requires spatial backbone features')
            return module(features,None)
        if kind=='paired_logits':
            other=context.get('alternative_logits')
            if other is None: raise ValueError('Consistency loss requires a second label-preserving view')
            return module(logits,other)
        return module(logits,targets)

    def forward(self,logits,targets,context=None):
        context=context or {}
        if isinstance(self.primary,CombinedLoss):
            has_binary=any(w>0 and getattr(p,'kind','binary')=='binary' for p,w in zip(self.primary.losses,self.primary.weights))
            loss=logits.sum()*0 if has_binary else F.binary_cross_entropy_with_logits(logits,targets)
            for component,weight in zip(self.primary.losses,self.primary.weights):
                if weight: loss=loss+weight*self._component(component,logits,targets,context)
        elif getattr(self.primary,'kind','binary')!='binary':
            loss=F.binary_cross_entropy_with_logits(logits,targets)
            if self.primary_weight: loss=loss+self.primary_weight*self._component(self.primary,logits,targets,context)
        else: loss=self.primary(logits,targets)
        if self.config.get('frequency_loss_weight',0)>0 and getattr(self.primary,'kind',None)!='spatial':
            loss=loss+self.config['frequency_loss_weight']*self._component(self.frequency,logits,targets,context)
        if self.config.get('consistency_loss_weight',0)>0 and getattr(self.primary,'kind',None)!='paired_logits':
            loss=loss+self.config['consistency_loss_weight']*self._component(self.consistency,logits,targets,context)
        if 'auxiliary_logits' in context and self.primary_weight:
            loss=loss+self.primary_weight*F.binary_cross_entropy_with_logits(context['auxiliary_logits'],targets)
        if self.config.get('domain_loss_weight',0)>0:
            from deepfake_detection.training.domain_adaptation import grad_reverse,domain_alignment
            domains=context.get('domain_ids')
            if domains is None: raise ValueError('Enabled domain adaptation requires explicit batch domain IDs')
            if self.domain_method=='dann':
                domain_logits=self.domain_classifier(grad_reverse(context['features'],self.config.get('grl_lambda',.1)))
                auxiliary=F.cross_entropy(domain_logits,domains)
            else: auxiliary=domain_alignment(context['features'],domains,self.domain_method,self.config.get('mmd_kernel','rbf'))
            loss=loss+self.config['domain_loss_weight']*auxiliary
        return loss


def objective_for_model(model,config):
    classifier=model.get_classifier() if hasattr(model,'get_classifier') else None
    dimension=getattr(model,'embedding_dim',getattr(classifier,'in_features',getattr(model,'num_features',512)))
    return TrainingObjective(config,dimension)


def objective_forward(model,inputs,targets,objective,domain_ids=None):
    context={};handles=[]
    need=objective.needs_features or objective.needs_spatial
    try:
        if hasattr(model,'forward_with_features'):
            packet=model.forward_with_features(inputs)
            if isinstance(packet,tuple): logits,context=packet
            else: logits=packet['logits'];context.update(packet)
        else:
            if objective.needs_features:
                if not hasattr(model,'get_classifier'): raise ValueError('Feature loss model must expose get_classifier or forward_with_features')
                def capture_features(module,args):
                    features=args[0]
                    context['features']=features.mean(tuple(range(2,features.ndim))) if features.ndim>2 else features
                handles.append(model.get_classifier().register_forward_pre_hook(capture_features))
            if objective.needs_spatial:
                spatial=getattr(model,'conv_head',getattr(model,'conv4',None))
                if spatial is None:
                    layers=[m for m in model.modules() if isinstance(m,nn.Conv2d)]
                    spatial=layers[-1] if layers else None
                if spatial is None: raise ValueError('Frequency penalty model has no spatial feature map')
                handles.append(spatial.register_forward_hook(lambda module,args,output:context.update(spatial_features=output)))
            logits=model(inputs)
    finally:
        for handle in handles: handle.remove()
    if logits.shape!=targets.shape: raise ValueError('Model must return exactly one REAL logit per input')
    if objective.needs_alternative:
        view=inputs.flip(-1) if inputs.ndim==4 else inputs+torch.randn_like(inputs)*.01
        with torch.no_grad(): context['alternative_logits']=model(view)
    if domain_ids is not None: context['domain_ids']=domain_ids
    return logits,objective(logits,targets,context)
