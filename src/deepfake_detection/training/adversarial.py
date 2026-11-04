"""Pixel-bounded attacks for normalized RGB inputs; no optimizer side effects."""
import math
import torch
from torch.nn import functional as F


def pgd_attack(model,images,labels,eps=.01,alpha=.002,steps=3,*,mean=None,std=None,random_start=True):
    if images.ndim!=4 or images.shape[1]!=3 or not images.is_floating_point(): raise ValueError('Attack requires floating NCHW RGB images')
    if not math.isfinite(eps) or not math.isfinite(alpha) or eps<0 or alpha<0 or steps<1: raise ValueError('Invalid attack budget')
    means=torch.tensor(mean if mean is not None else [0]*3,device=images.device,dtype=torch.float32)[None,:,None,None]
    deviations=torch.tensor(std if std is not None else [1]*3,device=images.device,dtype=torch.float32)[None,:,None,None]
    if not torch.isfinite(deviations).all() or (deviations<=0).any(): raise ValueError('Attack normalization std must be positive')
    lower=-means/deviations; upper=(1-means)/deviations
    original=images.detach().float().clamp(lower,upper)
    if not eps: return original.to(images.dtype)
    radius=eps/deviations; step=alpha/deviations
    value=original+torch.empty_like(original).uniform_(-1,1)*radius if random_start else original.clone()
    flags=[(module,module.training) for module in model.modules()]; model.eval()
    try:
        with torch.enable_grad(),torch.amp.autocast(images.device.type,enabled=False):
            for _ in range(steps):
                value=value.detach().clamp(lower,upper).requires_grad_(True)
                logits=model(value)
                targets=labels.to(device=logits.device,dtype=logits.dtype).reshape_as(logits)
                loss=F.binary_cross_entropy_with_logits(logits,targets)
                gradient=torch.autograd.grad(loss,value,only_inputs=True)[0]
                if not torch.isfinite(gradient).all(): raise FloatingPointError('Nonfinite adversarial input gradient')
                value=value.detach()+step*gradient.sign()
                value=torch.maximum(torch.minimum(value,original+radius),original-radius).clamp(lower,upper)
        return value.detach().to(images.dtype)
    finally:
        for module,flag in flags: module.training=flag


def fgsm_attack(model,images,labels,eps=.01,*,mean=None,std=None):
    return pgd_attack(model,images,labels,eps,eps,1,mean=mean,std=std,random_start=False)
