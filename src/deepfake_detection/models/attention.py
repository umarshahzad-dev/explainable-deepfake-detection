"""Spatial and channel attention operating on genuine backbone feature maps."""
import torch
from torch import nn
from torch.nn import functional as F


class AttentionPooling(nn.Module):
    def __init__(self,channels): super().__init__(); self.score=nn.Conv2d(channels,1,1)
    def forward(self,x):
        weights=self.score(x).flatten(2).softmax(-1)
        return (x.flatten(2)*weights).sum(-1)


class LayerAttentionWeights(nn.Module):
    def __init__(self,channels,features):
        super().__init__()
        if len(channels)<2: raise ValueError('Layer attention requires multiple feature levels')
        self.projections=nn.ModuleList([nn.Linear(c,features) for c in channels])
        self.log_weights=nn.Parameter(torch.zeros(len(channels)))
    @property
    def weights(self): return self.log_weights.softmax(0)
    def forward(self,levels):
        if len(levels)!=len(self.projections): raise ValueError('Layer attention level count mismatch')
        pooled=[projection(x.mean((2,3))) for projection,x in zip(self.projections,levels)]
        return (torch.stack(pooled)*self.weights[:,None,None]).sum(0)


class GlobalContext(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__(); hidden=max(2,channels//reduction)
        self.score=nn.Conv2d(channels,1,1)
        self.transform=nn.Sequential(nn.Conv2d(channels,hidden,1),nn.LayerNorm((hidden,1,1)),nn.ReLU(),nn.Conv2d(hidden,channels,1))
        nn.init.zeros_(self.transform[-1].weight); nn.init.zeros_(self.transform[-1].bias)
    def forward(self,x):
        attention=self.score(x).flatten(2).softmax(-1)
        context=(x.flatten(2)*attention).sum(-1)[:,:,None,None]
        return x+self.transform(context)


class CrossAttentionFusion(nn.Module):
    def __init__(self,channels,spectral_features,heads=4):
        super().__init__()
        if channels%heads: raise ValueError('Cross-attention channels must be divisible by heads')
        self.tokens=nn.Linear(spectral_features,4*channels)
        self.attention=nn.MultiheadAttention(channels,heads,batch_first=True)
        self.norm=nn.LayerNorm(channels); self.channels=channels
    def forward(self,spatial,spectral):
        query=F.adaptive_avg_pool2d(spatial,(4,4)).flatten(2).transpose(1,2)
        context=self.tokens(spectral).reshape(len(spatial),4,self.channels)
        attended,_=self.attention(query,context,context,need_weights=False)
        return self.norm(query+attended).mean(1)


class SqueezeExcite(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__()
        if channels<1 or reduction<1: raise ValueError('Positive attention dimensions required')
        self.gate=nn.Sequential(nn.Conv2d(channels,max(1,channels//reduction),1),nn.ReLU(),nn.Conv2d(max(1,channels//reduction),channels,1),nn.Sigmoid())
    def forward(self,x): return x*self.gate(x.mean((2,3),keepdim=True))


class ECA(nn.Module):
    def __init__(self,channels,kernel_size=3):
        super().__init__()
        if channels<1 or kernel_size<1 or kernel_size%2==0: raise ValueError('ECA requires positive channels and an odd kernel')
        self.filter=nn.Conv1d(1,1,kernel_size,padding=kernel_size//2,bias=False)
    def forward(self,x):
        gate=torch.sigmoid(self.filter(x.mean((2,3)).unsqueeze(1))).squeeze(1)
        return x*gate[:,:,None,None]


class CBAM(nn.Module):
    def __init__(self,channels,reduction=16):
        super().__init__()
        if channels<1 or reduction<1: raise ValueError('Positive attention dimensions required')
        hidden=max(1,channels//reduction)
        self.channel=nn.Sequential(nn.Conv2d(channels,hidden,1,bias=False),nn.ReLU(),nn.Conv2d(hidden,channels,1,bias=False))
        self.spatial=nn.Conv2d(2,1,7,padding=3,bias=False)
    def forward(self,x):
        gate=torch.sigmoid(self.channel(x.mean((2,3),keepdim=True))+self.channel(x.amax((2,3),keepdim=True)))
        x=x*gate
        return x*torch.sigmoid(self.spatial(torch.cat([x.mean(1,keepdim=True),x.amax(1,keepdim=True)],1)))
