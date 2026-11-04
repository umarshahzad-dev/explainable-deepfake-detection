"""Opt-in heads; initialization loads the base before these fresh modules exist."""
from torch import nn
import torch
from torch.nn import functional as F
from deepfake_detection.models.attention import CBAM,SqueezeExcite,ECA,CrossAttentionFusion,AttentionPooling,LayerAttentionWeights,GlobalContext
from deepfake_detection.models.frequency import GaborBank,FFTBranch,DCTBranch,NoiseResidualBranch,WaveletBranch

ALTERNATIVE_MODELS={'vit':'vit_tiny_patch16_224','swin':'swin_tiny_patch4_window7_224','convnext':'convnext_tiny','hybrid_cnn_vit':'vit_tiny_r_s16_p8_224'}
SPATIAL_ENHANCEMENTS=('use_cbam','use_se','use_eca','use_gabor','use_frequency','use_dct','use_prnu','use_wavelet','use_dual_stream','use_cross_attention','use_fpn','use_auxiliary_classifier','use_multi_head','use_attention_pooling','use_layernorm','head_hidden','classifier_dropout','use_se_before_head','use_layer_attention','use_gcnet','use_weight_standardization','use_multi_branch','activation')


def alternative_name(name):
    return ALTERNATIVE_MODELS.get(name,name if name in ALTERNATIVE_MODELS.values() else None)


def create_alternative_model(cfg):
    import timm
    name=alternative_name(cfg['name'])
    if name is None or not timm.is_model(name): raise ValueError(f'Installed timm does not provide the registered alternative: {cfg["name"]}')
    if any(cfg.get(key) for key in SPATIAL_ENHANCEMENTS): raise ValueError('Alternative backbone requires disabling Xception spatial/head enhancements; use its native head')
    if cfg.get('input_size',224)!=224: raise ValueError('Registered alternative currently requires224input and fixed-size TTA')
    return timm.create_model(name,pretrained=False,num_classes=1,drop_rate=cfg.get('dropout',.2),drop_path_rate=cfg.get('drop_path',0))


class WeightStandardization(nn.Module):
    def forward(self,weight):
        values=weight.float(); mean=values.mean((1,2,3),keepdim=True)
        variance=values.var((1,2,3),unbiased=False,keepdim=True)
        return ((values-mean)*torch.rsqrt(variance+1e-5)).to(weight.dtype)


def standardize_weights(base):
    from torch.nn.utils import parametrize
    for module in list(base.modules()):
        if isinstance(module,nn.Conv2d) and not parametrize.is_parametrized(module,'weight'):
            parametrize.register_parametrization(module,'weight',WeightStandardization())


def replace_activations(base,name):
    if name not in ('relu','gelu','silu'): raise ValueError('Backbone activation must be relu, gelu or silu')
    kinds={'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}
    for parent in base.modules():
        for key,child in list(parent.named_children()):
            if isinstance(child,(nn.ReLU,nn.GELU,nn.SiLU)): setattr(parent,key,kinds[name]())


class DropPath(nn.Module):
    def __init__(self,probability=0):
        super().__init__()
        if not 0<=probability<1: raise ValueError('DropPath probability must be in [0,1)')
        self.probability=probability
    def forward(self,x):
        if not self.training or not self.probability: return x
        keep=1-self.probability
        mask=torch.empty((len(x),)+(1,)*(x.ndim-1),device=x.device,dtype=x.dtype).bernoulli_(keep)
        return x*mask/keep


def inject_drop_path(base,probability,*,schedule='constant'):
    if schedule not in ('constant','linear'): raise ValueError('Stochastic depth schedule must be constant or linear')
    if not 0<=probability<1: raise ValueError('DropPath probability must be in [0,1)')
    if not probability: return 0
    blocks=[m for m in base.modules() if hasattr(m,'rep') and isinstance(m.rep,nn.Module)]
    if not blocks: raise ValueError('DropPath adapter requires residual blocks exposing rep')
    for i,block in enumerate(blocks):
        rate=probability*(i+1)/len(blocks) if schedule=='linear' else probability
        if not any(isinstance(m,DropPath) for m in block.rep.modules()): block.rep=nn.Sequential(block.rep,DropPath(rate))
    return len(blocks)


class DualStreamFusion(nn.Module):
    def __init__(self,spatial_features,spectral_features,output_features):
        super().__init__(); self.projection=nn.Sequential(nn.Linear(spatial_features+spectral_features,output_features),nn.GELU())
    def forward(self,spatial,spectral): return self.projection(torch.cat([spatial,spectral],1))


class MultiHeadClassifier(nn.Module):
    """Independent REAL logits averaged for the public binary classifier."""
    def __init__(self,features,heads=3):
        super().__init__()
        if heads<1: raise ValueError('At least one classifier head required')
        self.heads=nn.ModuleList([nn.Linear(features,1) for _ in range(heads)])
        self.in_features,self.out_features=features,1
    def forward(self,x): return torch.stack([head(x) for head in self.heads]).mean(0)


class FPNHead(nn.Module):
    def __init__(self,channels,output_features,hidden=128):
        super().__init__()
        if len(channels)<2: raise ValueError('FPN requires at least two actual feature levels')
        self.lateral=nn.ModuleList([nn.Conv2d(c,hidden,1) for c in channels])
        self.smooth=nn.ModuleList([nn.Conv2d(hidden,hidden,3,padding=1) for _ in channels])
        self.projection=nn.Linear(hidden*len(channels),output_features)
    def forward(self,levels):
        if len(levels)!=len(self.lateral): raise ValueError('FPN level count mismatch')
        outputs=[conv(x) for conv,x in zip(self.lateral,levels)]
        for i in range(len(outputs)-2,-1,-1): outputs[i]=outputs[i]+F.interpolate(outputs[i+1],size=outputs[i].shape[-2:],mode='nearest')
        return self.projection(torch.cat([F.gelu(conv(x)).mean((2,3)) for conv,x in zip(self.smooth,outputs)],1))


def feature_taps(base,cfg):
    modules=dict(base.named_modules())
    names=cfg.get('fpn_layers')
    if names is None:
        names=[n for n in ('block4','block8','block12') if n in modules]
        if not names: names=[n for n in ('blocks.1','blocks.3','blocks.5') if n in modules]
    if not names: raise ValueError('No known feature taps; specify model.fpn_layers')
    channels=[]
    for name in names:
        if name not in modules: raise ValueError(f'Unknown feature tap: {name}')
        candidates=[m for m in modules[name].modules() if isinstance(m,(nn.Conv2d,nn.modules.batchnorm._BatchNorm))]
        if not candidates: raise ValueError(f'Cannot infer channels for feature tap: {name}')
        last=candidates[-1]; channels.append(last.out_channels if isinstance(last,nn.Conv2d) else last.num_features)
    return names,channels


class EnhancedClassifier(nn.Module):
    def __init__(self,base,cfg):
        super().__init__(); self.base=base
        channels=base.num_features
        self.streams=nn.ModuleDict()
        if cfg.get('use_gabor'): self.streams['gabor']=GaborBank(cfg.get('frequency_features',128))
        if cfg.get('use_frequency'): self.streams['fft']=FFTBranch(cfg.get('frequency_features',128))
        if cfg.get('use_dct'): self.streams['dct']=DCTBranch(cfg.get('frequency_features',128))
        if cfg.get('use_prnu'): self.streams['residual']=NoiseResidualBranch(cfg.get('frequency_features',128))
        if cfg.get('use_wavelet'): self.streams['wavelet']=WaveletBranch(cfg.get('frequency_features',128))
        if (cfg.get('use_dual_stream') or cfg.get('use_cross_attention')) and not self.streams: self.streams['residual']=NoiseResidualBranch(cfg.get('frequency_features',128))
        stream_dim=sum(stream.features for stream in self.streams.values())
        self.cross_attention=CrossAttentionFusion(channels,stream_dim,cfg.get('cross_attention_heads',4)) if cfg.get('use_cross_attention') else None
        self.fusion=DualStreamFusion(channels,stream_dim,channels) if cfg.get('use_dual_stream') else None
        self.embedding_dim=channels if self.fusion is not None else channels+stream_dim
        self.tap_names,self.tap_channels=feature_taps(base,cfg) if cfg.get('use_fpn') or cfg.get('use_auxiliary_classifier') or cfg.get('use_layer_attention') else ([],[])
        self.fpn=FPNHead(self.tap_channels+[channels],channels,cfg.get('fpn_features',128)) if cfg.get('use_fpn') else None
        self.layer_attention=LayerAttentionWeights(self.tap_channels+[channels],channels) if cfg.get('use_layer_attention') else None
        self.auxiliary_classifier=nn.Linear(self.tap_channels[len(self.tap_channels)//2],1) if cfg.get('use_auxiliary_classifier') else None
        self.cbam=CBAM(channels,cfg.get('cbam_reduction',16)) if cfg.get('use_cbam') else nn.Identity()
        self.se=SqueezeExcite(channels,cfg.get('se_reduction',16)) if cfg.get('use_se') else nn.Identity()
        self.eca=ECA(channels,cfg.get('eca_kernel',3)) if cfg.get('use_eca') else nn.Identity()
        self.global_context=GlobalContext(channels,cfg.get('gcnet_reduction',16)) if cfg.get('use_gcnet') else nn.Identity()
        self.pool=AttentionPooling(channels) if cfg.get('use_attention_pooling') else None
        self.feature_norm=nn.LayerNorm(self.embedding_dim) if cfg.get('use_layernorm') else nn.Identity()
        hidden=cfg.get('head_hidden',0)
        if hidden<0: raise ValueError('head_hidden cannot be negative')
        activation=cfg.get('classifier_activation','relu')
        if activation not in ('relu','gelu','silu'): raise ValueError('Unsupported classifier activation')
        self.feature_projection=nn.Sequential(nn.Linear(self.embedding_dim,hidden),{'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}[activation]()) if hidden else nn.Identity()
        if hidden: self.embedding_dim=hidden
        self.head_se=SqueezeExcite(self.embedding_dim,cfg.get('se_reduction',16)) if cfg.get('use_se_before_head') else None
        self.classifier_dropout=nn.Dropout(cfg.get('classifier_dropout',cfg.get('dropout',0)))
        self.classifier=MultiHeadClassifier(self.embedding_dim,cfg.get('classification_heads',3)) if cfg.get('use_multi_head') else nn.Linear(self.embedding_dim,1)
    def get_classifier(self): return self.classifier
    def head_parameters(self):
        return [p for name,child in self.named_children() if name!='base' for p in child.parameters()]
    def forward_with_features(self,x):
        captured={}; handles=[]; modules=dict(self.base.named_modules())
        try:
            for name in self.tap_names:
                handles.append(modules[name].register_forward_hook(lambda m,a,y,key=name: captured.__setitem__(key,y)))
            spatial=self.global_context(self.eca(self.se(self.cbam(self.base.forward_features(x)))))
        finally:
            for handle in handles: handle.remove()
        features=self.pool(spatial) if self.pool is not None else spatial.mean((2,3))
        if self.fpn is not None: features=features+self.fpn([captured[name] for name in self.tap_names]+[spatial])
        if self.layer_attention is not None: features=features+self.layer_attention([captured[name] for name in self.tap_names]+[spatial])
        if self.streams:
            spectral=torch.cat([stream(x) for stream in self.streams.values()],1)
            if self.cross_attention is not None: features=features+self.cross_attention(spatial,spectral)
            features=self.fusion(features,spectral) if self.fusion is not None else torch.cat([features,spectral],1)
        features=self.feature_norm(features)
        features=self.feature_projection(features)
        if self.head_se is not None: features=self.head_se(features[:,:,None,None]).flatten(1)
        context={'features':features,'spatial_features':spatial}
        if self.auxiliary_classifier is not None:
            midpoint=captured[self.tap_names[len(self.tap_names)//2]].mean((2,3))
            context['auxiliary_logits']=self.auxiliary_classifier(midpoint)
        return self.classifier(self.classifier_dropout(features)),context
    def forward(self,x): return self.forward_with_features(x)[0]


def enhance(base,cfg):
    if cfg.get('use_multi_branch'):
        raise NotImplementedError('Multi-branch Xception requires an approved multi-backbone design and a new VRAM profile; compact dual-stream fusion is implemented separately')
    if cfg.get('drop_path',0)>0: inject_drop_path(base,cfg['drop_path'],schedule=cfg.get('stochastic_depth_schedule','constant'))
    if cfg.get('use_weight_standardization'): standardize_weights(base)
    if cfg.get('activation'): replace_activations(base,cfg['activation'])
    return EnhancedClassifier(base,cfg) if any(cfg.get(k) for k in ('use_cbam','use_se','use_eca','use_gabor','use_frequency','use_dct','use_prnu','use_wavelet','use_dual_stream','use_cross_attention','use_fpn','use_auxiliary_classifier','use_multi_head','use_attention_pooling','use_layernorm','head_hidden','classifier_dropout','use_se_before_head','use_layer_attention','use_gcnet')) else base
