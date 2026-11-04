"""Offline optimizer registry and serializable Lookahead/SAM/ASAM wrappers."""
import importlib
import math
import torch


def _common(config):
    return {'lr':config.get('head_lr',.001),'weight_decay':config.get('weight_decay',.0001)}


def _adaptive(config):
    return {**_common(config),'betas':tuple(config.get('optimizer_betas',[.9,.999])),'eps':config.get('optimizer_eps',1e-8)}


def _optional_optimizer(params,config,package,name):
    try: module=importlib.import_module(package)
    except ImportError as exc:
        raise ImportError(f'{name} is RESOURCE-PENDING: optional {package} is unavailable; install it explicitly if desired. No installation is performed.') from exc
    constructor=getattr(module,name,None)
    if constructor is None: raise ImportError(f'{package} lacks the required {name} optimizer')
    return constructor(params,**_adaptive(config))


OPTIMIZER_REGISTRY={
    'adamw':lambda p,c:torch.optim.AdamW(p,**_adaptive(c)),
    'adam':lambda p,c:torch.optim.Adam(p,**_adaptive(c)),
    'sgd':lambda p,c:torch.optim.SGD(p,**_common(c),momentum=c.get('momentum',.9)),
    'sgd_nesterov':lambda p,c:torch.optim.SGD(p,**_common(c),momentum=c.get('momentum',.9),nesterov=True),
    'rmsprop':lambda p,c:torch.optim.RMSprop(p,**_common(c),momentum=c.get('momentum',.9),eps=c.get('optimizer_eps',1e-8)),
    'adagrad':lambda p,c:torch.optim.Adagrad(p,**_common(c),eps=c.get('optimizer_eps',1e-8)),
    'lamb':lambda p,c:_optional_optimizer(p,c,'torch_optimizer','Lamb'),
    'lion':lambda p,c:_lion(p,c),
    'sophia':lambda p,c:_sophia(p,c),
}


def _lion(params,config):
    try: module=importlib.import_module('lion_pytorch')
    except ImportError as exc: raise ImportError('Lion is RESOURCE-PENDING: lion_pytorch package required; no installation performed') from exc
    return module.Lion(params,**_common(config),betas=tuple(config.get('optimizer_betas',[.9,.99])))


def _sophia(params,config):
    try: importlib.import_module('sophia_optimizer')
    except ImportError as exc: raise ImportError('Sophia is RESOURCE-PENDING: sophia_optimizer package and curvature-update integration required') from exc
    raise NotImplementedError('Sophia requires an explicit Hessian/Fisher-update adapter for the installed sophia_optimizer version; generic first-order stepping would be misleading')


class OptimizerWrapper(torch.optim.Optimizer):
    def __init__(self,base):
        super().__init__(base.param_groups,base.defaults)
        self.base_optimizer=base
        self.param_groups=base.param_groups
        self.defaults=base.defaults
        self.state=base.state

    def parameters(self):
        return [p for group in self.param_groups for p in group['params']]

    def zero_grad(self,set_to_none=True): return self.base_optimizer.zero_grad(set_to_none=set_to_none)


class LookaheadWrapper(OptimizerWrapper):
    def __init__(self,base,k=5,alpha=.5):
        if not isinstance(k,int) or k<1 or not 0<alpha<=1: raise ValueError('Lookahead requires positive k and alpha in (0,1]')
        super().__init__(base);self.k,self.alpha,self.steps=k,alpha,0
        self.slow_weights=[p.detach().clone() for p in self.parameters()]

    @property
    def is_sam(self): return getattr(self.base_optimizer,'is_sam',False)

    def first_step(self,**kwargs): return self.base_optimizer.first_step(**kwargs)
    def restore_weights(self): return self.base_optimizer.restore_weights()

    @torch.no_grad()
    def _synchronize(self):
        self.steps+=1
        if self.steps%self.k==0:
            for p,slow in zip(self.parameters(),self.slow_weights):
                slow.add_(p-slow,alpha=self.alpha);p.copy_(slow)

    def step(self,closure=None,**kwargs):
        result=self.base_optimizer.step(closure=closure,**kwargs)
        self._synchronize();return result

    def state_dict(self):
        return {'base_optimizer':self.base_optimizer.state_dict(),'slow_weights':self.slow_weights,'steps':self.steps,'k':self.k,'alpha':self.alpha}

    def load_state_dict(self,state):
        self.base_optimizer.load_state_dict(state['base_optimizer']);self.param_groups=self.base_optimizer.param_groups;self.state=self.base_optimizer.state
        if len(state['slow_weights'])!=len(self.parameters()): raise ValueError('Lookahead parameter count mismatch')
        self.slow_weights=[s.to(device=p.device,dtype=p.dtype).clone() for s,p in zip(state['slow_weights'],self.parameters())]
        self.steps,self.k,self.alpha=state['steps'],state['k'],state['alpha']


class SAMWrapper(OptimizerWrapper):
    is_sam=True
    def __init__(self,base,rho=.05,adaptive=False,eta=.01):
        if not math.isfinite(rho) or rho<=0 or eta<0: raise ValueError('SAM requires positive finite rho and nonnegative eta')
        super().__init__(base);self.rho,self.adaptive,self.eta=rho,adaptive,eta
        self.perturbations={};self.original_weights={}

    @torch.no_grad()
    def first_step(self,zero_grad=True):
        params=[p for p in self.parameters() if p.grad is not None]
        if not params: raise RuntimeError('SAM requires gradients from its first closure')
        norms=[((p.abs()+self.eta) if self.adaptive else 1)*p.grad for p in params]
        norm=torch.linalg.vector_norm(torch.stack([g.norm().to(params[0].device) for g in norms]))
        if not torch.isfinite(norm): raise FloatingPointError('Nonfinite SAM gradient norm')
        for p in params:
            self.original_weights[p]=p.detach().clone()
            perturb=((p.abs()+self.eta).square() if self.adaptive else 1)*p.grad*(self.rho/(norm+1e-12))
            self.perturbations[p]=perturb;p.add_(perturb)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def restore_weights(self):
        for p,original in self.original_weights.items(): p.copy_(original)
        self.original_weights.clear();self.perturbations.clear()

    def second_step(self,zero_grad=True):
        self.restore_weights();self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def step(self,closure=None,*,_sam_second_pass=False):
        if _sam_second_pass: return self.base_optimizer.step()
        if closure is None: raise RuntimeError('SAM requires a closure with two forward-backward passes')
        with torch.enable_grad(): loss=closure()
        self.first_step(zero_grad=True)
        try:
            with torch.enable_grad(): closure()
        except BaseException:
            self.restore_weights();raise
        self.second_step();return loss

    def state_dict(self):
        if self.perturbations: raise RuntimeError('Cannot checkpoint SAM during an uncommitted ascent step')
        return {'base_optimizer':self.base_optimizer.state_dict(),'rho':self.rho,'adaptive':self.adaptive,'eta':self.eta}

    def load_state_dict(self,state):
        self.restore_weights();self.base_optimizer.load_state_dict(state['base_optimizer'])
        self.param_groups=self.base_optimizer.param_groups;self.state=self.base_optimizer.state
        self.rho,self.adaptive,self.eta=state['rho'],state['adaptive'],state['eta']


class ASAMWrapper(SAMWrapper):
    def __init__(self,base,rho=.05,eta=.01): super().__init__(base,rho=rho,adaptive=True,eta=eta)


def make_optimizer(params,config):
    name='lion' if config.get('use_lion',False) else config.get('optimizer','adamw')
    if name not in OPTIMIZER_REGISTRY: raise ValueError(f'Unknown optimizer {name}')
    optimizer=OPTIMIZER_REGISTRY[name](params,config)
    if config.get('use_sam',False):
        kind=ASAMWrapper if config.get('sam_adaptive',False) else SAMWrapper
        optimizer=kind(optimizer,rho=config.get('sam_rho',.05))
    if config.get('use_lookahead',False):
        optimizer=LookaheadWrapper(optimizer,k=config.get('lookahead_k',5),alpha=config.get('lookahead_alpha',.5))
    return optimizer
