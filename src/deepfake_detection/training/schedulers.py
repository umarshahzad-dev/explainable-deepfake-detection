"""Serializable update/validation-aware scheduler alternatives with warmup."""
import math
import torch


class NoamSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Normalized Noam curve: each configured base LR is its warmup peak."""
    def __init__(self,optimizer,warmup_steps,min_lr=0):
        warmup=max(1,warmup_steps)
        functions=[]
        for group in optimizer.param_groups:
            floor=min_lr/group['lr']
            functions.append(lambda step,floor=floor:max(floor,min((step+1)/warmup,math.sqrt(warmup/(step+1)))))
        super().__init__(optimizer,functions)


class SGDRSchedule(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self,optimizer,cycle_steps,multiplier=2,min_lr=0):
        super().__init__(optimizer,T_0=cycle_steps,T_mult=multiplier,eta_min=min_lr)


def _lambda_decay(optimizer,config,total,warmup,kind):
    functions=[]
    for group in optimizer.param_groups:
        floor=config.get('min_lr',1e-6)/group['lr']
        power=config.get('scheduler_power',2)
        def factor(step,floor=floor,power=power):
            if warmup and step<warmup: return .1+.9*step/warmup
            progress=min(1,max(0,(step-warmup)/max(1,total-warmup)))
            value=(1+math.cos(math.pi*progress))/2 if kind=='cosine' else (1-progress)**power if kind=='polynomial' else 1-progress
            return floor+(1-floor)*value
        functions.append(factor)
    return torch.optim.lr_scheduler.LambdaLR(optimizer,functions)


SCHEDULER_REGISTRY={
    'cosine':lambda o,c,s,t,w:_lambda_decay(o,c,t,w,'cosine'),
    'linear':lambda o,c,s,t,w:_lambda_decay(o,c,t,w,'linear'),
    'polynomial':lambda o,c,s,t,w:_lambda_decay(o,c,t,w,'polynomial'),
    'cosine_warm_restarts':lambda o,c,s,t,w:torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(o,T_0=c.get('scheduler_cycle_length',10)*s,T_mult=c.get('scheduler_restart_mult',2),eta_min=c.get('min_lr',1e-6)),
    'warm_restarts':lambda o,c,s,t,w:SGDRSchedule(o,c.get('scheduler_cycle_length',10)*s,c.get('scheduler_restart_mult',2),c.get('min_lr',1e-6)),
    'onecycle':lambda o,c,s,t,w:torch.optim.lr_scheduler.OneCycleLR(o,max_lr=[g['lr'] for g in o.param_groups],total_steps=t,pct_start=min(.95,max(.05,w/t if w else .3)),div_factor=10,final_div_factor=1000),
    'step':lambda o,c,s,t,w:torch.optim.lr_scheduler.StepLR(o,step_size=c.get('scheduler_cycle_length',10)*s,gamma=c.get('scheduler_step_gamma',.1)),
    'reduce_on_plateau':lambda o,c,s,t,w:torch.optim.lr_scheduler.ReduceLROnPlateau(o,mode='max',factor=c.get('scheduler_factor',.5),patience=c.get('scheduler_patience',2),min_lr=c.get('min_lr',1e-6)),
    'noam':lambda o,c,s,t,w:NoamSchedule(o,w or s,c.get('min_lr',1e-6)),
}


class SchedulerAdapter:
    def __init__(self,optimizer,raw,kind,min_lr):
        self.optimizer,self.raw,self.kind,self.min_lr=optimizer,raw,kind,min_lr
        self.updates=0;self.metrics=0
        self.unscaled_lrs=None
    def _before_step(self):
        if self.unscaled_lrs is not None:
            for group,lr in zip(self.optimizer.param_groups,self.unscaled_lrs): group['lr']=lr
    def _after_step(self):
        if self.unscaled_lrs is not None:
            self.unscaled_lrs=[group['lr'] for group in self.optimizer.param_groups]
            for group,lr in zip(self.optimizer.param_groups,self.unscaled_lrs): group['lr']=lr*group.get('layer_warmup_factor',1)
    def set_layer_warmup(self,epoch,epochs):
        if epochs<0 or epoch<0: raise ValueError('Layer warmup epoch counts cannot be negative')
        if not epochs: return
        if self.unscaled_lrs is None: self.unscaled_lrs=[group['lr'] for group in self.optimizer.param_groups]
        for group,lr in zip(self.optimizer.param_groups,self.unscaled_lrs):
            factor=1
            if 'layer_depth' in group and any(p.requires_grad for p in group['params']):
                group.setdefault('unfrozen_at_epoch',epoch)
                factor=min(1,(epoch-group['unfrozen_at_epoch']+1)/epochs)
            group['layer_warmup_factor']=factor; group['lr']=lr*factor
    def step(self):
        if self.kind!='reduce_on_plateau':
            self._before_step()
            self.raw.step();self.updates+=1
            for group in self.optimizer.param_groups: group['lr']=max(self.min_lr,group['lr'])
            self._after_step()
    def step_metric(self,metric):
        if self.kind=='reduce_on_plateau':
            if not math.isfinite(metric): raise ValueError('Plateau scheduler metric must be finite')
            self._before_step()
            self.raw.step(metric);self.metrics+=1
            self._after_step()
    def state_dict(self):
        return {'kind':self.kind,'raw':self.raw.state_dict(),'updates':self.updates,'metrics':self.metrics,'min_lr':self.min_lr,'unscaled_lrs':self.unscaled_lrs}
    def load_state_dict(self,state):
        if 'kind' not in state:
            if self.kind!='cosine': raise ValueError('Only legacy cosine schedule states can migrate without a kind')
            self.raw.load_state_dict(state);self.updates=state.get('last_epoch',0);return
        if state['kind']!=self.kind: raise ValueError('Scheduler kind mismatch on resume')
        self.raw.load_state_dict(state['raw']);self.updates=state['updates'];self.metrics=state['metrics'];self.min_lr=state['min_lr']
        self.unscaled_lrs=state.get('unscaled_lrs')
    def get_last_lr(self): return [group['lr'] for group in self.optimizer.param_groups]


def make_scheduler(optimizer,config,steps_per_epoch):
    kind=config.get('scheduler','cosine')
    if config.get('scheduler_warm_restarts',False):
        if kind not in {'cosine','cosine_warm_restarts','warm_restarts'}: raise ValueError('Warm-restart flag conflicts with selected scheduler')
        kind='cosine_warm_restarts'
    if kind not in SCHEDULER_REGISTRY: raise ValueError(f'Unknown scheduler {kind}')
    total=steps_per_epoch*config['finetune_epochs'];warmup=steps_per_epoch*config.get('warmup_epochs',1)
    if steps_per_epoch<1 or total<1: raise ValueError('Schedule needs positive update and epoch counts')
    raw=SCHEDULER_REGISTRY[kind](optimizer,config,steps_per_epoch,total,warmup)
    # The cosine/linear/polynomial and Noam factories handle warmup intrinsically.
    # Restart/step schedules use an explicit prefix, while plateau has epoch metrics.
    if warmup>0 and kind in {'cosine_warm_restarts','warm_restarts','step'}:
        prefix=torch.optim.lr_scheduler.LinearLR(optimizer,start_factor=.1,total_iters=warmup)
        raw=torch.optim.lr_scheduler.SequentialLR(optimizer,[prefix,raw],milestones=[warmup])
    return SchedulerAdapter(optimizer,raw,kind,config.get('min_lr',1e-6))
