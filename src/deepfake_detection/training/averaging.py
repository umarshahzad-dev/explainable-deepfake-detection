"""CPU-resident weight averages with exception-safe temporary model swaps."""
from contextlib import contextmanager
import copy
from collections import deque
import torch


def cpu_state(model): return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}


class WeightEMA:
    def __init__(self,model,decay=.999):
        if not 0<=decay<1: raise ValueError('EMA decay must be in [0,1)')
        self.decay=decay; self.shadow=cpu_state(model); self.updates=0
    @torch.no_grad()
    def update(self,model):
        current=model.state_dict()
        if current.keys()!=self.shadow.keys(): raise ValueError('EMA model structure changed')
        for key,value in current.items():
            value=value.detach().cpu()
            if value.is_floating_point(): self.shadow[key].mul_(self.decay).add_(value,alpha=1-self.decay)
            else: self.shadow[key].copy_(value)
        self.updates+=1
    def apply_to(self,model): model.load_state_dict(self.shadow,strict=True)
    @contextmanager
    def average_parameters(self,model):
        original=cpu_state(model)
        try:
            self.apply_to(model); yield model
        finally: model.load_state_dict(original,strict=True)
    def state_dict(self): return {'decay':self.decay,'updates':self.updates,'shadow':copy.deepcopy(self.shadow)}
    def load_state_dict(self,state):
        if state['decay']!=self.decay: raise ValueError('EMA decay differs from resume configuration')
        values=state['shadow']
        if values.keys()!=self.shadow.keys() or any(values[k].shape!=self.shadow[k].shape or values[k].dtype!=self.shadow[k].dtype for k in values): raise ValueError('EMA resume model shape/dtype mismatch')
        if state['updates']<0 or any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in values.values()): raise ValueError('Invalid EMA state')
        self.shadow={k:v.detach().cpu().clone() for k,v in values.items()}; self.updates=state['updates']


class StochasticWeightAveraging(WeightEMA):
    def __init__(self,model,start_epoch=15):
        if start_epoch<1: raise ValueError('SWA start epoch must be positive')
        super().__init__(model,decay=0); self.start_epoch=start_epoch; self.model=model
    @torch.no_grad()
    def update(self,model,epoch=None):
        if epoch is not None and epoch<self.start_epoch: return
        current=model.state_dict()
        if current.keys()!=self.shadow.keys(): raise ValueError('SWA model structure changed')
        for key,value in current.items():
            value=value.detach().cpu()
            if value.is_floating_point(): self.shadow[key].add_((value-self.shadow[key])/(self.updates+1))
            else: self.shadow[key].copy_(value)
        self.updates+=1
    def apply(self,model): self.apply_to(model); self.model=model
    def state_dict(self): return {**super().state_dict(),'start_epoch':self.start_epoch}
    def load_state_dict(self,state):
        if state['start_epoch']!=self.start_epoch: raise ValueError('SWA start epoch differs from resume configuration')
        super().load_state_dict(state)
    @torch.no_grad()
    def bn_update(self,loader,device,*,model=None):
        """Called while averaged weights are applied; updates only saved BN buffers."""
        model=model or self.model
        dataset=getattr(loader,'dataset',None); rows=getattr(dataset,'rows',None)
        if not rows or any(row.get('split')!='train' for row in rows): raise ValueError('SWA BN update requires explicitly train-only dataset rows')
        modules=list(model.modules()); norms=[m for m in modules if isinstance(m,torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats]
        if not norms: return
        flags=[m.training for m in modules]; momenta=[m.momentum for m in norms]
        model.eval()
        for norm in norms: norm.reset_running_stats(); norm.momentum=None; norm.train()
        seen=0
        try:
            for inputs,*_ in loader:
                # BN1d cannot estimate batch variance for a single example.
                if len(inputs)==1 and any(isinstance(m,torch.nn.BatchNorm1d) for m in norms): continue
                model(inputs.to(device)); seen+=len(inputs)
            if not seen: raise ValueError('SWA BN update had no usable train samples')
            updated=model.state_dict()
            for name,norm in model.named_modules():
                if norm in norms:
                    for suffix in ('running_mean','running_var','num_batches_tracked'):
                        key=f'{name}.{suffix}' if name else suffix
                        self.shadow[key]=updated[key].detach().cpu().clone()
        finally:
            for norm,momentum in zip(norms,momenta): norm.momentum=momentum
            for module,flag in zip(modules,flags): module.training=flag


class WeightAveraging(WeightEMA):
    def __init__(self,model,window=5):
        if window<1: raise ValueError('Weight average window must be positive')
        super().__init__(model,decay=0); self.window=window; self.snapshots=deque(maxlen=window); self.last_epoch=0
    @torch.no_grad()
    def update(self,model,epoch):
        if epoch<=self.last_epoch: raise ValueError('Epoch averages require strictly increasing epoch numbers')
        current=cpu_state(model)
        if current.keys()!=self.shadow.keys(): raise ValueError('Window average model structure changed')
        self.snapshots.append(current)
        for key,value in current.items():
            if value.is_floating_point():
                total=torch.zeros_like(value)
                for state in self.snapshots: total.add_(state[key])
                self.shadow[key]=total/len(self.snapshots)
            else: self.shadow[key]=value.clone()
        self.last_epoch=epoch; self.updates+=1
    def state_dict(self): return {**super().state_dict(),'window':self.window,'last_epoch':self.last_epoch,'snapshots':copy.deepcopy(list(self.snapshots))}
    def load_state_dict(self,state):
        if state['window']!=self.window or not 0<=len(state['snapshots'])<=self.window: raise ValueError('Window average resume configuration/history mismatch')
        super().load_state_dict(state)
        for snapshot in state['snapshots']:
            if snapshot.keys()!=self.shadow.keys() or any(snapshot[k].shape!=self.shadow[k].shape for k in snapshot): raise ValueError('Window average history shape mismatch')
        self.snapshots=deque([{k:v.detach().cpu().clone() for k,v in snapshot.items()} for snapshot in state['snapshots']],maxlen=self.window)
        self.last_epoch=state['last_epoch']
