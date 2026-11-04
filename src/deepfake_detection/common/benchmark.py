"""Bounded crop/worker/augmentation/forward/backward benchmark; never steps or exports."""
import copy
import time
import torch
from torch.utils.data import DataLoader,Dataset
from deepfake_detection.data.datasets import CropDataset,crop_rows
from deepfake_detection.data.manifests import stratified_limit
from deepfake_detection.models.model_loader import initialize_model
from deepfake_detection.common.paths import Workspace,write_json
from deepfake_detection.common.reproducibility import seed_all,seed_worker
from deepfake_detection.training.trainer import freeze_bn
from deepfake_detection.common.telemetry import MemoryMonitor
from deepfake_detection.common.progress import Progress


class TimedDataset(Dataset):
    def __init__(self,dataset,clock): self.dataset,self.clock=dataset,clock
    def __len__(self): return len(self.dataset)
    def __getitem__(self,index): return self.dataset.timed_item(index,self.clock)


def benchmark(cfg,manifests,workspace=None,*,crops=200,device=None,batch_size=None,clock=None):
    if not 1<=crops<=2000: raise ValueError("Benchmark crops must be 1..2000")
    ws=workspace or Workspace()
    cfg=copy.deepcopy(cfg)
    if batch_size is not None:
        if batch_size<1: raise ValueError("Benchmark batch size must be positive")
        cfg["training"]["batch_size"]=batch_size
    clock=clock or time.perf_counter
    device=torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_all(cfg["seed"])
    rows=stratified_limit(crop_rows(manifests,ws),crops,cfg["seed"])
    if not rows: raise ValueError("Benchmark requires nonempty completed crops")
    dataset=TimedDataset(CropDataset(rows,cfg,training=True,workspace=ws),clock)
    recipe=cfg["training"]
    loader=DataLoader(dataset,batch_size=recipe["batch_size"],shuffle=False,num_workers=recipe["workers"],
                      worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(cfg["seed"]),
                      pin_memory=device.type=="cuda" and recipe["pin_memory"])
    model=initialize_model(cfg,device=device).train()
    if recipe["freeze_bn"]: freeze_bn(model)
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    load_seconds=augment_seconds=transform_seconds=compute_seconds=wait_seconds=0
    from deepfake_detection.training.precision import amp_settings
    amp_enabled,amp_dtype,_=amp_settings(recipe["amp"],device,recipe)
    samples=0
    progress=Progress("Benchmarked",len(rows),unit="crops")
    started=clock()
    with MemoryMonitor(device) as memory:
        iterator=iter(loader)
        while True:
            wait=clock()
            try: inputs,labels,timings=next(iterator)
            except StopIteration: break
            wait_seconds+=clock()-wait
            load_seconds+=float(timings["load"].sum())
            augment_seconds+=float(timings["augment"].sum())
            transform_seconds+=float(timings["transform"].sum())
            if device.type=="cuda": torch.cuda.synchronize(device)
            begin=clock()
            model.zero_grad(set_to_none=True)
            inputs=inputs.to(device,non_blocking=device.type=="cuda")
            labels=labels.to(device).float().reshape(-1,1)
            with torch.amp.autocast(device.type,enabled=amp_enabled,dtype=amp_dtype):
                logits=model(inputs)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)
            if not bool(torch.isfinite(loss)): raise FloatingPointError("Nonfinite benchmark loss")
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite benchmark gradients")
            if device.type=="cuda": torch.cuda.synchronize(device)
            compute_seconds+=clock()-begin
            samples+=len(labels)
            progress.update(samples)
    elapsed=clock()-started
    progress.finish(samples)
    measurements=memory.report()
    result={"samples":samples,"selected_sample_ids":[r["sample_id"] for r in rows],"seed":cfg["seed"],"optimizer_updates":0,
            "images_per_second":samples/elapsed,"dataset_load_seconds":load_seconds,"augmentation_seconds":augment_seconds,
            "tensor_transform_seconds":transform_seconds,"forward_backward_seconds":compute_seconds,"loader_wait_seconds":wait_seconds,
            "elapsed_seconds":elapsed,"peak_allocated_mib":measurements["peak_allocated_mib"],"peak_reserved_mib":measurements["peak_reserved_mib"],
            "vram_sampled_mib":measurements["peak_process_mib"],"memory":measurements,"device":str(device),
            "batch_size":recipe["batch_size"],"workers":recipe["workers"],"pilot":True,"report_path":str(ws.report("benchmark")),
            "note":"No optimizer step or weights export. Load/augmentation times sum worker CPU time; wall throughput includes worker startup/wait. Host thermal/load variation can exceed20%."}
    write_json(ws.report("benchmark")/"benchmark.json",result,ws)
    return result
