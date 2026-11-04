"""Explicit CUDA AMP dtype policy; unsupported native BF16 fails early."""
import torch


def amp_settings(amp,device,recipe=None):
    device=torch.device(device); recipe=recipe or {}
    name=recipe.get('amp_dtype','float16')
    if name not in {'float16','fp16','bfloat16','bf16'}: raise ValueError('AMP dtype must be float16 or bfloat16')
    dtype=torch.bfloat16 if name in {'bfloat16','bf16'} else torch.float16
    enabled=bool(amp and device.type=='cuda')
    if amp and dtype==torch.bfloat16:
        if device.type!='cuda' or torch.cuda.get_device_capability(device)[0]<8:
            raise ValueError('BF16 AMP requires native CUDA capability8.0+; GTX1650 must use float16 or disable AMP')
    return enabled,dtype,enabled and dtype==torch.float16
