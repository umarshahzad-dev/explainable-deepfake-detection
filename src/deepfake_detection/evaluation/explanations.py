import numpy as np
import torch


def grad_cam(model, tensor, *, target_class, layer=None):
    """Class 1 uses +REAL logit; class 0 uses -REAL logit. No sigmoid gradient."""
    if target_class not in {0, 1} or tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError("Grad-CAM expects one face and a binary class")
    if layer is None:
        layer = getattr(model, "bn2", None)
        if layer is None:
            layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            if not layers:
                raise ValueError("Model has no usable spatial convolution map")
            layer = layers[-1]
    activation = []
    def capture(module, inputs, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            raise ValueError("CAM layer must produce a spatial tensor")
        output.retain_grad()
        activation.append(output)
    handle = layer.register_forward_hook(capture)
    model.eval()
    model.zero_grad(set_to_none=True)
    try:
        with torch.enable_grad(), torch.amp.autocast(tensor.device.type, enabled=False):
            inputs = tensor.detach().float().requires_grad_(True)
            output = model(inputs)
            if output.numel() != 1:
                raise ValueError("Explanation requires a single binary logit")
            target = output.reshape(())[()] * (1 if target_class == 1 else -1)
            target.backward()
            if not activation or activation[-1].grad is None:
                raise ValueError("No gradient reached the selected CAM layer")
            features = activation[-1]
            weights = features.grad.mean(dim=(2, 3), keepdim=True)
            cam = torch.relu((weights * features).sum(dim=1, keepdim=True))
            cam = torch.nn.functional.interpolate(cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            maximum = cam.max()
            if maximum > 0:
                cam = cam / maximum
            values = cam.detach().cpu().numpy().astype(np.float32)
            if not np.isfinite(values).all():
                raise FloatingPointError("Nonfinite Grad-CAM output")
            return values
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
