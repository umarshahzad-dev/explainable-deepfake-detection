"""Shared deterministic RGB -> square299 PNG -> bicubic224 tensor contract."""
from dataclasses import dataclass
import math
import numpy as np
from PIL import Image, ImageOps
from deepfake_detection.common.configuration import DEFAULTS
from deepfake_detection.common.paths import fingerprint


@dataclass(frozen=True)
class Geometry:
    x: int
    y: int
    side: int
    source_width: int
    source_height: int
    rotation_degrees: float = 0.0

    def as_dict(self):
        result=dict(vars(self))
        if not self.rotation_degrees: result.pop('rotation_degrees')
        return result


@dataclass
class Crop:
    image: Image.Image
    geometry: Geometry
    aligned_landmarks: object = None


def to_rgb(image):
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba)
    return image.convert("RGB")


def validate_image(image, max_pixels=8000000):
    if getattr(image, "is_animated", False):
        raise ValueError("Only still images are supported")
    if image.width * image.height > max_pixels:
        raise ValueError("Image exceeds the decoded pixel limit")
    if min(image.size) < 50:
        raise ValueError("Image is too small for the configured face detector")
    return to_rgb(image)


def square_crop(image, box, *, margin=1.3, size=299,anchor='box_center'):
    image = to_rgb(image)
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        raise ValueError("Bounding box must contain four finite coordinates")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1 or margin <= 0 or size < 1:
        raise ValueError("Invalid crop dimensions")
    side = math.ceil(max(x2 - x1, y2 - y1) * margin)
    if side > max(image.size) * 4:
        raise ValueError("Bounding box would allocate an unreasonable padded crop")
    if anchor not in {'center','box_center'}:
        raise ValueError('Crop anchor must be center or box_center')
    cx,cy=(image.width/2,image.height/2) if anchor=='center' else ((x1+x2)/2,(y1+y2)/2)
    x = math.floor(cx - side / 2)
    y = math.floor(cy - side / 2)
    native = image.crop((x, y, x + side, y + side))  # PIL supplies black outside bounds.
    master = native.resize((size, size), Image.Resampling.BICUBIC)
    return Crop(master, Geometry(x, y, side, image.width, image.height))


def model_tensor(master, model_config=None):
    import torch
    cfg = model_config or DEFAULTS["model"]
    size = cfg["input_size"]
    resized = to_rgb(master).resize((size, size), Image.Resampling.BICUBIC)
    array = np.array(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
    mean = torch.tensor(cfg["mean"], dtype=torch.float32)[:, None, None]
    std = torch.tensor(cfg["std"], dtype=torch.float32)[:, None, None]
    return (tensor - mean) / std


def canonical_pose_landmarks(output_size=299):
    """Internal normalized eye/nose/mouth reference; not a verified identity template."""
    if not isinstance(output_size,int) or output_size<1:
        raise ValueError('Canonical landmark output size must be a positive integer')
    return np.array([[.35,.35],[.65,.35],[.5,.52],[.38,.68],[.62,.68]],dtype=float)*output_size


def align_by_landmarks(image,landmarks,output_size=299,margin=1.5,*,anchor='box_center'):
    """Rotate five landmarks to horizontal eyes, then square-crop their bounding box.

    Returns Crop with inverse-projectable rotation geometry and output landmarks.
    No identity-specific warp or inference-only transform is introduced.
    """
    points=np.asarray(landmarks,dtype=float)
    if points.shape!=(5,2) or not np.isfinite(points).all():
        raise ValueError('Alignment requires five finite eye/nose/mouth landmarks')
    eye=points[1]-points[0]
    if np.linalg.norm(eye)<1e-6:
        raise ValueError('Alignment eye landmarks must be distinct')
    image=to_rgb(image)
    angle=math.degrees(math.atan2(eye[1],eye[0]))
    radians=math.radians(angle)
    center=np.array([image.width/2,image.height/2])
    matrix=np.array([[math.cos(radians),math.sin(radians)],[-math.sin(radians),math.cos(radians)]])
    rotated_points=(points-center)@matrix.T+center
    rotated=image.rotate(angle,resample=Image.Resampling.BICUBIC,center=tuple(center),fillcolor=(0,0,0))
    low=rotated_points.min(axis=0);high=rotated_points.max(axis=0)
    crop=square_crop(rotated,[*low,*high],margin=margin,size=output_size,anchor=anchor)
    geometry=crop.geometry
    crop.geometry=Geometry(geometry.x,geometry.y,geometry.side,image.width,image.height,angle)
    crop.aligned_landmarks=((rotated_points-np.array([geometry.x,geometry.y]))*output_size/geometry.side).tolist()
    return crop


def face_crop_variants(image,box,landmarks,config):
    margins=config.get('multi_margin',[config['margin_factor']])
    if not isinstance(margins,(list,tuple)) or not margins:
        raise ValueError('multi_margin must be a nonempty list')
    margins=[float(value) for value in margins]
    if any(not math.isfinite(value) or value<=0 for value in margins) or len({f'{v:g}' for v in margins})!=len(margins):
        raise ValueError('Crop margins must be finite, positive and uniquely named')
    variants=[]
    for margin in margins:
        alignment=config.get('alignment','none')
        if alignment=='landmark':
            crop=align_by_landmarks(image,landmarks,output_size=config['master_size'],margin=margin,anchor=config.get('crop_anchor','box_center'))
        elif alignment in {'none','box'}:
            crop=square_crop(image,box,margin=margin,size=config['master_size'],anchor=config.get('crop_anchor','box_center'))
        else: raise ValueError('Alignment must be none, box or landmark')
        variants.append((margin,crop))
    return variants


def preprocessing_hash(config):
    return fingerprint({"preprocessing": config["preprocessing"]})


def overlay_heatmap(image, geometry, heatmap, alpha=0.45):
    """Project a crop CAM through padding back onto original image coordinates."""
    image = to_rgb(image)
    values = np.asarray(heatmap, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Heatmap must be a finite 2-D array")
    values = np.clip(values, 0, 1)
    heat = Image.fromarray(np.uint8(values * 255)).resize((geometry.side, geometry.side), Image.Resampling.BILINEAR)
    if geometry.rotation_degrees:
        # CAM is in the rotated crop frame; undo roll before original-image overlay.
        canvas=Image.new('L',image.size,0)
        canvas.paste(heat,(geometry.x,geometry.y))
        canvas=canvas.rotate(-geometry.rotation_degrees,resample=Image.Resampling.BILINEAR,
                             center=(image.width/2,image.height/2))
        bounds=canvas.getbbox()
        if bounds is None: return image.copy()
        tile=np.asarray(canvas.crop(bounds),dtype=np.float32)/255
        colors=np.stack([np.ones_like(tile),1-tile,np.zeros_like(tile)],axis=-1)*255
        base=np.asarray(image.crop(bounds),dtype=np.float32)
        opacity=alpha*tile[...,None]
        result=image.copy()
        result.paste(Image.fromarray(np.uint8(np.clip(base*(1-opacity)+colors*opacity,0,255))),bounds[:2])
        return result
    x0, y0 = max(0, geometry.x), max(0, geometry.y)
    x1, y1 = min(image.width, geometry.x + geometry.side), min(image.height, geometry.y + geometry.side)
    if x1 <= x0 or y1 <= y0:
        return image.copy()
    tile = np.array(heat.crop((x0 - geometry.x, y0 - geometry.y, x1 - geometry.x, y1 - geometry.y))) / 255.0
    colors = np.stack([np.ones_like(tile), 1 - tile, np.zeros_like(tile)], axis=-1) * 255
    base = np.array(image.crop((x0, y0, x1, y1)), dtype=float)
    opacity = alpha * tile[..., None]
    blend = np.uint8(np.clip(base * (1 - opacity) + colors * opacity, 0, 255))
    result = image.copy()
    result.paste(Image.fromarray(blend), (x0, y0))
    return result
