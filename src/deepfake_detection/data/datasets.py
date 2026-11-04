from collections import defaultdict
from io import BytesIO
from pathlib import Path
import random
import time
import math
import numpy as np
from functools import lru_cache
import importlib
import warnings
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps,ImageChops
from torch.utils.data import Dataset, Sampler
from deepfake_detection.common.paths import Workspace, read_jsonl
from deepfake_detection.data.preprocessing import model_tensor, preprocessing_hash


def mixup_batch(images,labels,alpha):
    if not math.isfinite(alpha) or alpha<0:
        raise ValueError('MixUp alpha must be finite and nonnegative')
    if alpha==0: return images,labels
    if images.ndim<2 or len(images)!=len(labels) or images.device!=labels.device:
        raise ValueError('MixUp images/labels must share batch size and device')
    lam=float(np.random.beta(alpha,alpha))
    permutation=torch.randperm(len(images),device=images.device)
    targets=labels.float()
    return lam*images+(1-lam)*images[permutation],lam*targets+(1-lam)*targets[permutation]


def cutmix_batch(images,labels,alpha):
    if not math.isfinite(alpha) or alpha<0: raise ValueError('CutMix alpha must be finite and nonnegative')
    if alpha==0: return images,labels
    if images.ndim!=4 or len(images)!=len(labels) or images.device!=labels.device:
        raise ValueError('CutMix expects matching NCHW images/labels on one device')
    lam=float(np.random.beta(alpha,alpha))
    permutation=torch.randperm(len(images),device=images.device)
    height,width=images.shape[-2:]
    ratio=math.sqrt(1-lam)
    patch_w,patch_h=int(width*ratio),int(height*ratio)
    cx=int(torch.randint(width,(1,),device=images.device).item())
    cy=int(torch.randint(height,(1,),device=images.device).item())
    x1,x2=max(0,cx-patch_w//2),min(width,cx+(patch_w+1)//2)
    y1,y2=max(0,cy-patch_h//2),min(height,cy+(patch_h+1)//2)
    output=images.clone()
    output[:,:,y1:y2,x1:x2]=images[permutation,:,y1:y2,x1:x2]
    actual=1-(x2-x1)*(y2-y1)/(width*height)
    targets=labels.float()
    return output,actual*targets+(1-actual)*targets[permutation]


def _spectral_image(image):
    if isinstance(image,Image.Image):
        return torch.from_numpy(np.array(image.convert('RGB'),dtype=np.float32).transpose(2,0,1).copy()/255),True
    if not isinstance(image,torch.Tensor) or not image.is_floating_point() or image.ndim<2 or not torch.isfinite(image).all():
        raise ValueError('Spectral augmentation requires a PIL image or finite floating spatial tensor')
    return image,False


def _spectral_restore(tensor,pil,dtype):
    if pil:
        array=tensor.detach().cpu().clamp(0,1).mul(255).round().byte().permute(1,2,0).numpy()
        return Image.fromarray(array,'RGB')
    return tensor.to(dtype=dtype)


def fouriermix(image_a,image_b,alpha=.1):
    """Convex amplitude blend with source phase; tensor inputs keep their scale."""
    if not math.isfinite(alpha) or not 0<=alpha<=1: raise ValueError('FourierMix alpha must be in [0,1]')
    if alpha==0: return image_a
    a,pil=_spectral_image(image_a);b,_=_spectral_image(image_b)
    if a.shape!=b.shape or a.device!=b.device: raise ValueError('Spectral pairs must match shape and device')
    dtype=a.dtype
    a=a.float() if a.dtype in {torch.float16,torch.bfloat16} else a
    b=b.to(dtype=a.dtype)
    fa,fb=torch.fft.fft2(a),torch.fft.fft2(b)
    amplitude=(1-alpha)*fa.abs()+alpha*fb.abs()
    result=torch.fft.ifft2(amplitude*torch.exp(1j*torch.angle(fa))).real
    return _spectral_restore(result,pil,dtype)


def fouriermix_batch(images,probability,alpha=.1):
    if not math.isfinite(probability) or not 0<=probability<=1: raise ValueError('FourierMix probability must be in [0,1]')
    if probability==0: return images
    if images.ndim!=4: raise ValueError('FourierMix batch expects NCHW images')
    donor=images[torch.randperm(len(images),device=images.device)]
    mask=(torch.rand(len(images),device=images.device)<probability).reshape(-1,1,1,1)
    return torch.where(mask,fouriermix(images,donor,alpha),images)


def fft_amplitude_mix(image_a,image_b,alpha=.5):
    """One image: convex amplitudes and shortest-arc interpolation of phases."""
    if not math.isfinite(alpha) or not 0<=alpha<=1: raise ValueError('FFT amplitude mix alpha must be in [0,1]')
    if alpha==0: return image_a
    a,pil=_spectral_image(image_a);b,_=_spectral_image(image_b)
    if a.shape!=b.shape or a.device!=b.device: raise ValueError('Spectral pairs must match shape and device')
    dtype=a.dtype
    a=a.float() if a.dtype in {torch.float16,torch.bfloat16} else a
    fa,fb=torch.fft.fft2(a),torch.fft.fft2(b.to(dtype=a.dtype))
    phase_a,phase_b=torch.angle(fa),torch.angle(fb)
    difference=phase_b-phase_a
    delta=torch.atan2(torch.sin(difference),torch.cos(difference))
    phase=phase_a+alpha*delta
    result=torch.fft.ifft2(((1-alpha)*fa.abs()+alpha*fb.abs())*torch.exp(1j*phase)).real
    return _spectral_restore(result,pil,dtype)


def fft_amplitude_mix_batch(images,labels,probability,alpha=.5):
    if not math.isfinite(probability) or not 0<=probability<=1: raise ValueError('FFT amplitude mix probability must be in [0,1]')
    if probability==0 or alpha==0: return images,labels
    if images.ndim!=4 or len(images)!=len(labels): raise ValueError('Phase mixing expects matching NCHW images and targets')
    permutation=torch.randperm(len(images),device=images.device)
    mask=torch.rand(len(images),device=images.device)<probability
    output=torch.where(mask.reshape(-1,1,1,1),fft_amplitude_mix(images,images[permutation],alpha),images)
    label_mask=mask.reshape((len(images),)+(1,)*(labels.ndim-1))
    targets=torch.where(label_mask,(1-alpha)*labels.float()+alpha*labels[permutation].float(),labels.float())
    return output,targets


@lru_cache(maxsize=16)
def _dct_basis(size,device,dtype):
    coordinates=torch.arange(size,device=device,dtype=dtype)
    frequencies=coordinates[:,None]
    matrix=torch.cos(math.pi/size*(coordinates[None,:]+.5)*frequencies)*math.sqrt(2/size)
    matrix[0]*=1/math.sqrt(2)
    return matrix


def dct2(tensor):
    if not isinstance(tensor,torch.Tensor) or tensor.ndim<2 or not tensor.is_floating_point():
        raise ValueError('DCT expects a floating spatial tensor')
    tensor=tensor.float() if tensor.dtype in {torch.float16,torch.bfloat16} else tensor
    h,w=tensor.shape[-2:]
    left=_dct_basis(h,tensor.device,tensor.dtype);right=_dct_basis(w,tensor.device,tensor.dtype)
    return torch.matmul(torch.matmul(left,tensor),right.T)


def idct2(coefficients):
    h,w=coefficients.shape[-2:]
    left=_dct_basis(h,coefficients.device,coefficients.dtype);right=_dct_basis(w,coefficients.device,coefficients.dtype)
    return torch.matmul(torch.matmul(left.T,coefficients),right)


def random_dct_mask(image,num_masks=8,mask_size=8):
    if not isinstance(num_masks,int) or num_masks<0 or not isinstance(mask_size,int) or mask_size<1:
        raise ValueError('DCT mask count/size must be nonnegative/positive integers')
    if num_masks==0: return image
    tensor,pil=_spectral_image(image);dtype=tensor.dtype
    coefficients=dct2(tensor);h,w=coefficients.shape[-2:]
    mh,mw=min(mask_size,h),min(mask_size,w)
    mask=torch.ones((h,w),device=tensor.device,dtype=coefficients.dtype)
    for _ in range(num_masks):
        y=int(torch.randint(h-mh+1,(1,),device=tensor.device).item())
        x=int(torch.randint(w-mw+1,(1,),device=tensor.device).item())
        mask[y:y+mh,x:x+mw]=0
    return _spectral_restore(idct2(coefficients*mask),pil,dtype)


def dct_mask_batch(images,probability,num_masks=8,mask_size=8):
    if not math.isfinite(probability) or not 0<=probability<=1: raise ValueError('DCT masking probability must be in [0,1]')
    if probability==0: return images
    output=[]
    for image in images:
        output.append(random_dct_mask(image,num_masks,mask_size) if torch.rand(()).item()<probability else image)
    return torch.stack(output)


def jpeg_compress(image,quality_range=(60,100)):
    if len(quality_range)!=2 or not all(isinstance(value,int) for value in quality_range) or not 1<=quality_range[0]<=quality_range[1]<=100:
        raise ValueError('JPEG quality range must be ordered integer values in [1,100]')
    stream=BytesIO()
    image.convert('RGB').save(stream,format='JPEG',quality=random.randint(*quality_range))
    with Image.open(BytesIO(stream.getvalue())) as decoded:
        return decoded.convert('RGB')


def random_rescale(image,scale_range=(.5,1),prob=.3):
    if not math.isfinite(prob) or not 0<=prob<=1: raise ValueError('Rescale probability must be in [0,1]')
    if prob==0: return image
    if len(scale_range)!=2 or any(not math.isfinite(v) for v in scale_range) or not 0<scale_range[0]<=scale_range[1]<=4:
        raise ValueError('Rescale range must be ordered positive scales up to4')
    if scale_range[0]==scale_range[1]==1: return image
    if random.random()>=prob: return image
    scale=random.uniform(*scale_range)
    smaller=(max(1,round(image.width*scale)),max(1,round(image.height*scale)))
    return image.resize(smaller,Image.Resampling.BICUBIC).resize(image.size,Image.Resampling.BICUBIC)


def add_gaussian_noise(image,std):
    """Add noise in supplied normalized tensor coordinates, without pixel clipping."""
    if not math.isfinite(std) or std<0: raise ValueError('Gaussian noise std must be finite and nonnegative')
    if std==0: return image
    if not isinstance(image,torch.Tensor) or not image.is_floating_point():
        raise ValueError('Normalized Gaussian noise requires a floating tensor after model normalization')
    return image+torch.randn_like(image)*std


def add_salt_pepper(image,prob=.01):
    """Pixel-space corruption: shared RGB masks, endpoints0/1 for tensor inputs."""
    if not math.isfinite(prob) or not 0<=prob<=1: raise ValueError('Salt/pepper probability must be in [0,1]')
    if prob==0: return image
    tensor,pil=_spectral_image(image)
    if tensor.ndim not in {3,4}: raise ValueError('Salt/pepper expects CHW/NCHW pixel-space input')
    mask=torch.rand((*tensor.shape[:-3],*tensor.shape[-2:]),device=tensor.device).unsqueeze(-3)
    output=torch.where(mask<prob/2,torch.zeros_like(tensor),tensor)
    output=torch.where((mask>=prob/2)&(mask<prob),torch.ones_like(tensor),output)
    return _spectral_restore(output,pil,tensor.dtype)


def _probability(prob,name):
    if not math.isfinite(prob) or not 0<=prob<=1: raise ValueError(f'{name} probability must be in [0,1]')
    return prob>0 and random.random()<prob


def _ordered_range(values,name,*,minimum=0,maximum=float('inf')):
    if len(values)!=2 or any(not math.isfinite(value) for value in values) or not minimum<=values[0]<=values[1]<=maximum:
        raise ValueError(f'{name} must be an ordered finite range')
    return values


def random_erasing(image,prob=.1):
    """Erase one bounded pixel rectangle on a copy, before normalization."""
    if not _probability(prob,'Random erasing'): return image
    if not isinstance(image,Image.Image): raise ValueError('Random erasing expects a PIL image')
    width,height=image.size
    area=width*height*random.uniform(.02,.2)
    aspect=math.exp(random.uniform(math.log(.3),math.log(3.3)))
    ew=min(width,max(1,round(math.sqrt(area*aspect))))
    eh=min(height,max(1,round(math.sqrt(area/aspect))))
    x,y=random.randint(0,width-ew),random.randint(0,height-eh)
    output=image.copy()
    output.paste((0,0,0),(x,y,x+ew,y+eh))
    return output


def random_sharpen(image,factor_range=(.5,2),prob=.15):
    if not _probability(prob,'Sharpness'): return image
    factor_range=_ordered_range(factor_range,'Sharpness factor',minimum=0)
    return ImageEnhance.Sharpness(image).enhance(random.uniform(*factor_range))


def random_color_jitter(image,hue=.05,sat=.15,prob=.3):
    if not math.isfinite(hue) or not 0<=hue<=.5 or not math.isfinite(sat) or not 0<=sat<=1:
        raise ValueError('Hue/saturation jitter must be in [0,.5]/[0,1]')
    if (hue==0 and sat==0) or not _probability(prob,'Color jitter'): return image
    from torchvision.transforms.functional import adjust_hue
    if hue>0: image=adjust_hue(image,random.uniform(-hue,hue))
    if sat>0: image=ImageEnhance.Color(image).enhance(random.uniform(1-sat,1+sat))
    return image


def random_gamma(image,gamma_range=(.8,1.2),prob=.3):
    gamma_range=_ordered_range(gamma_range,'Gamma',minimum=1e-6)
    if gamma_range[0]==gamma_range[1]==1 or not _probability(prob,'Gamma'): return image
    tensor,pil=_spectral_image(image)
    if tensor.min()<0 or tensor.max()>1: raise ValueError('Gamma augmentation expects unnormalized pixel-space values in [0,1]')
    gamma=random.uniform(*gamma_range)
    return _spectral_restore(tensor.pow(gamma),pil,tensor.dtype)


def coarse_dropout(image,prob=.15,num_holes=4,hole_ratio=.2):
    if not isinstance(num_holes,int) or num_holes<0 or not math.isfinite(hole_ratio) or not 0<=hole_ratio<=1:
        raise ValueError('Coarse dropout requires nonnegative holes and a ratio in [0,1]')
    if num_holes==0 or hole_ratio==0 or not _probability(prob,'Coarse dropout'): return image
    tensor,pil=_spectral_image(image);output=tensor.clone()
    h,w=tensor.shape[-2:];hh,hw=max(1,round(h*hole_ratio)),max(1,round(w*hole_ratio))
    for _ in range(num_holes):
        y,x=random.randint(0,h-hh),random.randint(0,w-hw)
        output[...,y:y+hh,x:x+hw]=0
    return _spectral_restore(output,pil,tensor.dtype)


def random_grid_shuffle(image,grid_size=4,prob=.05):
    if not isinstance(grid_size,int) or grid_size<1: raise ValueError('Grid size must be a positive integer')
    if grid_size==1 or not _probability(prob,'Grid shuffle'): return image
    tensor,pil=_spectral_image(image)
    if tensor.ndim not in {3,4}: raise ValueError('Grid shuffle expects CHW/NCHW images')
    h,w=tensor.shape[-2:];grid=min(grid_size,h,w)
    ph,pw=(-h)%grid,(-w)%grid
    padded=torch.nn.functional.pad(tensor,(0,pw,0,ph),mode='replicate') if ph or pw else tensor
    th,tw=padded.shape[-2]//grid,padded.shape[-1]//grid
    order=list(range(grid*grid));random.shuffle(order)
    output=torch.empty_like(padded)
    for destination,source in enumerate(order):
        dy,dx=divmod(destination,grid);sy,sx=divmod(source,grid)
        output[...,dy*th:(dy+1)*th,dx*tw:(dx+1)*tw]=padded[...,sy*th:(sy+1)*th,sx*tw:(sx+1)*tw]
    return _spectral_restore(output[...,:h,:w],pil,tensor.dtype)


def random_perspective(image,scale=.05,prob=.1):
    if not math.isfinite(scale) or not 0<=scale<.5: raise ValueError('Perspective scale must be in [0,.5)')
    if scale==0 or not _probability(prob,'Perspective'): return image
    from torchvision.transforms.functional import perspective
    from torchvision.transforms import InterpolationMode
    w,h=image.size if isinstance(image,Image.Image) else (image.shape[-1],image.shape[-2])
    dx,dy=round(w*scale),round(h*scale)
    start=[[0,0],[w-1,0],[w-1,h-1],[0,h-1]]
    end=[[random.randint(0,dx),random.randint(0,dy)],
         [w-1-random.randint(0,dx),random.randint(0,dy)],
         [w-1-random.randint(0,dx),h-1-random.randint(0,dy)],
         [random.randint(0,dx),h-1-random.randint(0,dy)]]
    interpolation=InterpolationMode.BICUBIC if isinstance(image,Image.Image) else InterpolationMode.BILINEAR
    return perspective(image,start,end,interpolation=interpolation,fill=0)


def random_crop_pad(image,crop_ratio=.9,prob=.1):
    if not math.isfinite(crop_ratio) or not 0<crop_ratio<=1: raise ValueError('Crop/pad ratio must be in (0,1]')
    if crop_ratio==1 or not _probability(prob,'Crop/pad'): return image
    tensor,pil=_spectral_image(image);h,w=tensor.shape[-2:]
    ch,cw=max(1,round(h*crop_ratio)),max(1,round(w*crop_ratio))
    y,x=random.randint(0,h-ch),random.randint(0,w-cw)
    output=torch.zeros_like(tensor);py,px=(h-ch)//2,(w-cw)//2
    output[...,py:py+ch,px:px+cw]=tensor[...,y:y+ch,x:x+cw]
    return _spectral_restore(output,pil,tensor.dtype)


def elastic_deform(image,alpha=1,sigma=.05,prob=0):
    if not math.isfinite(alpha) or alpha<0 or not math.isfinite(sigma) or sigma<=0:
        raise ValueError('Elastic alpha/sigma must be nonnegative/positive finite values')
    if alpha==0 or not _probability(prob,'Elastic deformation'): return image
    try: ndimage=importlib.import_module('scipy.ndimage')
    except ImportError:
        warnings.warn('Elastic deformation skipped: optional scipy.ndimage is unavailable; no installation performed',RuntimeWarning,stacklevel=2)
        return image
    tensor,pil=_spectral_image(image)
    if tensor.ndim!=3 or tensor.requires_grad: raise ValueError('Elastic deformation is a preprocessing-only CHW/PIL operation')
    array=tensor.detach().cpu().numpy();h,w=array.shape[-2:]
    smooth=sigma*min(h,w)
    dx=ndimage.gaussian_filter(np.random.uniform(-1,1,(h,w)),smooth,mode='reflect')*alpha
    dy=ndimage.gaussian_filter(np.random.uniform(-1,1,(h,w)),smooth,mode='reflect')*alpha
    yy,xx=np.meshgrid(np.arange(h),np.arange(w),indexing='ij')
    warped=np.stack([ndimage.map_coordinates(channel,[yy+dy,xx+dx],order=1,mode='reflect') for channel in array])
    output=torch.from_numpy(warped.copy()).to(device=tensor.device,dtype=tensor.dtype)
    return _spectral_restore(output,pil,tensor.dtype)


def random_gaussian_blur(image,sigma_range=(.1,1),prob=.1):
    sigma_range=_ordered_range(sigma_range,'Gaussian blur sigma',minimum=0)
    if sigma_range[1]==0 or not _probability(prob,'Gaussian blur'): return image
    sigma=random.uniform(*sigma_range)
    if sigma==0: return image
    if isinstance(image,Image.Image): return image.filter(ImageFilter.GaussianBlur(sigma))
    from torchvision.transforms.functional import gaussian_blur
    dimension=min(image.shape[-2:])
    kernel=min(2*math.ceil(3*sigma)+1,dimension if dimension%2 else dimension-1)
    return gaussian_blur(image,[max(1,kernel)]*2,[sigma]*2)


def motion_blur(image,kernel_size=5,prob=.05):
    if not isinstance(kernel_size,int) or kernel_size<1 or kernel_size%2==0:
        raise ValueError('Motion blur kernel must be a positive odd integer')
    if kernel_size==1 or not _probability(prob,'Motion blur'): return image
    tensor,pil=_spectral_image(image)
    if tensor.ndim not in {3,4}: raise ValueError('Motion blur expects CHW/NCHW input')
    if kernel_size//2>=min(tensor.shape[-2:]): raise ValueError('Motion blur kernel exceeds image dimensions')
    kernel=torch.zeros((kernel_size,kernel_size),dtype=tensor.dtype,device=tensor.device)
    direction=random.choice(['horizontal','vertical','diagonal','antidiagonal'])
    if direction=='horizontal': kernel[kernel_size//2,:]=1
    elif direction=='vertical': kernel[:,kernel_size//2]=1
    elif direction=='diagonal': kernel.fill_diagonal_(1)
    else: kernel=torch.eye(kernel_size,dtype=tensor.dtype,device=tensor.device).flip(1)
    kernel/=kernel.sum()
    channels=tensor.shape[-3]
    batch=tensor.unsqueeze(0) if tensor.ndim==3 else tensor
    padded=torch.nn.functional.pad(batch,(kernel_size//2,)*4,mode='reflect')
    output=torch.nn.functional.conv2d(padded,kernel.expand(channels,1,kernel_size,kernel_size),groups=channels)
    return _spectral_restore(output[0] if tensor.ndim==3 else output,pil,tensor.dtype)


def random_grayscale(image,prob=.05):
    if not _probability(prob,'Grayscale'): return image
    from torchvision.transforms.functional import rgb_to_grayscale
    return rgb_to_grayscale(image,num_output_channels=3)


def _policy_fallback(image,num_ops,magnitude):
    """Documented PIL subset fallback, not a claim of the full native policy."""
    if not isinstance(image,Image.Image): raise ValueError('Fallback augmentation policy expects a PIL pixel image')
    severity=magnitude/30
    for _ in range(num_ops):
        operation=random.choice(['rotate','brightness','contrast','sharpness','autocontrast'])
        if operation=='rotate': image=image.rotate(random.uniform(-30,30)*severity,resample=Image.Resampling.BICUBIC)
        elif operation=='autocontrast': image=ImageOps.autocontrast(image)
        else:
            enhancer={'brightness':ImageEnhance.Brightness,'contrast':ImageEnhance.Contrast,'sharpness':ImageEnhance.Sharpness}[operation]
            image=enhancer(image).enhance(max(0,1+random.uniform(-.9,.9)*severity))
    return image


def randaugment(image,num_ops=2,magnitude=9):
    if not isinstance(num_ops,int) or not 0<=num_ops<=10 or not isinstance(magnitude,int) or not 0<=magnitude<=30:
        raise ValueError('RandAugment requires operations0..10 and magnitude0..30')
    if num_ops==0: return image
    try:
        transforms=importlib.import_module('torchvision.transforms')
        policy=getattr(transforms,'RandAugment',None)
    except ImportError: policy=None
    if policy is None: return _policy_fallback(image,num_ops,magnitude)
    return policy(num_ops=num_ops,magnitude=magnitude,interpolation=transforms.InterpolationMode.BILINEAR,fill=0)(image)


def trivialaugment(image,enabled=True,num_magnitude_bins=31):
    if not enabled: return image
    if not isinstance(num_magnitude_bins,int) or num_magnitude_bins<2:
        raise ValueError('TrivialAugment requires at least two magnitude bins')
    try:
        transforms=importlib.import_module('torchvision.transforms')
        policy=getattr(transforms,'TrivialAugmentWide',None)
    except ImportError: policy=None
    if policy is None: return _policy_fallback(image,1,random.randint(0,30))
    return policy(num_magnitude_bins=num_magnitude_bins,interpolation=transforms.InterpolationMode.BILINEAR,fill=0)(image)


def augmix(image,num_aug=3,depth=3,alpha=1,*,enabled=True):
    """Hendrycks-style Dirichlet chain mixture with a beta original-image blend.

    Uses the documented PIL augmentation subset; no JSD training objective is
    implied unless the separate consistency-loss switch is explicitly selected.
    """
    if not enabled or depth==0: return image
    if not isinstance(image,Image.Image): raise ValueError('AugMix expects a PIL pixel image')
    if not isinstance(num_aug,int) or not 1<=num_aug<=10 or not isinstance(depth,int) or not 1<=depth<=10 or not math.isfinite(alpha) or alpha<=0:
        raise ValueError('AugMix requires width/depth1..10 and positive alpha')
    weights=np.random.dirichlet([alpha]*num_aug)
    blend=float(np.random.beta(alpha,alpha))
    original=np.asarray(image.convert('RGB'),dtype=np.float32)/255
    mixture=np.zeros_like(original)
    for weight in weights:
        variant=_policy_fallback(image.copy(),depth,random.randint(1,10)*3)
        mixture+=float(weight)*np.asarray(variant.convert('RGB'),dtype=np.float32)/255
    output=(1-blend)*original+blend*mixture
    return Image.fromarray(np.uint8(np.clip(np.rint(output*255),0,255)),'RGB')


def style_transfer(image,style_images_path=None,style_weights_path=None,*,enabled=True):
    if not enabled: return image
    raise NotImplementedError('Style transfer is RESOURCE-PENDING: requires authorized local style images and compatible style-model weights plus a transfer adapter. No resource is fetched automatically.')


def _artifact_rectangle(height,width):
    rh=max(1,round(height*random.uniform(.2,.5)))
    rw=max(1,round(width*random.uniform(.2,.5)))
    y,x=random.randint(0,height-rh),random.randint(0,width-rw)
    return x,y,x+rw,y+rh


def inject_blend_artifact(image,prob=.05):
    if not _probability(prob,'Blend artifact'): return image
    if not isinstance(image,Image.Image): raise ValueError('Blend artifact injection expects a PIL pixel image')
    rectangle=_artifact_rectangle(image.height,image.width)
    blurred=image.filter(ImageFilter.GaussianBlur(random.uniform(.5,2)))
    output=image.copy();output.paste(blurred.crop(rectangle),rectangle[:2])
    return output


def inject_color_mismatch(image,prob=.05):
    if not _probability(prob,'Color mismatch'): return image
    tensor,pil=_spectral_image(image)
    if tensor.ndim!=3: raise ValueError('Color mismatch expects CHW/PIL input')
    h,w=tensor.shape[-2:];x1,y1,x2,y2=_artifact_rectangle(h,w)
    output=tensor.clone()
    shift=torch.tensor(np.random.uniform(-.15,.15,(tensor.shape[0],1,1)),device=tensor.device,dtype=tensor.dtype)
    output[:,y1:y2,x1:x2]=(output[:,y1:y2,x1:x2]+shift).clamp(0,1)
    return _spectral_restore(output,pil,tensor.dtype)


def inject_warp_artifact(image,prob=.05):
    """Local displacement near an approximate normalized facial-boundary ellipse."""
    if not _probability(prob,'Warp artifact'): return image
    tensor,pil=_spectral_image(image)
    if tensor.ndim!=3: raise ValueError('Warp artifact expects CHW/PIL input')
    h,w=tensor.shape[-2:]
    dtype=tensor.dtype
    tensor=tensor.float() if dtype in {torch.float16,torch.bfloat16} else tensor
    yy,xx=torch.meshgrid(torch.linspace(-1,1,h,device=tensor.device,dtype=tensor.dtype),torch.linspace(-1,1,w,device=tensor.device,dtype=tensor.dtype),indexing='ij')
    radius=torch.sqrt((xx/.65)**2+(yy/.85)**2)
    ring=torch.exp(-((radius-1)/.1)**2)
    grid=torch.stack([xx+ring*random.uniform(-.08,.08),yy+ring*random.uniform(-.08,.08)],dim=-1)
    output=torch.nn.functional.grid_sample(tensor[None],grid[None],mode='bilinear',padding_mode='border',align_corners=True)[0]
    return _spectral_restore(output,pil,dtype)


def _training_artifacts(image,label,config):
    if not config.get('artifact_injection',False): return image,label
    policy=config.get('artifact_label_policy')
    if policy not in {'pseudo_fake','preserve'}:
        raise ValueError('Enabled injection requires explicit artifact_label_policy=pseudo_fake or preserve')
    if policy=='pseudo_fake' and label!=1: return image,label
    original=image
    for function in (inject_blend_artifact,inject_color_mismatch,inject_warp_artifact):
        image=function(image,prob=config.get('artifact_prob',.05))
    changed=ImageChops.difference(original,image).getbbox() is not None
    return image,0 if policy=='pseudo_fake' and changed else label


class Augment:
    """Same corruption/geometric distributions for every label/source."""
    def __init__(self, mode="robust",training_config=None):
        if mode not in {"none", "basic", "robust","robust_v2"}:
            raise ValueError("augmentation must be none, basic, robust or robust_v2")
        self.mode = mode
        self.config=training_config or {}
        if sum(bool(self.config.get(key,False)) for key in ('randaugment','trivialaugment','augmix'))>1:
            raise ValueError('Choose one global augmentation policy: RandAugment, TrivialAugment or AugMix')
        rotation=self.config.get('rotation_range',5)
        if not math.isfinite(rotation) or not 0<=rotation<=180: raise ValueError('Rotation range must be in [0,180] degrees')

    def __call__(self, image):
        if self.mode == "none":
            return image
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if random.random() < 0.3:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.1))
        if random.random() < 0.3:
            from torchvision.transforms.functional import affine
            from torchvision.transforms import InterpolationMode
            size = image.width
            rotation=self.config.get('rotation_range',5)
            image = affine(image, angle=random.uniform(-rotation, rotation),
                           translate=[round(random.uniform(-0.03, 0.03) * size), round(random.uniform(-0.03, 0.03) * size)],
                           scale=random.uniform(0.9, 1.1), shear=0, interpolation=InterpolationMode.BICUBIC, fill=0)
        image=random_gaussian_blur(image,self.config.get('blur_sigma_range',[.1,1]),self.config.get('blur_prob',.1))
        # The controlled Branch B ablation adds only compression and rescaling.
        if self.mode in {"robust","robust_v2"}:
            probability=self.config.get('jpeg_prob',.5)
            if probability>0 and random.random() < probability:
                image=jpeg_compress(image,self.config.get('jpeg_quality_range',[60,100]))
            image=random_rescale(image,self.config.get('rescale_range',[.5,1]),self.config.get('rescale_prob',.3))
        if self.mode=='robust_v2' and self.config.get('salt_pepper_prob',0)>0:
            image=add_salt_pepper(image,self.config['salt_pepper_prob'])
        if self.mode=='robust_v2' and self.config.get('sharpen_prob',0)>0:
            image=random_sharpen(image,self.config.get('sharpen_factor_range',[.5,2]),self.config['sharpen_prob'])
        if self.mode=='robust_v2' and (self.config.get('hue_jitter',0)>0 or self.config.get('saturation_jitter',0)>0):
            image=random_color_jitter(image,self.config.get('hue_jitter',0),self.config.get('saturation_jitter',0),self.config.get('color_jitter_prob',.3))
        if self.mode=='robust_v2' and 'gamma_range' in self.config:
            image=random_gamma(image,self.config['gamma_range'],self.config.get('gamma_prob',.3))
        if self.mode=='robust_v2' and self.config.get('coarse_dropout_prob',0)>0:
            image=coarse_dropout(image,self.config['coarse_dropout_prob'],self.config.get('coarse_dropout_holes',4),self.config.get('coarse_dropout_ratio',.2))
        if self.mode=='robust_v2' and self.config.get('grid_shuffle_prob',0)>0:
            image=random_grid_shuffle(image,self.config.get('grid_shuffle_size',4),self.config['grid_shuffle_prob'])
        if self.mode=='robust_v2' and self.config.get('perspective_prob',0)>0:
            image=random_perspective(image,self.config.get('perspective_scale',.05),self.config['perspective_prob'])
        if self.mode=='robust_v2' and self.config.get('crop_pad_prob',0)>0:
            image=random_crop_pad(image,self.config.get('crop_pad_ratio',.9),self.config['crop_pad_prob'])
        if self.mode=='robust_v2' and self.config.get('elastic_prob',0)>0:
            image=elastic_deform(image,self.config.get('elastic_alpha',1),self.config.get('elastic_sigma',.05),self.config['elastic_prob'])
        if self.mode=='robust_v2' and self.config.get('motion_blur_prob',0)>0:
            image=motion_blur(image,self.config.get('motion_blur_kernel',5),self.config['motion_blur_prob'])
        if self.mode=='robust_v2' and self.config.get('grayscale_prob',0)>0:
            image=random_grayscale(image,self.config['grayscale_prob'])
        if self.mode=='robust_v2' and self.config.get('randaugment',False):
            image=randaugment(image,self.config.get('randaugment_num_ops',2),self.config.get('randaugment_magnitude',9))
        if self.mode=='robust_v2' and self.config.get('trivialaugment',False):
            image=trivialaugment(image,num_magnitude_bins=self.config.get('trivialaugment_bins',31))
        if self.mode=='robust_v2' and self.config.get('augmix',False):
            image=augmix(image,self.config.get('augmix_width',3),self.config.get('augmix_depth',3),self.config.get('augmix_alpha',1))
        if self.mode=='robust_v2' and self.config.get('style_transfer',False):
            image=style_transfer(image,self.config.get('style_images_path'),self.config.get('style_weights_path'))
        if self.mode=='robust_v2' and self.config.get('random_erase_prob',0)>0:
            image=random_erasing(image,self.config['random_erase_prob'])
        return image


class CropDataset(Dataset):
    def __init__(self, rows, cfg, *, training=False, workspace=None, domain_map=None):
        self.rows = list(rows)
        self.cfg = cfg
        self.training=training
        if training and cfg['training'].get('sbi_prob',0)>0:
            from deepfake_detection.models.research import master_landmarks
            for row in self.rows:
                if row['label']==1: master_landmarks(row,cfg['preprocessing']['master_size'])
        self.domain_map=dict(domain_map) if domain_map is not None else None
        if self.domain_map is not None and (not training or any(r['dataset'] not in self.domain_map for r in self.rows)):
            raise ValueError('Domain metadata requires training rows covered by the domain map')
        if training and cfg['training'].get('artifact_injection',False) and cfg['training'].get('artifact_label_policy') not in {'pseudo_fake','preserve'}:
            raise ValueError('Enabled injection requires explicit artifact_label_policy=pseudo_fake or preserve')
        self.workspace = workspace or Workspace()
        self.augment = Augment(cfg["training"]["augmentation"] if training else "none",cfg['training'])
        expected = preprocessing_hash(cfg)
        for row in self.rows:
            if row["label"] == 0 and not row.get("subject_verified"):
                raise ValueError("Fake subject provenance must be reviewed before training/evaluation")
            if row["label"] not in {0, 1}:
                raise ValueError("Crop labels must be fake=0, real=1")
            if row.get("preprocessing_hash") != expected:
                raise ValueError("Crop preprocessing contract does not match this experiment")
            if row.get("extraction_status", "completed") != "completed":
                raise ValueError("Partial/failed extraction is for inspection, not training/evaluation")
            self.workspace.output(row["crop_path"])

    def __len__(self):
        return len(self.rows)

    def read_master(self,index):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as image:
            image.load()
            if image.mode != "RGB" or image.size != (self.cfg["preprocessing"]["master_size"],) * 2:
                raise ValueError(f"Invalid saved master: {row['crop_path']}")
            image=image.copy()
        return image,int(row["label"])

    def __getitem__(self,index):
        image,label=self.read_master(index)
        image,label=self.self_blend(image,label,index)
        if self.training: image,label=_training_artifacts(image,label,self.cfg['training'])
        item=(model_tensor(self.augment(image),self.cfg["model"]),label)
        return item+(self.domain_map[self.rows[index]['dataset']],) if self.domain_map is not None else item

    def self_blend(self,image,label,index):
        probability=self.cfg['training'].get('sbi_prob',0) if self.training else 0
        if probability and label==1:
            from deepfake_detection.models.research import master_landmarks,self_blended_images
            points=master_landmarks(self.rows[index],self.cfg['preprocessing']['master_size'])
            image,applied=self_blended_images(image,probability,landmarks=points,return_applied=True)
            if applied: label=0
        return image,label

    def timed_item(self,index,clock=time.perf_counter):
        start=clock()
        image,label=self.read_master(index)
        loaded=clock()
        image,label=self.self_blend(image,label,index)
        if self.training: image,label=_training_artifacts(image,label,self.cfg['training'])
        image=self.augment(image)
        augmented=clock()
        tensor=model_tensor(image,self.cfg["model"])
        transformed=clock()
        return tensor,label,{"load":loaded-start,"augment":augmented-loaded,"transform":transformed-augmented}


class HierarchicalSampler(Sampler):
    """Equal class/domain/method/video selection; never sample validation or OOD."""
    def __init__(self, rows, *, samples, seed):
        if not rows or any(r.get("split") != "train" or r.get("dataset") == "dfd" for r in rows):
            raise ValueError("Sampler accepts training rows only; DFD stays external")
        self.samples, self.seed, self.epoch = samples, seed, 0
        self.groups = {}
        for index, row in enumerate(rows):
            self.groups.setdefault(row["label"], {}).setdefault(row["dataset"], {}).setdefault(row["method"], {}).setdefault(row["video_id"], []).append(index)
        if set(self.groups) != {0, 1}:
            raise ValueError("Training requires both classes")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.samples

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        for _ in range(self.samples):
            domain = self.groups[rng.choice([0, 1])]
            methods = domain[rng.choice(sorted(domain))]
            videos = methods[rng.choice(sorted(methods))]
            frames = videos[rng.choice(sorted(videos))]
            yield rng.choice(frames)


def crop_rows(names, workspace=None):
    ws = workspace or Workspace()
    return [row for name in names for row in read_jsonl(ws.manifest(name) / "crops.jsonl")]
