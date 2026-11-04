"""Compact differentiable spectral/residual streams, computed in float32."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class SpectralEncoder(nn.Module):
    def __init__(self,channels,features=128):
        super().__init__()
        if features<1: raise ValueError('Positive frequency feature count required')
        self.features=features
        self.encoder=nn.Sequential(nn.Conv2d(channels,16,3,padding=1),nn.GELU(),nn.Conv2d(16,32,3,padding=1,stride=2),nn.GELU(),nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32,features))
    def encode(self,x):
        # Bound spectral branch activations independently of source resolution.
        return self.encoder(F.adaptive_avg_pool2d(x,(32,32)))


class FFTBranch(SpectralEncoder):
    def __init__(self,features=128): super().__init__(3,features)
    def forward(self,x):
        with torch.amp.autocast(x.device.type,enabled=False):
            spectrum=torch.fft.fftshift(torch.fft.fft2(x.float(),norm='ortho'),dim=(-2,-1))
            return self.encode(torch.log1p(spectrum.abs()))


class DCTBranch(SpectralEncoder):
    def __init__(self,features=128): super().__init__(3,features)
    def forward(self,x):
        from deepfake_detection.data.datasets import dct2
        with torch.amp.autocast(x.device.type,enabled=False):
            spectrum=dct2(F.adaptive_avg_pool2d(x.float(),(32,32)))
            return self.encode(torch.log1p(spectrum.abs()))


class NoiseResidualBranch(SpectralEncoder):
    """Local high-pass proxy; not a camera PRNU estimate without reference images."""
    def __init__(self,features=128): super().__init__(3,features)
    @staticmethod
    def residual(x): return x-F.avg_pool2d(F.pad(x,(2,2,2,2),mode='replicate'),5,stride=1)
    def forward(self,x):
        with torch.amp.autocast(x.device.type,enabled=False): return self.encode(self.residual(x.float()))


class WaveletBranch(SpectralEncoder):
    def __init__(self,features=128): super().__init__(12,features)
    @staticmethod
    def transform(x):
        x=F.pad(x,(0,x.shape[-1]%2,0,x.shape[-2]%2),mode='replicate')
        a=x[...,0::2,0::2]; b=x[...,0::2,1::2]; c=x[...,1::2,0::2]; d=x[...,1::2,1::2]
        return torch.cat([(a+b+c+d)/2,(a-b+c-d)/2,(a+b-c-d)/2,(a-b-c+d)/2],1)
    def forward(self,x):
        with torch.amp.autocast(x.device.type,enabled=False): return self.encode(self.transform(x.float()))


class GaborBank(nn.Module):
    def __init__(self,features=128,kernel_size=15):
        super().__init__()
        if kernel_size<3 or kernel_size%2==0 or features<1: raise ValueError('Invalid Gabor dimensions')
        axis=torch.arange(kernel_size,dtype=torch.float32)-kernel_size//2
        yy,xx=torch.meshgrid(axis,axis,indexing='ij'); kernels=[]
        for wavelength in (3.,6.):
            for theta in (0.,math.pi/4,math.pi/2,3*math.pi/4):
                rotated=xx*math.cos(theta)+yy*math.sin(theta)
                across=-xx*math.sin(theta)+yy*math.cos(theta)
                kernel=torch.exp(-(rotated.square()+.25*across.square())/(2*(.56*wavelength)**2))*torch.cos(2*math.pi*rotated/wavelength)
                kernel=kernel-kernel.mean(); kernel=kernel/kernel.norm().clamp_min(1e-8)
                kernels.append(kernel)
        self.register_buffer('kernels',torch.stack(kernels)[:,None])
        self.projection=nn.Linear(len(kernels),features); self.features=features
    def forward(self,x):
        with torch.amp.autocast(x.device.type,enabled=False):
            response=F.conv2d(x.float().mean(1,keepdim=True),self.kernels.float(),padding=self.kernels.shape[-1]//2)
            return self.projection(response.abs().mean((2,3)))
