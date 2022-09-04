import torch.nn.functional as F
import numpy as np
import torch
import torch.optim
import torch.nn as nn
from PIL import Image
import skimage
import skimage.transform
import glob
import matplotlib.pyplot as plt
import scipy.io

def interleave3d(t, spec_dim = 2): #takes 1,2,2,x,a,a returns 1,2,2x,a,a
    assert(spec_dim > 0) #we need 0th dimension to be the blurs
    spec_dim = spec_dim - 1
    batchsize = len(t)
    outlist = []
    for i in range(batchsize):
        ab = t[i]
        stacked = torch.stack([ab[0],ab[1]], dim = spec_dim)
        interleaved = torch.flatten(stacked, start_dim = spec_dim - 1, end_dim = spec_dim)
        outlist.append(interleaved)
    return torch.stack(outlist)

def fft_psf(model, h):
    h_complex = pad_zeros_torch(model, torch.complex(h,torch.zeros_like(h)))
    H = torch.fft.fft2(torch.fft.ifftshift(h_complex)).unsqueeze(1)
    return H

def fft_im(im):
    xc = torch.complex(im, torch.zeros_like(im))  
    Xi = torch.fft.fft2(xc)    
    return Xi

def tt(x, device = 'cuda:0'):
    return torch.tensor(x, dtype = torch.float32, device = device)

def load_mask(path = 'sample_data/calibration.mat'):
    spectral_mask=scipy.io.loadmat(path)
    mask=spectral_mask['mask']
    mask=mask[100:356,100:356,:-1]
    mask = (mask[...,0::2] + mask[...,1::2])/2
    mask = mask[...,0:30]
    return mask

def flip_channels(image):
    image_color = np.zeros_like(image);
    image_color[:,:,0] = image[:,:,2]; image_color[:,:,1]  = image[:,:,1]
    image_color[:,:,2] = image[:,:,0];
    return(image_color)
def pad_zeros_torch(model, x):
    PADDING = (model.PAD_SIZE1//2, model.PAD_SIZE1//2, model.PAD_SIZE0//2, model.PAD_SIZE0//2)
    return F.pad(x, PADDING, 'constant', 0)
def crop(model, x):
    C01 = model.PAD_SIZE0; C02 = model.PAD_SIZE0 + model.DIMS0              # Crop indices
    C11 = model.PAD_SIZE1; C12 = model.PAD_SIZE1 + model.DIMS1              # Crop indices
    return x[:, :, C01:C02, C11:C12]
def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def ifftshift2d(x):
    for dim in range(len(x.size()) - 1, 0, -1):
        x = roll_n(x, axis=dim, n=x.size(dim)//2)
        
    return x  # last dim=2 (real&imag)

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
###### Complex operations ##########
def complex_multiplication(t1, t2):
    real1, imag1 = torch.unbind(t1, dim=-1)
    real2, imag2 = torch.unbind(t2, dim=-1)
    return torch.stack([real1 * real2 - imag1 * imag2, real1 * imag2 + imag1 * real2], dim = -1)
def complex_abs(t1):
    real1, imag1 = torch.unbind(t1, dim=2)
    return torch.sqrt(real1**2 + imag1**2)
def make_real(c):
    out_r, _ = torch.unbind(c,-1)
    return out_r
def make_complex(r, i = 0):
    if i==0:
        i = torch.zeros_like(r, dtype=torch.float32)
    return torch.stack((r, i), -1)
def tv_loss(x, beta = 0.5):
    '''Calculates TV loss for an image `x`.
    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    x = x.cuda()
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    return torch.sum(dh[:, :, :-1] + dw[:, :, :, :-1] )

def my_pad(model, x):
    PADDING = (model.PAD_SIZE1//2, model.PAD_SIZE1//2, model.PAD_SIZE0//2, model.PAD_SIZE0//2)
    return F.pad(x, PADDING, 'constant', 0)
def crop_forward(model, x):
    C01 = model.PAD_SIZE0//2; C02 = model.PAD_SIZE0//2 + model.DIMS0              # Crop indices
    C11 = model.PAD_SIZE1//2; C12 = model.PAD_SIZE1//2 + model.DIMS1              # Crop indices
    return x[..., C01:C02, C11:C12]
#def crop_forward2(model, x):
#    C01 = model.PAD_SIZE0//2; C02 = model.PAD_SIZE0//2 + model.DIMS0//2              # Crop indices
#    C11 = model.PAD_SIZE1//2; C12 = model.PAD_SIZE1//2 + model.DIMS1//2              # Crop indices
#    return x[:, :, :, C01:C02, C11:C12]
class Forward_Model(torch.nn.Module):
    def __init__(self, h_in, shutter=0, cuda_device = 0):
        super(Forward_Model, self).__init__()
        self.cuda_device = cuda_device
       
         ## Initialize constants
        self.DIMS0 = h_in.shape[0]  # Image Dimensions
        self.DIMS1 = h_in.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))                           # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))                           # Pad size
        
        
#         self.h_var = torch.nn.Parameter(torch.tensor(h_in, dtype=torch.float32, device=self.cuda_device),
#                                             requires_grad=False)
#         self.h_zeros = torch.nn.Parameter(torch.zeros(self.DIMS0*2, self.DIMS1*2, dtype=torch.float32, device=self.cuda_device),
#                                           requires_grad=False)
        self.h_complex = pad_zeros_torch(self,torch.tensor(h_in, dtype=torch.cfloat, device=self.cuda_device).unsqueeze(0))
        self.const = torch.tensor(1/np.sqrt(self.DIMS0*2 * self.DIMS1*2), dtype=torch.float32, device=self.cuda_device)
        self.H = torch.fft.fft2(ifftshift2d(self.h_complex))
        
        self.shutter = np.transpose(shutter, (2,0,1))
        self.shutter_var = torch.tensor(self.shutter, dtype=torch.float32, device=self.cuda_device).unsqueeze(0)
    def Hfor(self, x):

        xc = torch.complex(x, torch.zeros_like(x))
        X = torch.fft.fft2(xc)

        HX = self.H*X
        out = torch.fft.ifft2(HX)
        out_r= out.real
        return out_r
    def forward(self, in_image):

        output = torch.sum(self.shutter_var * crop_forward(self,  self.Hfor(my_pad(self, in_image))), 1)
#         output = torch.sum((self.Hfor(my_pad(self, in_image))), 1)

        return output