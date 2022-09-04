import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os, glob, cv2
from diffuser_utils import *

class Forward_Model(torch.nn.Module):
    # Model initialization 
    def __init__(self, mask,               # mask: response matrix for the spectral filter array
                 num_ims = 2,              # number of blurred images to simulate
                 w_init = None,            # initialization for the blur kernels e.g., w_init = [.01, 0.01]
                 cuda_device = 0,          # cuda device parameter 
                 blur_type = 'symmetric',  # symmetric or asymmetric blur kernel 
                 optimize_blur = False):   # choose whether to learn the best blur or not (warning, not stable)
        super(Forward_Model, self).__init__()
        self.cuda_device = cuda_device
        self.blur_type  = blur_type
        self.num_ims = num_ims
    
         ## Initialize constants
        self.DIMS0 = mask.shape[0]  # Image Dimensions
        self.DIMS1 = mask.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))     # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))     # Pad size

  
        if w_init is None: #if no blur specified, use default
            if self.blur_type == 'symmetric':
                w_init = np.linspace(.002, .035, num_ims) # sharp bound, blurry bound: (deflt:.002,0.035)
            else:
                w_init = np.linspace(.002, .01, num_ims)
                w_init =  np.repeat(np.array(w_init)[np.newaxis], self.num_ims, axis = 0).T
                w_init[:,1] *=.5
             
        print('optimize blur', optimize_blur)
        if self.blur_type == 'symmetric':
            #self.w_init =  np.repeat(np.array(w_init[0])[np.newaxis], self.num_ims, axis = 0)
            self.w_init = w_init
            self.w_blur = torch.nn.Parameter(torch.tensor(self.w_init, dtype = torch.float32, 
                                                          device = self.cuda_device,
                                                         requires_grad = optimize_blur))
        else:
            self.w_init =  w_init
            self.w_blur = torch.nn.Parameter(torch.tensor(self.w_init, dtype = torch.float32, 
                                                          device = self.cuda_device,
                                                         requires_grad = optimize_blur))

        # set up grid 
        x=np.linspace(-1,1,self.DIMS1); 
        y=np.linspace(-1,1,self.DIMS0); 
        X,Y=np.meshgrid(x,y)
        
        self.X = torch.tensor(X, dtype=torch.float32, device=self.cuda_device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=self.cuda_device)
        
        self.mask = np.transpose(mask, (2,0,1))
        self.mask_var = torch.tensor(self.mask, dtype=torch.float32, device=self.cuda_device).unsqueeze(0)
        self.psf = np.empty((num_ims, self.DIMS0, self.DIMS1))
        
    
    def make_psfs(self, ):
        psfs = []
        for i in range(0,self.num_ims):
            if self.blur_type == 'symmetric':
                psf= torch.exp(-((self.X/self.w_blur[i])**2+(self.Y/self.w_blur[i])**2))
            else:
                psf= torch.exp(-((self.X/self.w_blur[i,0])**2+(self.Y/self.w_blur[i,1])**2))
            psf = psf/torch.linalg.norm(psf, ord=float('inf'))
            psfs.append(psf)
        return torch.stack(psfs, 0)
    
    def Hfor(self):
        H = fft_psf(self, self.psfs)
        #print(H.shape)
        X = self.Xi.unsqueeze(2)[0]
        #print(X.shape)
        out = torch.fft.ifft2(H*X).real
        #print(out.shape)
        output = self.mask_var * crop_forward(self,  out)
        output = torch.sum(output, 2)
        #print('hfor out: ',output.shape)
        return output
    
    def Hadj(self, sim_meas):
        #print(sim_meas.shape, self.mask_var.shape)
        Hconj = torch.conj(fft_psf(self, self.psfs))
        sm = pad_zeros_torch(self, sim_meas.unsqueeze(2) * self.mask_var)
        #print(sm.shape)
        SM = fft_im(sm)#[0].unsqueeze(1)
        #print(SM.shape, Hconj.shape)
        adj_meas = torch.fft.ifft2(Hconj*SM).real
        return adj_meas
    
    # forward call for the model
    def forward(self, in_image):
        self.Xi = fft_im(my_pad(self,in_image)).unsqueeze(0)
        self.psfs = self.make_psfs()
        sim_output = self.Hfor()
        final_output = crop_forward(self, self.Hadj(sim_output))
        return final_output