import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffuser_utils import *

class Forward_Model(torch.nn.Module):
    def __init__(self, shutter=0, num_ims = 2, w_init = [.01, 0.01], cuda_device = 0, blur_type = 'symmetric'):
        # blur type: single_axis or multi_axis 
        super(Forward_Model, self).__init__()
        self.cuda_device = cuda_device
        self.blur_type  = blur_type
       
    
         ## Initialize constants
        self.DIMS0 = shutter.shape[0]  # Image Dimensions
        self.DIMS1 = shutter.shape[1]  # Image Dimensions
        self.PAD_SIZE0 = int((self.DIMS0))                           # Pad size
        self.PAD_SIZE1 = int((self.DIMS1))     # Pad size
        
        self.num_ims = num_ims
        
        if self.blur_type == 'symmetric':
            self.w_init =  np.repeat(np.array(w_init[0])[np.newaxis], self.num_ims, axis = 0)
            self.w_blur = torch.nn.Parameter(torch.tensor(self.w_init, dtype = torch.float32, device = self.cuda_device))
        else:
            self.w_init =  np.repeat(np.array(w_init)[np.newaxis], self.num_ims, axis = 0)
            self.w_blur = torch.nn.Parameter(torch.tensor(self.w_init, dtype = torch.float32, device = self.cuda_device))

        x=np.linspace(-1,1,self.DIMS0); 
        y=np.linspace(-1,1,self.DIMS1); 
        X,Y=np.meshgrid(x,y)
        
        self.X = torch.tensor(X, dtype=torch.float32, device=self.cuda_device)
        self.Y = torch.tensor(Y, dtype=torch.float32, device=self.cuda_device)
        
        self.shutter = np.transpose(shutter, (2,0,1))
        self.shutter_var = torch.tensor(self.shutter, dtype=torch.float32, device=self.cuda_device).unsqueeze(0)
        
    def Hfor(self, x, i):
        if self.blur_type == 'symmetric':
            psf_sharp= torch.exp(-((self.X/self.w_blur[i])**2+(self.Y/self.w_blur[i])**2))
        else:
            psf_sharp= torch.exp(-((self.X/self.w_blur[i,0])**2+(self.Y/self.w_blur[i,1])**2))
            
        psf_sharp = psf_sharp/torch.linalg.norm(psf_sharp)
        h_complex = pad_zeros_torch(self,torch.complex(psf_sharp,torch.zeros_like(psf_sharp)).unsqueeze(0))
        H = torch.fft.fft2(ifftshift2d(h_complex))

        HX = H*self.Xi
        out = torch.fft.ifft2(HX)
        out_r= out.real
        return out_r
    
    def forward(self, in_image):
        x = my_pad(self, in_image)
        xc = torch.complex(x, torch.zeros_like(x))
        self.Xi = torch.fft.fft2(xc)

        out_list = []
        for i in range(0,self.num_ims):
            output = torch.sum(self.shutter_var * crop_forward(self, self.Hfor(my_pad(self, in_image), i)), 1)
            output = output/torch.max(output)
            out_list.append(output)
        final_output = torch.stack(out_list, 1)

        return final_output
    
class MyEnsemble(nn.Module):
    def __init__(self, model1, model2):
        super(MyEnsemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
    def forward(self, x, spec_dim = 2): #spectral channel needs to be padded up to power of 2
        self.output1 = self.model1(x)[0].unsqueeze(0)
        self.padded = self.spectral_pad(self.output1, spec_dim = 2, size = 2)
        self.output2 = self.model2(self.padded)[:,:,1:-1,:,:][0]
        return self.output2

    def spectral_pad(self, x, spec_dim = 2, size = -1):
        spec_channels = x.shape[spec_dim]
        padsize = 0
        while spec_channels & (spec_channels - 1) != 0:
            spec_channels += 1
            padsize += 1
        padsize = size if size >= 0 else padsize
        return F.pad(x, (0,0,0,0,padsize//2,padsize//2 + padsize % 2), 'constant', 0)
        
        