#configuration file

import os 
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import torch.nn as torchnn
import torch
import torchvision.transforms
import csv
from ipywidgets import IntProgress
from IPython.display import display
import time
from skimage import io
import numpy as np
import scipy.io as spio
from scipy.interpolate import PchipInterpolator
from bisect import bisect
import jax.numpy as jnp
import jax.lax


xglobal = 0
losslist = 0
l2losslist = 0
dtype = torch.float32

def find_max_pixel(x):
    return (torch.sum(x,0)==torch.max(torch.sum(x,0))).nonzero().squeeze()

def get_split_idx(num_splits, split, target, dim):
    leng = target.shape[dim]
    split_size = int(np.floor(leng/num_splits))
    return split_size*split, split_size*(split+1)

def interpolate(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(1)
    xp = xp.unsqueeze(1)
    m = (fp[1:,:] - fp[:-1,:]) / (xp[1:,:] - xp[:-1,:])
    b = fp[:-1,:] - (m.mul(xp[:-1,:]) )
    indices = (torch.sum(torch.ge(x[:, None, :], xp[None,:, :]),-2)-1).clamp(0,xp.shape[0]-1)
    return m[indices.squeeze(),:]*x + b[indices.squeeze(),:] 

def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

def sumFilterArray(filterstack,wv,wvmin,wvmax,wvstep):
    #filterstack is ndarray.  wvmin, wvmax, wvstep are scalers
    #returns new ndarray msum with dimensions yx same as filterstack, and lambda depending on length of wvnew
    wvnew = torch.arange(wvmin,wvmax+wvstep,wvstep)
    #resample the filterarray
    #find where in wvindex meets wvnew
    j0 = torch.where(wvnew[0]==wv)[0][0]
    (dim0,dim1,dim2) = filterstack.shape
    dim0 = len(range(len(wvnew)))
    msum = torch.zeros((dim0,dim1,dim2))

    for k in range(dim0):
        #get index according to wv
        #sum and add
        if k<dim0-1:
            j1 = np.where(wvnew[k+1]==wv)[0][0]
        else:
            print(wvmax+wvstep)
            j0 = np.where(wvmax==wv)[0][0] #handle the last index
            j1 = np.where(wvmax+wvstep==wv)[0][0]
        msum[k,:,:] = torch.sum(filterstack[j0:j1,:,:],0)
        j0=j1
    return msum

#import specific tiff file from directory
#inputs - datafolder (string): directory, fname (string): specific file
#outputs - imarray (numpy array) containing tiff file as image
def importTiff(datafolder,fname):

    im = io.imread(os.path.join(datafolder, fname)).astype(float)
    imarray = torch.tensor(im, dtype = dtype)
    return imarray

#import all tiff files from a directory
#inputs - path (string): directory
#outputs - imageStack (torch array): stack of tiff images along dim 2
def tif_loader(path):
    fnames = [fname for fname in os.listdir(path) if fname.endswith('.tiff')]
    fnames.sort()
    prog = IntProgress(min=0, max=len(fnames)) # instantiate the bar
    display(prog) # display the bar
    for ii in range(len(fnames)):
        prog.value+=1
        file = fnames[ii]
        im = io.imread(os.path.join(path, file)).astype(float)
#         im = np.asarray(im).astype(float)
        if ii==0:
            imageStack = torch.zeros((len(fnames),im.shape[0],im.shape[1]))
        imageStack[ii,:,:] = torch.tensor(im, dtype = dtype)
    return imageStack

def cropci(im,ci):
    if len(im.shape)==3:
        return im[:,ci[0]:ci[1],ci[2]:ci[3]]
    elif len(im.shape)==2:
        return im[ci[0]:ci[1],ci[2]:ci[3]]
    else:
        print('not an image')
        return

def resample(psf,oldpix=1.67,newpix=5.3):
    zoom = oldpix/newpix
    s = psf.shape
    newsize = (int(s[1]*zoom),int(s[0]*zoom)) #flip to x,y
    pilpsf = Image.fromarray(psf)
    pilpsfzoom = pilpsf.resize(newsize)
    psf0 = np.array(pilpsfzoom)
    return psf0

def pad(x):
    #get size of x. double the size by padding with zeros
    dims = x.shape
    # only pad along last two dimensions, assume x and y
    padder = torch.nn.ConstantPad2d((int(np.floor(dims[-1]/2)),int(np.ceil(dims[-1]/2)),int(np.floor(dims[-2]/2)),int(np.ceil(dims[-2]/2))),0)
    xpad = padder(x)
    return xpad 
    
def crop(x, size):
    crop = torchvision.transforms.CenterCrop(size)
    return crop(x)

def forwardmodel3d(x,hfftpad,m):
    xfftpad = torch.fft.fft2(pad(x)) # get fft of padded object
    y = (torch.fft.ifftshift(torch.fft.ifft2(hfftpad*xfftpad), dim=(-1,-2))).real # do convolution
    y = crop(y, m.shape[1:]) * m # apply spectral filter
    y = torch.sum(y,dim=0)  # integrate of spectral dimension
    return y

def forwardmodel3d(x,hfftpad,m):
    xfftpad = torch.fft.fft2(pad(x)) # get fft of padded object
    y = (torch.fft.ifftshift(torch.fft.ifft2(hfftpad*xfftpad), dim=(-1,-2))).real # do convolution
    y = crop(y, m.shape[1:]) * m # apply spectral filter
    y = torch.sum(y,dim=0)  # integrate of spectral dimension
    return y

def adjointoperation(y,hfftpad,m): 
    y = y * m # this repeats Y and applies the hyper spectral filter
    ypad = pad(y) # zero pad by .5 on each side
    yfftpad = torch.fft.fft2(ypad) # 2d fft
    x = (torch.fft.ifftshift(torch.fft.ifft2(yfftpad*torch.conj(hfftpad)),dim=(-1,-2))).real # apply conv between psf and y
    x = crop(x, m.shape[1:]) # crop back down to size
    return x

def softthresh(x,thresh):
    #for all values less than thresh, set to 0, else subtract thresh.  #maintain pos/neg signage
    xout = torch.maximum(torch.abs(x)-thresh,torch.tensor(0,dtype = dtype))
    return torch.multiply(torch.sign(x),xout)#.clip(0,1e10) #need to implement complex softthresh. 

def nonneg(x,thresh=0): 
    #set negative values to zero
    return torch.maximum(x,torch.tensor(0, dtype = dtype)) 

def computel2loss(resid):
    l2loss = torch.linalg.norm(resid)
    return l2loss.cpu().detach()

def computel1loss(x):
    l1loss = torch.sum(torch.abs(x))
    return l1loss.cpu().detach()

def flatten(x):
    return torch.sum(x,dim=2)

def fistaloop3dGPU (xk,h,m,ytrue,specs):
    try:  #put everything inside a try/except loop to help free up memory in case of KeyboardInterrupt
        #xk = xpad
        #h = hpadfft 
        #unpack arguments
        kmax = specs['iterations']
        step_size = specs['step_size']
        tau1 = specs['tau1']
        thresh = tau1*step_size # threshold is equal to tau
        kprint = specs['print_every']
        # TDOO: add an if statement for total variation
        if specs['prior'] == 'soft-threshold':
            prox = lambda x, tmax: nonneg(softthresh(x,tmax))
            computeloss = lambda r,x: computel2loss(r)**2 + tau1*computel1loss(x) #weighted sum, remember to square the l2-loss!
        if specs['prior'] == 'non-negativity':
            prox = nonneg
            computeloss = lambda r,x: computel2loss(r)**2 #remember to square the l2-loss!
        else: #do nothing and just return x
            prox = lambda x, tmax : x
            computeloss = lambda r,x: computel2loss(r)**2 #remember to square the l2-loss!
        xkm1 = xk
        kcheck = specs['listevery']

        # store into global variables in case function doesn't run to completion
        global xglobal
        global losslist
        global l2losslist
        losslist = np.zeros(0) #reinitialize 
        l2losslist = np.zeros(0) #reinitialize 
        xglobal = torch.zeros_like(xk).to('cpu')  #keep off the gpu to save memory

        tk = 1 #fista variable
        vk = xk #fista variable

        # gradient descent loop
        for k in range(kmax):
            if np.mod(k,kcheck)==0 or k==kmax-1:
                print(k)
                xglobal = xk.cpu() # create a copy on cpu
            #compute yest
            yest = forwardmodel3d(xk,h,m)
            #compute residual
            resid = yest-ytrue #unpadded
            #gradient update
            gradupdate = adjointoperation(resid,h,m) #remember to use adjoint instead of inverse!
            vk = vk - step_size*gradupdate
            #proximal update
            xk = prox(vk,thresh)
            #fista code - from Beck et al paper
            tkp1 = (1 + np.sqrt(1+4*np.square(tk)))/2
            vkp1 = xk + (tk - 1)/(tkp1)*(xk - xkm1)
            #update fista variables
            vk = vkp1
            tk = tkp1
            xkm1 = xk

            #compute loss
            l2loss = computel2loss(resid)
            totloss = computeloss(resid,xk) # depends on prior
            losslist = np.append(losslist,totloss)
            l2losslist = np.append(l2losslist,l2loss)

            # plot every once in a while
            if np.mod(k,kprint)==0 or k==kmax-1:
                plt.figure(figsize = (12,4))
                plt.subplot(1,3,1)
                xcrop = (flatten(xk)).cpu().detach().numpy() #zoom to centerish
                plt.imshow(xcrop)
                plt.title('X Estimate (Zoomed)')
                plt.subplot(1,3,2)
                ycrop = (yest).cpu().detach().numpy() #zoom to centerish
                plt.imshow(ycrop)
                plt.title('Y Estimate (Zoomed)')
                plt.subplot(1,3,3)
                #plt.plot(losslist,'b')
                plt.plot(l2losslist,'r')
                plt.xlabel('Iteration')
                plt.ylabel('Cost Function')
                plt.title('L2 Loss Only')
                plt.tight_layout()
                plt.show()

        return (xk,gradupdate,vk)
    except KeyboardInterrupt:
        torch.cuda.empty_cache()
        return

def loadspectrum(path):
    file = open(path)
    csvreader = csv.reader(file)
    header = next(csvreader)
    rows = []
    for row in csvreader:
    #     print(row)
        rows.append(row)
    file.close()
    spec = rows[32:-1]
    wavelength = []
    intensity = []
    for ii in spec:
        vals = ii[0].split(';')
        wavelength.append(float(vals[0]))
        intensity.append(float(vals[1]))
    return (wavelength,intensity)

#helper functions
def gauss(x, mean, std):
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))

def gauss_green(x):
    std = 20
    mean = 550
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))

def gauss_red(x):
    std = 50
    mean = 625
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))

def gauss_blue(x):
    std = 42
    mean = 450
    return 1/(std*np.sqrt(6.28))*np.exp(-(x-mean)**2/(2*std**2))

def make_false_color_filter(waves, scaling=[1,1,2.5], device = 'cpu'):
    red_filt = gauss_red(waves)
    red_filt = torch.tensor(red_filt/np.max(red_filt)*scaling[0], device = device, dtype = dtype).unsqueeze(-1).unsqueeze(-1)

    green_filt = gauss_green(waves)
    green_filt = torch.tensor(green_filt/np.max(green_filt)*scaling[1], device = device, dtype = dtype).unsqueeze(-1).unsqueeze(-1)

    blue_filt = gauss_blue(waves)
    blue_filt = torch.tensor(blue_filt/np.max(blue_filt)*scaling[2], device = device, dtype = dtype).unsqueeze(-1).unsqueeze(-1)

    return torch.stack([red_filt, blue_filt, green_filt])

def make_false_color_img(img, filt):
    return torch.sum(img.unsqueeze(0)*filt, 1).permute(1,2,0)


def nonneg_soft_thresh(gamma, signal):
    binary = (signal-gamma) > 0
    sign = torch.sign(signal)
    return (signal.abs()-gamma) * binary * sign

def soft_py(x, tau):
    threshed = torch.maximum(torch.abs(x)-tau, torch.tensor(0, dtype = dtype))
    threshed = threshed*torch.sign(x)
    return threshed


# total variation functions
def ht3(x, ax, shift, thresh, device):
    C = 1./np.sqrt(2.)
    
    if shift == True:
        x = torch.roll(x, -1, dims = ax)
    if ax == 0:
        w1 = C*(x[1::2,:,:] + x[0::2, :, :])
        w2 = soft_py(C*(x[1::2,:,:] - x[0::2, :, :]), thresh)
    elif ax == 1:
        w1 = C*(x[:, 1::2,:] + x[:, 0::2, :])
        w2 = soft_py(C*(x[:,1::2,:] - x[:,0::2, :]), thresh)
    elif ax == 2:
        w1 = C*(x[:,:,1::2] + x[:,:, 0::2])
        w2 = soft_py(C*(x[:,:,1::2] - x[:,:,0::2]), thresh)
    return w1, w2

def iht3(w1, w2, ax, shift, shape, device):
    
    C = 1./np.sqrt(2.)
    y = torch.zeros(shape,device=device)

    x1 = C*(w1 - w2); x2 = C*(w1 + w2); 
    if ax == 0:
        y[0::2, :, :] = x1
        y[1::2, :, :] = x2
     
    if ax == 1:
        y[:, 0::2, :] = x1
        y[:, 1::2, :] = x2
    if ax == 2:
        y[:, :, 0::2] = x1
        y[:, :, 1::2] = x2
        
    
    if shift == True:
        y = torch.roll(y, 1, dims = ax)
    return y

def tv3dApproxHaar(x, tau, alpha):
    device = x.device
    D = 3
    fact = np.sqrt(2)*2

    thresh = D*fact
    

    y = torch.zeros_like(x,device=device)
    for ax in range(0,len(x.shape)):
        if ax == 0:
            t_scale = alpha
        else:
            t_scale = tau;

        w0, w1 = ht3(x, ax, False, thresh*t_scale, device)
        w2, w3 = ht3(x, ax, True, thresh*t_scale, device)
        
        t1 = iht3(w0, w1, ax, False, x.shape, device)
        t2 = iht3(w2, w3, ax, True, x.shape, device)
        y = y + t1 + t2
        
    y = y/(2*D)
    return y

def HSI2RGB(wY,HSI,ydim,xdim,d,threshold):
# wY: wavelengths in nm
# Y : HSI as a (#pixels x #bands) matrix,
# dims: x & y dimension of image
# d: 50, 55, 65, 75, determines the illuminant used, if in doubt use d65
# thresholdRGB : True if thesholding should be done to increase contrast
#
#
# If you use this method, please cite the following paper:
#  M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson, 
#  H. Deborah and J. R. Sveinsson, 
#  "Creating RGB Images from Hyperspectral Images Using a Color Matching Function", 
#  IEEE International Geoscience and Remote Sensing Symposium, Virtual Symposium, 2020
#
#  @INPROCEEDINGS{hsi2rgb, 
#  author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson} 
#  and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}}, 
#  booktitle={IEEE International Geoscience and Remote Sensing Symposium}, 
#  title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function}, 
#  year={2020}, volume={}, number={}, pages={}}
#
# Paper is available at
# https://www.researchgate.net/profile/Jakob_Sigurdsson
#
#

    
    # Load reference illuminant
    D = spio.loadmat('/home/emarkley/Workspace/PYTHON/HyperSpectralDiffuserScope/D_illuminants.mat')
    w = D['wxyz'][:,0]
    x = D['wxyz'][:,1]
    y = D['wxyz'][:,2]
    z = D['wxyz'][:,3]
    D = D['D']
    
    i = {50:2,
         55:3,
         65:1,
         75:4}
    wI = D[:,0];
    I = D[:,i[d]];
    # Changed I for flouresence imaging
    I[:] = 100
    
    # Interpolate to image wavelengths
    I = PchipInterpolator(wI,I,extrapolate=True)(wY) # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w,x,extrapolate=True)(wY) # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w,y,extrapolate=True)(wY) # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w,z,extrapolate=True)(wY) # interp1(w,z,wY,'pchip','extrap')';

    # Truncate at 780nm
    i=bisect(wY, 800)
    HSI=HSI[:,0:i]/HSI.max()
    wY=wY[:i]
    I=I[:i]
    x=x[:i]
    y=y[:i]
    z=z[:i]
    
    # Compute k
    k = 1/np.trapz(y * I, wY)
    
    # Compute X,Y & Z for image
    X = k * np.trapz(HSI @ np.diag(I * x), wY, axis=1)
    Z = k * np.trapz(HSI @ np.diag(I * z), wY, axis=1)
    Y = k * np.trapz(HSI @ np.diag(I * y), wY, axis=1)
    
    XYZ = np.array([X, Y, Z])
    
    # Convert to RGB
    M = np.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]]);
    
    sRGB=M@XYZ;
    
    # Gamma correction
    gamma_map = sRGB >  0.0031308;
    sRGB[gamma_map] = 1.055 * np.power(sRGB[gamma_map], (1. / 2.4)) - 0.055;
    sRGB[np.invert(gamma_map)] = 12.92 * sRGB[np.invert(gamma_map)];
    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB[sRGB > 1] = 1;
    sRGB[sRGB < 0] = 0;
    
    if threshold:
        for idx in range(3):
            y = sRGB[idx,:];
            a,b = np.histogram(y,100)
            b = b[:-1] + np.diff(b)/2
            a=np.cumsum(a)/np.sum(a)
            th = b[0]
            i = a<threshold;
            if i.any():
                th=b[i][-1];
            y=y-th
            y[y<0] = 0

            a,b=np.histogram(y,100)
            b = b[:-1] + np.diff(b)/2
            a=np.cumsum(a)/np.sum(a);
            i = a > 1-threshold
            th=b[i][0]
            y[y>th]=th
            y=y/th
            sRGB[idx,:]=y
        
    R = np.reshape(sRGB[0,:],[ydim,xdim]);
    G = np.reshape(sRGB[1,:],[ydim,xdim]);
    B = np.reshape(sRGB[2,:],[ydim,xdim]);
    return np.transpose(np.array([R,G,B]),[1,2,0])



def HSI2RGB_gpu(wY, HSI, ydim, xdim, d, threshold, device='cpu'):
    # Make tensors
    wY = torch.tensor(wY, device=device).double()
    HSI = torch.tensor(HSI, device=device).double()

    # Load reference illuminant
    D = spio.loadmat('/home/emarkley/Workspace/PYTHON/HyperSpectralDiffuserScope/D_illuminants.mat')
    w, x, y, z = [torch.tensor(D['wxyz'][:, i], device=device) for i in range(4)]
    D = torch.tensor(D['D'], device=device)
    
    i = {50:2, 55:3, 65:1, 75:4}
    
    wI = D[:, 0]
    I = D[:, i[d]]
    I[:] = 100
    
    # Interpolate to image wavelengths
    I = torch.tensor(PchipInterpolator(wI.cpu().numpy(), I.cpu().numpy())(wY.cpu().numpy()), device=device)
    x = torch.tensor(PchipInterpolator(w.cpu().numpy(), x.cpu().numpy())(wY.cpu().numpy()), device=device)
    y = torch.tensor(PchipInterpolator(w.cpu().numpy(), y.cpu().numpy())(wY.cpu().numpy()), device=device)
    z = torch.tensor(PchipInterpolator(w.cpu().numpy(), z.cpu().numpy())(wY.cpu().numpy()), device=device)
    
    # Truncate at 780nm
    i = bisect(wY.cpu().numpy(), 800)
    HSI = HSI[:, :i] / HSI.max()
    wY, I, x, y, z = wY[:i], I[:i], x[:i], y[:i], z[:i]
    
    # Compute k
    k = 1 / torch.trapz(y * I, wY)
    
    # Compute X,Y & Z for image
    X = k * torch.trapz(HSI @ torch.diag(I * x), wY)
    Z = k * torch.trapz(HSI @ torch.diag(I * z), wY)
    Y = k * torch.trapz(HSI @ torch.diag(I * y), wY)
    
    XYZ = torch.stack([X, Y, Z])
    
    # Convert to RGB
    M = torch.tensor([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]], device=device)
    
    sRGB = M.float() @ XYZ.float()
    
    # Gamma correction
    gamma_map = sRGB > 0.0031308
    sRGB[gamma_map] = 1.055 * torch.pow(sRGB[gamma_map], 1. / 2.4) - 0.055
    sRGB[~gamma_map] = 12.92 * sRGB[~gamma_map]
    
    # Clip values
    sRGB = torch.clamp(sRGB, 0, 1)
    
    if threshold:
        for idx in range(3):
            y = sRGB[idx]
            # Compute histogram
            hist = torch.histc(y, bins=100, min=0, max=1)
            b = torch.linspace(0, 1, steps=100)[:-1] + 0.005  # bin centers
            cum_hist = torch.cumsum(hist, dim=0) / torch.sum(hist)
            
            th = b[cum_hist < threshold][-1]
            y = y - th
            y[y < 0] = 0
            
            hist = torch.histc(y, bins=100, min=0, max=1)
            cum_hist = torch.cumsum(hist, dim=0) / torch.sum(hist)
            
            th = b[cum_hist > 1-threshold][0]
            y[y > th] = th
            y = y / th
            sRGB[idx] = y
            
    R = torch.reshape(sRGB[0], [ydim, xdim])
    G = torch.reshape(sRGB[1], [ydim, xdim])
    B = torch.reshape(sRGB[2], [ydim, xdim])
    
    return torch.stack([R, G, B]).permute(1,2,0)

def HSI2RGB_jax(wY,HSI,ydim,xdim,d,threshold):
# wY: wavelengths in nm
# Y : HSI as a (#pixels x #bands) matrix,
# dims: x & y dimension of image
# d: 50, 55, 65, 75, determines the illuminant used, if in doubt use d65
# thresholdRGB : True if thesholding should be done to increase contrast
#
#
# If you use this method, please cite the following paper:
#  M. Magnusson, J. Sigurdsson, S. E. Armansson, M. O. Ulfarsson, 
#  H. Deborah and J. R. Sveinsson, 
#  "Creating RGB Images from Hyperspectral Images Using a Color Matching Function", 
#  IEEE International Geoscience and Remote Sensing Symposium, Virtual Symposium, 2020
#
#  @INPROCEEDINGS{hsi2rgb, 
#  author={M. {Magnusson} and J. {Sigurdsson} and S. E. {Armansson} 
#  and M. O. {Ulfarsson} and H. {Deborah} and J. R. {Sveinsson}}, 
#  booktitle={IEEE International Geoscience and Remote Sensing Symposium}, 
#  title={Creating {RGB} Images from Hyperspectral Images using a Color Matching Function}, 
#  year={2020}, volume={}, number={}, pages={}}
#
# Paper is available at
# https://www.researchgate.net/profile/Jakob_Sigurdsson
#
#

    
    # Load reference illuminant
    D = spio.loadmat('/home/emarkley/Workspace/PYTHON/HyperSpectralDiffuserScope/D_illuminants.mat')
    w = D['wxyz'][:,0]
    x = D['wxyz'][:,1]
    y = D['wxyz'][:,2]
    z = D['wxyz'][:,3]
    D = D['D']
    
    i = {50:2,
         55:3,
         65:1,
         75:4}
    wI = D[:,0];
    I = D[:,i[d]];
    # Changed I for flouresence imaging
    I[:] = 100
    
    # Interpolate to image wavelengths
    I = PchipInterpolator(wI,I,extrapolate=True)(wY) # interp1(wI,I,wY,'pchip','extrap')';
    x = PchipInterpolator(w,x,extrapolate=True)(wY) # interp1(w,x,wY,'pchip','extrap')';
    y = PchipInterpolator(w,y,extrapolate=True)(wY) # interp1(w,y,wY,'pchip','extrap')';
    z = PchipInterpolator(w,z,extrapolate=True)(wY) # interp1(w,z,wY,'pchip','extrap')';

    # Truncate at 780nm
    i=bisect(wY, 800)
    HSI=HSI[:,0:i]/HSI.max()
    wY=wY[:i]
    I=I[:i]
    x=x[:i]
    y=y[:i]
    z=z[:i]
    
    # Compute k
    k = 1/jnp.trapz(y * I, wY)
    
    # Compute X,Y & Z for image
    X = k * jnp.trapz(HSI @ jnp.diag(I * x), wY, axis=1)
    Z = k * jnp.trapz(HSI @ jnp.diag(I * z), wY, axis=1)
    Y = k * jnp.trapz(HSI @ jnp.diag(I * y), wY, axis=1)
    
    XYZ = jnp.array([X, Y, Z])
    
    # Convert to RGB
    M = jnp.array([[3.2404542, -1.5371385, -0.4985314],
                  [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]]);
    
    sRGB=M@XYZ;
    
    # Gamma correction
    gamma_map = sRGB >  0.0031308;
    sRGB = sRGB.at[gamma_map].set(1.055 * jnp.power(sRGB[gamma_map], (1. / 2.4)) - 0.055)
    sRGB = sRGB.at[jnp.invert(gamma_map)].set(12.92 * sRGB[jnp.invert(gamma_map)])
    # Note: RL, GL or BL values less than 0 or greater than 1 are clipped to 0 and 1.
    sRGB = sRGB.at[sRGB > 1].set(1);
    sRGB = sRGB.at[sRGB < 0].set(0);
    
    if threshold:
        for idx in range(3):
            y = sRGB[idx,:];
            a,b = jnp.histogram(y,100)
            b = b[:-1] + jnp.diff(b)/2
            a=jnp.cumsum(a)/jnp.sum(a)
            th = b[0]
            i = a<threshold;
            if i.any():
                th=b[i][-1];
            y=y-th
            y = y.at[y<0].set(0)

            a,b=jnp.histogram(y,100)
            b = b[:-1] + jnp.diff(b)/2
            a=jnp.cumsum(a)/jnp.sum(a);
            i = a > 1-threshold
            th=b[i][0]
            y = y.at[y>th].set(th)
            y=y/th
            sRGB = sRGB.at[idx,:].set(y)
        
    R = jnp.reshape(sRGB[0,:],[ydim,xdim]);
    G = jnp.reshape(sRGB[1,:],[ydim,xdim]);
    B = jnp.reshape(sRGB[2,:],[ydim,xdim]);
    return jnp.transpose(jnp.array([R,G,B]),[1,2,0])


def jax_crop2D(target_shape, mat):
    y_margin = (mat.shape[-2] - target_shape[-2]) // 2
    x_margin = (mat.shape[-1] - target_shape[-1]) // 2
    if mat.ndim == 2:
        return mat[y_margin:-y_margin or None, x_margin:-x_margin or None]
    elif mat.ndim == 3:
        return mat[:, y_margin:-y_margin or None, x_margin:-x_margin or None]
    else:
        raise ValueError('crop2D only supports 2D and 3D arrays')

def jax_forward_model(object, spectral_filter, padded_fft_psf):
    paddings = ((0,0,0),(np.ceil(object.shape[1]/2).astype(int),np.floor(object.shape[1]/2).astype(int),0),(np.ceil(object.shape[2]/2).astype(int),np.floor(object.shape[2]/2).astype(int),0))
    padded_object = jax.lax.pad(object, 0.0, paddings)
    fft_object = jnp.fft.fft2(padded_object)
    fft_product = padded_fft_psf * fft_object

    ifft_product = jnp.fft.ifftshift(jnp.fft.ifft2(fft_product), axes=(1,2))
    ifft_product = abs(jax_crop2D(object.shape, ifft_product))
    ifft_product = ifft_product * spectral_filter
    measurement = jnp.sum(ifft_product, axis=0)
    return measurement.clip(0)

def jax_adjoint_model(measurement, spectral_filter, padded_fft_psf, padding):
    y = measurement[None, ...] * spectral_filter
    ypad = jax.lax.pad(y, 0.0, padding)
    yfftpad = jnp.fft.fft2(ypad)
    x = jnp.fft.ifftshift(jnp.fft.ifft2(yfftpad*jnp.conj(padded_fft_psf)), axes=(1,2))
    x = abs(jax_crop2D(measurement.shape, x))
    return x.clip(0)

def bw_visualize(image, title='', cmap='gray', colorbar=False, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap=cmap)
    if title!='':
        plt.title(title)
    if colorbar:
        plt.colorbar()
    # plt.axis('off')
    plt.show()