from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import scipy.io
import cv2
import torch
import numpy as np
import random as rand

class Wrapper(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets] 
        self.length = np.sum(self.lengths)
        #print('datasets:', self.datasets, '\n', 'lengths:', self.lengths, '\n', 'totallength:', self.length)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(f'{index} exceeds {self.length}')
        i = 0
        for length in self.lengths:
            if index < length:
                break
            index = index - length    
            i = i + 1
        return self.datasets[i].__getitem__(index)

    def __len__(self):
        return self.length

class SpectralDataset(Dataset): #initializer: can pass in transformation function/ composition of functions
    def __init__(self, img_dir_list, transform=None, target_transform=None, tag=None): #img_dir is a list of directory paths to image files
        self.img_dir = []
        for i in range(len(img_dir_list)):
            for sample in img_dir_list[i]:
                self.img_dir.append(sample)
        self.img_dir = img_dir_list
        self.transform = transform
        self.target_transform = target_transform
        self.tag = tag
        #possible tags: ['ref','cspaces','header', 'wc', 'pcc', 'pavia']
        
    def __len__(self):
        return len(self.img_dir)
    
    #if transform uses subImageRand, you can call the same item over and over
    def __getitem__(self, idx): 
        if self.tag == None:
            image = scipy.io.loadmat(self.img_dir[idx])
        elif type(self.tag) == list:
            dict = scipy.io.loadmat(self.img_dir[idx])
            for subtag in self.tag:
                if subtag in dict:
                    image = dict[subtag]
                    break
        else:
            image = scipy.io.loadmat(self.img_dir[idx])[self.tag]
        
        sample = {'image': image}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    
class Resize(object): # uses cv2.resize with a default size of 256,256 
    def __init__(self, output_size = (256, 256)):
        self.output_size = output_size
    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.output_size)
        return sample
    
class Normalize(object): 
    def __call__(self, sample):
        sample['image'] = sample['image']/np.max(sample['image'])
        return sample
    
class RandFlip(object): #flip image on the x and y axes depending on random size 2 array
    def __call__(self, sample):
        rand = np.random.randint(0,2,2)
        image = sample['image'].copy()
        if rand[0]==1:
            image = np.flipud(image)
        if rand[1]==1:
            image = np.fliplr(image)
        sample['image'] = image
        return sample
    
class chooseSpectralBands(object): # Selects spectral bands of object sample. Default bands idx: 0-30
    def __init__(self, bands = (0, 30), interp = False):
        self.bands = bands
        self.interp = interp
    def __call__(self, sample):
        sample['image'] = sample['image'][...,self.bands[0]:self.bands[1]]
        return sample
    
class toTensor(object): # automatically performs the numpy-pytorch tensor transpose. outputs a tensor of the sample image
    def __call__(self, sample):
        sample['image'] = torch.tensor(sample['image'].copy().transpose(2,0,1),dtype=torch.float32)
        return sample

#returns section of image at random with size equivalent to output size. image must be >= output_size
class subImageRand(object):
    def __init__(self, output_size = (256, 256)):
        self.output_size = output_size
    
    def __call__(self, sample):
        shape = sample['image'].shape
        height, width, channels = shape[0], shape[1], shape[2]
        xRand = rand.randint(0, max(height - self.output_size[0], 0))
        yRand = rand.randint(0, max(width - self.output_size[1], 0))
        
        sample['image'] = sample['image'][xRand:(xRand + self.output_size[0]), yRand:(yRand + self.output_size[1]), :]
        return sample

#allows for the reading of pca reduced images
class readCompressed(object):
    def __call__(self, sample):
        #get decompress image (image is a .mat file of pca compressed data)
        im = sample['image']
        wc, pcc, wid, hei = im['wc'], im['pcc'], im['wid'], im['hei']
        spectra = np.matmul(pcc, np.transpose(wc))

        #[:,:,None] makes 2d np array into 3d np array
        sample['image'] = np.reshape(np.transpose(spectra)[:,:, None], (wid[0][0],hei[0][0], len(spectra))) 
        return sample