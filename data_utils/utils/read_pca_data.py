from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import scipy.io
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import csv


def readCompressed(image):
    #get decompress image (image is a .mat file of pca compressed data)
    im = scipy.io.loadmat(image)
    wc, pcc, wid, hei = im['wc'], im['pcc'], im['wid'], im['hei']
    spectra = np.matmul(pcc, np.transpose(wc)) #pcc * wc'

    h = np.reshape(np.transpose(spectra)[:,:, None], (wid[0][0],hei[0][0], len(spectra))) #[:,:,None] makes 2d np array into 3d np array
    
    # read and crop the .csv wavelengths file
    with open('sample_data/fruitdata/matlabHyper/hyperWavelengths.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    w = data[24:389]
    
    return h, w
    
    