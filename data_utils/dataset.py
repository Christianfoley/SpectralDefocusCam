from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import scipy.io
import cv2
import torch
import numpy as np
import random as rand


class SpectralWrapper(Dataset):
    """
    Wrapper for multiple SpectralDatasets. Applies its own set of transforms.
    Also can partition data into train, val, test, returning subsets of
    wrappers which draw from the randomized partitions.
    """

    def __init__(self, datasets, transform=None, test_transform=None):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.length = np.sum(self.lengths)
        self.transform = transform
        self.test_transform = test_transform
        # print('datasets:', self.datasets, '\n', 'lengths:', self.lengths, '\n', 'totallength:', self.length)

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError(f"{index} exceeds {self.length}")
        i = 0
        for length in self.lengths:
            if index < length:
                break
            index = index - length
            i = i + 1

        sample = self.datasets[i].__getitem__(index)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.length

    def partition(self, train_frac, val_frac, test_frac):
        assert train_frac + val_frac + test_frac == 1, "partitions must sum to 1"

        partitions = [train_frac, val_frac, test_frac]
        for i in range(len(partitions)):
            frac = partitions[i]

            # split each dataset into partitions
            partition_datasets = []
            for dataset in self.datasets:
                img_dirs_frac = dataset.img_dirs[int(len(dataset) * frac)]
                dataset_frac = SpectralDataset(
                    img_dirs_frac, dataset.transform, dataset.tag
                )
                partition_datasets.append(dataset_frac)

            transform = self.transform
            if i == 2:  # if test
                transform = self.test_transform
            partitions[i] = SpectralWrapper(partition_datasets, transform)

        return tuple(partitions)


class SpectralDataset(Dataset):
    """
    Dataset for feeding spectral simulation networks. Here data is ground truth and
    target, as model acts as forward+backwards pair.

    Takes list of .mat files with test_trasnformthe data stored as a numpy array under the 'tag'
    key.
    """

    def __init__(self, img_dir_list, transform=None, tag=None):
        self.img_dirs = img_dir_list
        self.transform = transform
        self.tag = tag
        # possible tags: ['ref','cspaces','header', 'wc', 'pcc', 'pavia']

    def __len__(self):
        return len(self.img_dirs)

    # if transform uses subImageRand, you can call the same item over and over
    def __getitem__(self, idx):
        if self.tag == None:
            image = scipy.io.loadmat(self.img_dirs[idx])
        elif type(self.tag) == list:
            dict = scipy.io.loadmat(self.img_dirs[idx])
            for subtag in self.tag:
                if subtag in dict:
                    image = dict[subtag]
                    break
        else:
            image = scipy.io.loadmat(self.img_dirs[idx])[self.tag]

        sample = {"image": image}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Resize(object):  # uses cv2.resize with a default size of 256,256
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, sample):
        sample["image"] = cv2.resize(sample["image"], self.output_size)
        return sample


class Normalize(object):
    def __call__(self, sample):
        sample["image"] = sample["image"] / np.max(sample["image"])
        return sample


class RandFlip(object):
    """
    Flip image on the x and y axes depending on random size 2 array
    """

    def __call__(self, sample):
        rand = np.random.randint(0, 2, 2)
        image = sample["image"].copy()
        if rand[0] == 1:
            image = np.flipud(image)
        if rand[1] == 1:
            image = np.fliplr(image)
        sample["image"] = image
        return sample


class chooseSpectralBands(object):
    """
    Selects spectral bands of object sample. Default bands idx: 0-30
    """

    def __init__(self, bands=(0, 30), interp=False):
        self.bands = bands
        self.interp = interp

    def __call__(self, sample):
        sample["image"] = sample["image"][..., self.bands[0] : self.bands[1]]
        return sample


class toTensor(object):
    """
    automatically performs the numpy-pytorch tensor transpose. outputs a tensor
    of the sample image
    """

    def __call__(self, sample, device=None):
        if device:
            sample["image"] = torch.tensor(
                sample["image"].copy().transpose(2, 0, 1),
                dtype=torch.float32,
                device=device,
            )
        else:
            sample["image"] = torch.tensor(
                sample["image"].copy().transpose(2, 0, 1), dtype=torch.float32
            )
        return sample


class subImageRand(object):
    """
    returns section of image at random with size equivalent to output size.
    image must be >= output_size
    """

    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, sample):
        shape = sample["image"].shape
        height, width, channels = shape[0], shape[1], shape[2]
        xRand = rand.randint(0, max(height - self.output_size[0], 0))
        yRand = rand.randint(0, max(width - self.output_size[1], 0))

        sample["image"] = sample["image"][
            xRand : (xRand + self.output_size[0]),
            yRand : (yRand + self.output_size[1]),
            :,
        ]
        return sample


class readCompressed(object):
    """
    # allows for the reading of pca reduced images
    """

    def __call__(self, sample):
        # get decompress image (image is a .mat file of pca compressed data)
        im = sample["image"]
        wc, pcc, wid, hei = im["wc"], im["pcc"], im["wid"], im["hei"]
        spectra = np.matmul(pcc, np.transpose(wc))

        # [:,:,None] makes 2d np array into 3d np array
        sample["image"] = np.reshape(
            np.transpose(spectra)[:, :, None], (wid[0][0], hei[0][0], len(spectra))
        )
        return sample
