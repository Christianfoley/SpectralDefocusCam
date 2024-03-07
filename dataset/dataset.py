from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import scipy.io
import cv2
import torch
import numpy as np
import random as rand
import os
import glob
import time

from dataset.preprocess_data import read_compressed


def get_data(
    batch_size,
    data_split,
    base_path,
    patch_size=(256, 256),
    workers=1,
    apply_rand_aug=True,
    shuffle=False,
):
    assert os.path.exists(base_path), "base data path does not exist"
    pavia_chunked = glob.glob(os.path.join(base_path, "paviadata_chunked/Pavia*.mat"))
    fruitset_pca = glob.glob(os.path.join(base_path, "fruitdata/pca/*.mat"))
    harvard = glob.glob(os.path.join(base_path, "harvard/CZ_hsdb/*.mat"))
    harvard_indoor = glob.glob(os.path.join(base_path, "harvard/CZ_hsdbi/*.mat"))

    # load pavia images (validation set)
    pavia_data = SpectralDataset(
        pavia_chunked,
        transforms.Compose(
            [
                # Resize(),
                subImageRand(patch_size),
                chooseSpectralBands(interp=True),
            ]
            if apply_rand_aug
            else [
                # Resize(),
                chooseSpectralBands(interp=True),
            ]
        ),
        tag=["paviaU", "pavia"],
    )

    # load giessen images
    fruit_data = SpectralDataset(
        fruitset_pca,
        transforms.Compose(
            [
                readCompressed(),
                # Resize(),
                subImageRand(patch_size),
                chooseSpectralBands(),
            ]
            if apply_rand_aug
            else [
                readCompressed(),
                # Resize(),
                chooseSpectralBands(),
            ]
        ),
    )
    # load harvard images
    harvard_data = SpectralDataset(
        harvard_indoor + harvard,
        transforms.Compose(
            [
                # Resize(),
                subImageRand(patch_size),
                chooseSpectralBands(),
            ]
            if apply_rand_aug
            else [
                # Resize(),
                chooseSpectralBands(),
            ]
        ),
        tag="ref",
    )

    # wrap all training sets, specify transforms and partition
    all_data = SpectralWrapper(
        datasets=[pavia_data, fruit_data, harvard_data],
        transform=transforms.Compose(
            [
                Normalize(),
                RandFlip(),
                toTensor(),
            ]
            if apply_rand_aug
            else [
                Normalize(),
                toTensor(),
            ]
        ),
        test_transform=transforms.Compose(
            [
                Normalize(),
                toTensor(),
            ]
        ),
    )
    train, val, test = all_data.partition(*data_split)

    # make dataloaders for pytorch
    train_loader, val_loader, test_loader = None, None, None
    if data_split[0] > 0:
        train_loader = DataLoader(
            train, batch_size, shuffle=shuffle, num_workers=workers
        )
    if data_split[1] > 0:
        val_loader = DataLoader(val, batch_size, shuffle=shuffle, num_workers=workers)
    if data_split[2] > 0:
        test_loader = DataLoader(test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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

        self.readtime = 0

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
        sample["input"] = sample["image"]

        self.readtime += sum([ds.readtime for ds in self.datasets])
        return sample

    def __len__(self):
        return self.length

    def partition(self, train_frac, val_frac, test_frac):
        assert train_frac + val_frac + test_frac == 1, "partitions must sum to 1"

        # split each dataset into partitions
        partitions = []
        for part in ["train", "val", "test"]:
            partition_datasets = []
            for j, dataset in enumerate(self.datasets):
                # this is awkward but oh well
                train_idx = int(len(dataset) * train_frac)
                val_idx = int(len(dataset) * (train_frac + val_frac))
                test_idx = int(len(dataset))

                if part == "train":
                    img_dirs_frac = dataset.img_dirs[0:train_idx]
                elif part == "val":
                    img_dirs_frac = dataset.img_dirs[train_idx:val_idx]
                elif part == "test":
                    img_dirs_frac = dataset.img_dirs[val_idx:test_idx]

                dataset_frac = SpectralDataset(
                    img_dirs_frac, dataset.transform, dataset.tag
                )
                partition_datasets.append(dataset_frac)

            transform = self.transform
            if part == "test":
                transform = self.test_transform
            partitions.append(SpectralWrapper(partition_datasets, transform))

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

        self.readtime = 0

    def __len__(self):
        return len(self.img_dirs)

    # if transform uses subImageRand, you can call the same item over and over
    def __getitem__(self, idx):
        if self.tag == None:
            start = time.time()
            image = scipy.io.loadmat(self.img_dirs[idx])
            self.readtime += time.time() - start
        elif type(self.tag) == list:
            start = time.time()
            dict = scipy.io.loadmat(self.img_dirs[idx])
            self.readtime += time.time() - start
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
    def __init__(self, output_size=(768, 768)):
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
        if self.interp:
            pass
        sample["image"] = sample["image"][..., self.bands[0] : self.bands[1]]
        return sample


class toTensor(object):
    """
    performs the numpy-pytorch tensor transpose. outputs a tensor
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
        print(f"calling subimage: {xRand, yRand, self.output_size}")
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
        sample["image"] = read_compressed(sample["image"])
        return sample
