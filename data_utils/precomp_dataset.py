from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torch
import numpy as np
import random as rand
import os
import glob
import time


def get_data_precomputed(
    batch_size,
    measurements_path,
    patch_size=(256, 256),
    workers=1,
):
    """
    Return dataloaders to access the train, test, val partitions for a "precomputed"
     or "premeasured" dataset.
    Dataset cannot be used to train, instead should be used to load camera measurements

    Parameters
    ----------
    batch_size : int
        size of loading batch
    measurements_path : str
        path to measurement directory, containing subdirs "train", "val", "test" with
        numbered measurements as .npy files
    patch_size : tuple, optional
        size of prediction patch, by default (256, 256)
    workers : int, optional
        number of dataloading workers, by default 1

    Returns
    -------
    tuple
        tuple of dataloaders (train, val, test)
    """

    augmentations = transforms.Compose(
        [RandFlip(), subImageRand(patch_size), toTensor()]
    )

    train = MeasStackDataset(os.path.join(measurements_path, "train"), augmentations)
    val = MeasStackDataset(os.path.join(measurements_path, "val"), augmentations)
    test = MeasStackDataset(os.path.join(measurements_path, "test"), toTensor())

    # make dataloaders for pytorch
    train_dataloader = DataLoader(train, batch_size, shuffle=True, num_workers=workers)
    val_dataloader = DataLoader(val, batch_size, shuffle=True, num_workers=workers)
    test_dataloader = DataLoader(test, batch_size)

    return train_dataloader, val_dataloader, test_dataloader


class MeasStackDataset(Dataset):
    def __init__(self, dir, transform=None):
        self.file_list = glob.glob(os.path.join(dir, "*.npy"))
        self.transform = transform
        self.readtime = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        start = time.time()
        x = np.load(self.file_list[idx])[0, :, 0]

        if self.transform:
            try:
                x = self.transform(x)
            except Exception as e:
                raise IndexError(
                    f"Error with index {idx}, {e} \n shape: {x.shape} \n filename: {self.file_list[idx]}"
                )
        self.readtime += time.time() - start
        return {"image": x}


class subImageRand(object):
    """
    Selects a random spatial cropping of measurement stack
    """

    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def __call__(self, sample):
        shape = sample.shape
        height, width = shape[-2], shape[-1]
        xRand = rand.randint(0, max(height - self.output_size[0], 0))
        yRand = rand.randint(0, max(width - self.output_size[1], 0))

        sample = sample[
            :,
            xRand : (xRand + self.output_size[0]),
            yRand : (yRand + self.output_size[1]),
        ]
        return sample


class RandFlip(object):
    """
    Flip stack of measurements (stacked in first dimension) randomly lr or ud
    """

    def __call__(self, sample):
        rand = np.random.randint(0, 1)
        meas_stack = []
        for i in range(sample.shape[0]):
            if rand > 0.5:
                meas_stack.append(np.flipud(sample[i]))
            else:
                meas_stack.append(np.fliplr(sample[i]))
        return np.stack(meas_stack, axis=0)


class toTensor(object):
    """
    Tensor conversion alias. Enforces float32 dtype.
    """

    def __call__(self, sample, device=None):
        if device:
            sample = torch.tensor(sample, dtype=torch.float32, device=device)
        else:
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample
