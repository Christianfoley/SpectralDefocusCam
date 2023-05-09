# main libraries:
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.io

import os, glob

# packages needed for making a dataset:
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.diffuser_utils import *
import utils.early_stop_utils as stopping
import utils.helper_functions as helper
import utils.optimizer_utils as optim_utils

import data_utils.dataset as ds
import models_learning.spectral_model as sm
import models_learning.forward as fm


# packages needed for training the network
from torch.nn import MSELoss
from torch.optim import Adam
from datetime import date, datetime

import sys

sys.path.append("..")

import models_learning.unet3d as Unet3d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# --- hyperparameters ---#
CHECKPOINTS_DIR = "saved_models/"
PSF_DIR = "../defocuscamdata/calibration_data/psfs2"

DEVICE = "cpu"  # cuda:2"
OPTIMIZE_BLUR = False
BLUR_TYPE = "symmetric"
STACK_DEPTH = 3
EPOCHS = 500
OFFSET = 0
SIM_BLUR = False


def get_data():
    fruitset_pca = glob.glob("../defocuscamdata/sample_data/fruitdata/pca/*.mat")
    harvard = glob.glob("../defocuscamdata/sample_data/harvard/CZ_hsdb/*.mat")
    harvard_indoor = glob.glob("../defocuscamdata/sample_data/harvard/CZ_hsdbi/*.mat")
    paviaU = glob.glob("../defocuscamdata/sample_data/paviadata/Pavia*.mat")

    # load pavia images (validation set)
    pavia_test_data = ds.SpectralDataset(
        paviaU,
        transforms.Compose(
            [
                ds.subImageRand(),
                ds.chooseSpectralBands(interp=True),
                ds.Normalize(),
                ds.toTensor(),
            ]
        ),
        tag=["paviaU", "pavia"],
    )
    # load giessen images
    fruit_train_data = ds.SpectralDataset(
        fruitset_pca,
        transforms.Compose(
            [
                ds.readCompressed(),
                ds.Resize(),
                ds.chooseSpectralBands(),
                ds.RandFlip(),
                ds.Normalize(),
                ds.toTensor(),
            ]
        ),
    )
    # load harvard images
    summer_data = ds.SpectralDataset(
        harvard_indoor[6:] + harvard[6:],
        transforms.Compose(
            [
                ds.Resize(),
                ds.chooseSpectralBands(),
                ds.RandFlip(),
                ds.Normalize(),
                ds.toTensor(),
            ]
        ),
        tag="ref",
    )
    # we will load a subset of these harvard images as validation
    val_data = ds.SpectralDataset(
        harvard_indoor[:6] + harvard[:6],
        transforms.Compose(
            [
                ds.Resize(),
                ds.chooseSpectralBands(),
                ds.RandFlip(),
                ds.Normalize(),
                ds.toTensor(),
            ]
        ),
        tag="ref",
    )
    # wrap training sets
    train_data = ds.Wrapper([summer_data, fruit_train_data])

    # make dataloaders for pytorch
    test_dataloader = DataLoader(pavia_test_data, batch_size=1, shuffle=True)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    return test_dataloader, train_dataloader, val_dataloader


def get_model(stack_depth=STACK_DEPTH, device=DEVICE, psf_dir=PSF_DIR):
    # forward model
    mask = load_mask()
    forward_model = fm.Forward_Model(
        mask,
        num_ims=stack_depth,
        blur_type=BLUR_TYPE,
        optimize_blur=OPTIMIZE_BLUR,
        simulate_blur=SIM_BLUR,
        psf_dir=psf_dir,
    )

    # reconstruction model
    recon_model = Unet3d.Unet(n_channel_in=stack_depth, n_channel_out=1)

    return sm.MyEnsemble(forward_model.to(device), recon_model.to(device))


def get_save_folder():
    # specify training checkpoints
    args_dict = {
        "version": "4",
        "number_measurements": str(STACK_DEPTH),
        "blur_type": BLUR_TYPE,
        "optimize_blur": str(OPTIMIZE_BLUR),
    }
    save_folder = os.path.join(
        CHECKPOINTS_DIR, "checkpoint_" + "_".join(list(args_dict.values())) + "/"
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    return save_folder


def evaluate(model, dataloader, loss_function, device):
    model.eval()
    test_loss = 0
    sample_np = None
    for i, sample in enumerate(dataloader):
        helper.show_progress_bar(dataloader, i, "testing")

        sample_np = sample["image"].numpy()[0]
        output = model(sample["image"].to(device))  # Compute the output image
        loss = loss_function(output, sample["image"].to(device))  # Compute the loss
        test_loss += loss.item()

    test_loss = test_loss / dataloader.__len__()
    test_np = output.detach().cpu().numpy()[0]

    model.train()
    return test_loss, test_np, sample_np


def run_training(
    model,
    epochs,
    train_dataloader,
    val_dataloader,
    loss_function,
    optimizer,
    save_folder,
    device,
):
    w_list = []
    test_loss_list = []
    train_loss_list = []

    self.early_stopper = stopping.EarlyStopping(
        path=self.save_folder,
        patience=3,
        verbose=False,
    )

    for i in range(epochs):
        print(f"Epoch: {i}")
        train_loss = 0

        for j, sample in enumerate(train_dataloader):
            # pretty printing
            helper.show_progress_bar(train_dataloader, j, nl=True)

            # Compute the output image
            output = model(sample["image"].to(device))

            # Compute the loss
            loss = loss_function(output, sample["image"].to(device))
            train_loss += loss.item()

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Enforce a physical constraint on the parameters
            if OPTIMIZE_BLUR:
                model.model1.w_blur.data = torch.clip(
                    model.model1.w_blur.data, 0.0006, 1
                )

            if j == len(train_dataloader) - 1:
                net_output_np = output.detach().cpu().numpy()[0]
                sim_meas_np = model.output1.detach().cpu().numpy()[0]

        train_loss_list.append(train_loss / train_dataloader.__len__())

        # testing
        test_loss = 0
        if i % 10 == 0:
            test_loss, test_np, ground_truth_np = evaluate(
                model, val_dataloader, loss_function, device=DEVICE
            )
            test_loss_list.append(test_loss)

            # show test recon next to ground truth
            print("\t total Test Loss (", i, "): \t", round(test_loss_list[-1], 4))
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].imshow(np.mean(test_np, 0), cmap="RdBu")
            ax[0].set_title("reconstructed")
            ax[1].imshow(np.mean(ground_truth_np, 0), cmap="RdBu")
            ax[1].set_title("ground truth")

            plt.show()

            # save model state and metrics
            torch.save(
                model.state_dict(),
                save_folder + "saved_model_"
                "ep" + str(i + OFFSET) + "_testloss_" + str(test_loss) + ".pt",
            )
            scipy.io.savemat(
                save_folder + "saved_lists.mat",
                {
                    "test_loss": test_loss_list,
                    "train_loss": train_loss_list,
                    "w_list": w_list,
                },
            )
        test_loss_list.append(test_loss)


def main():
    print("Num devices: ", torch.cuda.device_count())

    test_dataloader, train_dataloader, val_dataloader = get_data()

    model = get_model(stack_depth=STACK_DEPTH, device=DEVICE)

    loss_function = MSELoss()
    optimizer = Adam(model.parameters(), lr=0.0001)

    save_folder = get_save_folder()

    run_training(
        model=model,
        epochs=EPOCHS,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        loss_function=loss_function,
        optimizer=optimizer,
        save_folder=save_folder,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
