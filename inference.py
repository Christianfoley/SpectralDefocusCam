# main libraries:
import torch
import os, glob
import tqdm

# packages needed for making a dataset:
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.diffuser_utils import *
import utils.helper_functions as helper
import data_utils.dataset as ds
import models_learning.spectral_model as sm
import models_learning.forward as fm

import sys

sys.path.append("..")

import models_learning.unet3d as Unet3d

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# --- hyperparameters ---#
DEVICE = "cpu"  # cuda:2"
PSF_DIR = "../defocuscamdata/calibration_data/psfs2"
SIM_BLUR = True
BLUR_TYPE = "symmetric"
STACK_DEPTH = (2, 3, 5, 5, 6)
WEIGHTS = (
    "saved_models/optimal_models/saved_model_ep640_2_testloss_0.0011793196899816394.pt",
    "saved_models/optimal_models/saved_model_ep870_3_testloss_0.0004674650845117867.pt",
    "saved_models/optimal_models/saved_model_ep660_5_testloss_0.0006101074395701289.pt",
    "saved_models/optimal_models/saved_model_ep220_5_testloss_0.0010784256737679243.pt",
    "saved_models/optimal_models/saved_model_ep1130_6_testloss_0.0003215452015865594.pt",
)
EVAL_METRIC = torch.nn.MSELoss()


def get_data_inference():
    # TODO save portion of each data distribution for validation/test set
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
                ds.Normalize(),
                ds.toTensor(),
            ]
        ),
        tag="ref",
    )
    # wrap training sets
    train_data = ds.Wrapper([summer_data, fruit_train_data])

    # make dataloaders for pytorch
    test_dataloader = DataLoader(pavia_test_data, batch_size=1, shuffle=False)
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=False)
    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=False)

    return test_dataloader, train_dataloader, val_dataloader


def get_model_pretrained(
    weights, stack_depth=STACK_DEPTH, device=DEVICE, psf_dir=PSF_DIR
):
    # forward model
    mask = load_mask()
    forward_model = fm.Forward_Model(
        mask,
        num_ims=stack_depth,
        blur_type=BLUR_TYPE,
        optimize_blur=False,
        simulate_blur=SIM_BLUR,
        psf_dir=psf_dir,
        cuda_device=DEVICE,
    )

    # reconstruction model
    recon_model = Unet3d.Unet(n_channel_in=stack_depth, n_channel_out=1)
    model = sm.MyEnsemble(forward_model.to(device), recon_model.to(device))

    model.load_state_dict(torch.load(weights, map_location=torch.device(DEVICE)))
    model.eval()
    return model


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


def run_inference(model, dataloader, save_folder, metric=None):
    loss = "no-metric"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, sample in tqdm.tqdm(list(enumerate(dataloader))):
        model, sample = model.to(DEVICE), sample["image"].to(DEVICE)
        pred = model(sample)
        if metric:
            loss = metric(pred, sample)
        pred = pred.detach().cpu().numpy()[0]
        sample = sample.detach().cpu().numpy()[0]
        simulated = model.model1.sim_meas.detach().cpu().numpy()[0]
        scipy.io.savemat(
            os.path.join(save_folder, f"{i}_{loss:.5f}.mat"),
            mdict={
                "sample": sample,
                "simulated_meas": simulated,
                "prediction": pred,
            },
        )


def visualize(save_folder, fc_scaling=[0.9, 0.74, 1.12], ids=-1):
    predictions = glob.glob(os.path.join(save_folder, "*.mat"))
    if isinstance(ids, int) and ids == -1:
        ids = range(len(predictions))

    fig, ax = plt.subplots(3, len(ids), figsize=(4 * len(ids), 12), facecolor=(1, 1, 1))
    for i, idx in enumerate(ids):
        mat = scipy.io.loadmat(predictions[idx])
        pred = mat["prediction"]
        sample = mat["sample"]
        simulated = mat["simulated_meas"]

        pred_fc = helper.stack_rgb_opt_30(pred.transpose(1, 2, 0), scaling=fc_scaling)
        samp_fc = helper.stack_rgb_opt_30(sample.transpose(1, 2, 0), scaling=fc_scaling)
        pred_fc = helper.value_norm(pred_fc)
        samp_fc = helper.value_norm(samp_fc)

        ax[0][i].set_title(f"sample {idx}")
        ax[0][i].imshow(samp_fc)
        ax[1][i].imshow(simulated[0], cmap="gray")
        ax[2][i].imshow(pred_fc)

    plt.suptitle("false color projections")
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "false_color_projections.png"))
    plt.show()


def main():
    print("Num devices: ", torch.cuda.device_count())

    test_dataloader, train_dataloader, val_dataloader = get_data_inference()
    for i, weights in enumerate(WEIGHTS):
        model = get_model_pretrained(
            weights=weights, stack_depth=STACK_DEPTH[i], device=DEVICE
        )
        model_name = os.path.basename(weights)[:-3]
        save_folder = os.path.join("predictions", model_name)

        print(f"Running inference with: {model_name}")
        run_inference(
            model=model,
            dataloader=test_dataloader,
            save_folder=save_folder,
            metric=EVAL_METRIC,
        )
        visualize(save_folder, ids=list(range(3)))


if __name__ == "__main__":
    main()
