import torch
import os, glob
import tqdm
import sys
from pathlib import Path

from utils.diffuser_utils import *
import utils.helper_functions as helper
import utils.metrics as metrics
import data_utils.dataset as ds
import models_learning.ensemble as ensemble
import models_learning.forward as fm

sys.path.append("..")

import models_learning.Unet.unet3d as Unet3d


def get_model_pretrained(weights, train_config, device):
    # forward model
    mask = load_mask()
    forward_model = fm.Forward_Model(
        mask,
        num_ims=train_config["stack_depth"],
        blur_type=train_config["blur_type"],
        optimize_blur=False,
        simulate_blur=train_config["sim_blur"],
        psf_dir=train_config["psf_dir"],
        cuda_device=device,
    )

    # reconstruction model
    recon_model = Unet3d.Unet(n_channel_in=train_config["stack_depth"], n_channel_out=1)
    model = ensemble.MyEnsemble(forward_model.to(device), recon_model.to(device))

    model.load_state_dict(torch.load(weights, map_location=torch.device(device)))
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


def run_inference(model, dataloader, save_folder, device, metric=None):
    score = "no-metric"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    for i, sample in tqdm.tqdm(list(enumerate(dataloader))):
        model, sample = model.to(device), sample["image"].to(device)
        pred = model(sample)

        pred = pred.detach().cpu().numpy()[0]
        sample = sample.detach().cpu().numpy()[0]
        simulated = model.model1.sim_meas.detach().cpu().numpy()[0]
        if metric:
            score = metrics.get_score(metric, pred, sample)

        scipy.io.savemat(
            os.path.join(save_folder, f"{i}_{metric}{score:.5f}.mat"),
            mdict={
                "sample": sample,
                "simulated_meas": simulated,
                "prediction": pred,
                "metric": metric,
                "score": score,
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


def main(config):
    # get device
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    print("Num devices: ", torch.cuda.device_count())
    device = helper.get_device(config["device"])
    print("Trying device: ", torch.cuda.get_device_properties(device).name)
    try:
        device = torch.device(device)
    except Exception as e:
        print(f"Failed to select device {device}: {e}")
        print("Running on CPU")
        device = "cpu"

    # read from the params of each weights dir
    for i, weights in enumerate(config["weights"]):
        train_dir = Path(weights).parent.absolute()
        train_config = helper.read_config(
            os.path.join(train_dir, "training_config.yml")
        )
        model_name = os.path.basename(weights)[:-3]

        # get model
        model = get_model_pretrained(
            weights=weights, train_config=train_config, device=device
        )

        # get data
        _, _, test_loader = ds.get_data(
            batch_size=1,
            data_split=train_config["data_partition"],
            base_path=train_config["base_data_path"],
            workers=config["num_workers"],
        )

        # run inference
        print(f"Running inference with: {model_name}")
        save_folder = os.path.join(train_dir, "predictions/")
        run_inference(
            model=model,
            dataloader=test_loader,
            save_folder=save_folder,
            device=device,
            metric=config["prelim_metric"],
        )
        visualize(save_folder, ids=list(range(3)))
