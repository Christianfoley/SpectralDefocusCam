import numpy as np
import time
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import scipy.io
import os

from torch.utils.tensorboard import SummaryWriter

from utils.diffuser_utils import *
import utils.early_stop_utils as stopping
import utils.helper_functions as helper
import utils.optimizer_utils as optim_utils

import data_utils.dataset as ds
import data_utils.precomp_dataset as pre_ds
import models.ensemble as ensemble
import models.forward as fm
import sys

sys.path.append("..")

import models.Unet.unet3d as Unet3d
import models.Unet.R2attunet as R2attunet3d
import models.LCNF.liif as liif
import models.fista.fista_spectral_cupy_batch as fista

# don't delete: registering
import models.LCNF.edsr
import models.LCNF.mlp


def get_model(config, device):
    """
    Constructs a model from the given forward and recon model params. If data_precomputed
    is true, forward model will be "passthrough".

    Parameters
    ----------
    config : dict
        config dictionary with model hyperparams
    device : torch.Device
        device to place models on

    Returns
    -------
    torch.nn.Module
        "ensemble" wrapper model for forward and recon models
    """
    fm_params = config["forward_model_params"]
    rm_params = config["recon_model_params"]
    rm_params["num_measurements"] = fm_params["stack_depth"]
    rm_params["blur_stride"] = fm_params["psf"]["stride"]

    # forward model
    mask = load_mask(
        path=config["mask_dir"],
        patch_crop_center=config["image_center"],
        patch_crop_size=config["patch_crop"],
        patch_size=config["patch_size"],
    )
    forward_model = fm.ForwardModel(
        mask,
        params=fm_params,
        psf_dir=config["psf_dir"],
        passthrough=config["data_precomputed"],
        device=device,
    )
    forward_model.init_psfs()

    # recon model
    if rm_params["model_name"] == "fista":
        recon_model = fista.fista_spectral_numpy(
            forward_model.psfs, torch.tensor(mask), params=rm_params, device=device
        )
    elif rm_params["model_name"] == "unet":
        recon_model = Unet3d.Unet(n_channel_in=rm_params["num_measurements"])
    elif rm_params["model_name"] == "r2attunet":
        recon_model = R2attunet3d.R2AttUnet(
            in_ch=rm_params["num_measurements"],
            t=rm_params.get("recurrence_t", 2),
        )
    elif rm_params["model_name"] == "lcnf":
        encoder_specs = [rm_params["encoder_specs"]] * rm_params["num_measurements"]
        recon_model = liif.LIIF(
            encoder_specs,
            rm_params["imnet_spec"],
            rm_params["enhancements"],
        )

    # build ensemble and load any pretrained weights
    full_model = ensemble.MyEnsemble(forward_model, recon_model)

    if config.get("preload_weights", False):
        full_model.load_state_dict(
            torch.load(config["checkpoint_dir"], map_location="cpu")
        )

    return full_model.to(device)


def get_save_folder(config):
    """
    Helper function for creating a save folder timestamped with beginning
    of model training.

    Parameters
    ----------
    config : dict
        training config

    Returns
    -------
    str
        path to save folder
    """
    # specify training checkpoints
    config_filename = os.path.basename(config["config_fname"])
    save_folder = os.path.join(
        config["checkpoints_dir"],
        f"checkpoint_{config_filename}",
        helper.get_now(),
    )

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    return save_folder


def evaluate(model, dataloader, loss_function, device):
    """
    Evaluation (validation) procedure for model.

    Parameters
    ----------
    model : torch.nn.Module
        model to run validation with
    dataloader : torch.data.utils.Dataloader
        validation dataloader
    loss_function : fn
        loss function
    device : torch.Device
        device

    Returns
    -------
    tuple(float, np.ndarray, np.ndarray)
        tuple of the validation set loss, and an example input and prediction
    """
    model.eval()
    val_loss = 0
    for sample in tqdm.tqdm(dataloader, desc="validating", leave=0):
        output = model(sample["input"].to(device))
        loss = loss_function(output, sample["image"].to(device))
        val_loss += loss.item()

    val_loss = val_loss / dataloader.__len__()

    gt_np = sample["image"].detach().cpu().numpy()[0]
    in_np = sample["input"].detach().cpu().numpy()[0]
    recon_np = output.detach().cpu().numpy()[0]

    model.train()
    return val_loss, in_np, recon_np, gt_np


def generate_plot(model_input, recon, gt):
    """
    Visualization utility for tensorboard logging. Generates a plot of recon next
    to ground truth in false color. Also plots a random pixel spectral response
    from the gt and recon.

    Parameters
    ----------
    recon : np.ndarray
        reconstructed hyperspectral image (model output) (c, y, x)
    gt : np.ndarray
        ground truth hyperspectral data (model input) (c, y, x)

    Returns
    -------
    plt.Figure
        matplotlib figure to log to tensorboard
    """
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.set_dpi(70)

    # strip channel and batch dimensions (select first)
    while len(model_input.shape) > 3:
        model_input = model_input[0]
    while len(recon.shape) > 3:
        recon = recon[0]
    while len(gt.shape) > 3:
        gt = gt[0]

    input_fc = helper.stack_rgb_opt_30(model_input.transpose(1, 2, 0))
    pred_fc = helper.stack_rgb_opt_30(recon.transpose(1, 2, 0))
    samp_fc = helper.stack_rgb_opt_30(gt.transpose(1, 2, 0))
    input_fc = helper.value_norm(input_fc)
    pred_fc = helper.value_norm(pred_fc)
    samp_fc = helper.value_norm(samp_fc)

    ax[0].imshow(input_fc, vmax=np.percentile(input_fc, 95))
    ax[0].set_title("input")

    ax[1].imshow(pred_fc, vmax=np.percentile(pred_fc, 95))
    ax[1].set_title("reconstructed")

    ax[2].imshow(samp_fc, vmax=np.percentile(samp_fc, 95))
    ax[2].set_title("ground truth")

    y = np.random.randint(0, recon.shape[1])
    x = np.random.randint(0, recon.shape[2])
    ax[3].plot(recon[:, y, x], color="red", label="prediction")
    ax[3].plot(gt[:, y, x], color="blue", label="ground_truth")
    ax[3].set_title(f"spectral response: pixel {(y,x)}")
    plt.legend()
    plt.tight_layout()

    return fig


def run_training(
    model,
    config,
    train_dataloader,
    val_dataloader,
    loss_function,
    optimizer,
    lr_scheduler,
    save_folder,
    device,
    plot=True,
):
    """
    Training procedure for model.

    Parameters
    ----------
    model : torch.nn.Module
        model to run training on
    config : dict
        training config params
    train_dataloader : torch.data.utils.Dataloader
        dataloader for training split
    val_dataloader : torch.data.utils.Dataloader
        dataloader for validation split
    loss_function : fn
        loss function
    optimizer : torch.optim.Optimizer
        optimizer
    lr_scheduler : torch.optin.lr_scheduler
        learning rate scheduler for optimizer
    save_folder : str
        folder to save training logs and checkpoints to
    device : torch.Device
        device to run training on
    plot : bool, optional
        whether to provide plots in tensorboard logs, by default True
    """
    early_stopper = stopping.EarlyStopping(
        path=save_folder,
        patience=config["early_stopping_patience"],
        verbose=False,
    )
    writer = SummaryWriter(log_dir=save_folder)
    helper.write_yaml(config, os.path.join(save_folder, "training_config.yml"))

    print(f"Readtime: {train_dataloader.dataset.readtime}")
    w_list = []
    val_loss_list = []
    train_loss_list = []
    logged_graph = False
    for i in tqdm.tqdm(range(config["epochs"]), desc="Epochs", position=0):
        dl_time, inf_time, prop_time, train_loss = 0, 0, 0, 0
        mark = time.time()
        for sample in tqdm.tqdm(train_dataloader, desc="iters", position=1, leave=0):
            dl_time += time.time() - mark
            mark = time.time()

            y, x = sample["image"], sample["input"]

            # Compute the output image
            output = model(x.to(device))
            inf_time += time.time() - mark
            mark = time.time()

            # Compute the loss
            loss = loss_function(output, y.to(device))
            train_loss += loss.item()

            # Update the model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Enforce a physical constraint on the blur parameters
            if model.model1.psf["optimize"]:
                model.model1.w_blur.data = torch.clip(
                    model.model1.w_blur.data, 0.0006, 1
                )
            prop_time += time.time() - mark
            mark = time.time()
            del y
            del x

        lr_scheduler.step()

        train_loss = train_loss / train_dataloader.__len__()
        train_loss_list.append(train_loss)

        # running validation
        if i % config["validation_stride"] == 0:
            val_loss, input_np, recon_np, ground_truth_np = evaluate(
                model, val_dataloader, loss_function, device=device
            )
            val_loss_list.append(val_loss)
            print(f"\nEpoch ({i}) losses  (train, val) : ({train_loss}, {val_loss})")

            if plot:
                fig = generate_plot(input_np, recon_np, ground_truth_np)
                writer.add_figure(f"epoch_{i}_fig", fig)
            early_stopper(val_loss=val_loss, model=model, epoch=i)
        else:
            print(f"\nEpoch ({i}) losses  (train) : ({train_loss})")

        # save model state
        if i % config["checkpoint_stride"] == 0:
            torch.save(
                model.state_dict(),
                os.path.join(
                    save_folder,
                    f"saved_model_ep{str(i + config['offset'])}_testloss_{str(val_loss)}.pt",
                ),
            )
        print(f"Readtime: {train_dataloader.dataset.readtime}")
        print(f"Dataloading time: {dl_time:.2f}s, Inference time: ", end="")
        print(f"{inf_time:.2f}s, Backprop Time: {prop_time:.2f}s\n")

        # log to tensorboard
        writer.add_scalar("train loss", train_loss_list[-1], global_step=i)
        writer.add_scalar("validation loss", val_loss_list[-1], global_step=i)
        writer.add_scalar("learning rate", optim_utils.get_lr(optimizer), global_step=i)
        if not logged_graph:
            writer.add_graph(model, sample["input"].to(device), verbose=False)
            logged_graph = True

        # Getting some error when adding a histogram. Using a scalar to log for now
        if config.get("log_grads", False):
            mean_np = lambda x: np.mean(x.detach().cpu().numpy())
            for name, param in model.named_parameters():
                writer.add_scalar(f"{name} grad", mean_np(param.grad), global_step=i)
                writer.add_scalar(f"{name} weight", mean_np(param.data), global_step=i)

        # early stopping
        if early_stopper.early_stop:
            print("\t Stopping early...")

            scipy.io.savemat(
                os.path.join(save_folder, "saved_lists.mat"),
                {
                    "val_loss": val_loss_list,
                    "train_loss": train_loss_list,
                    "w_list": w_list,
                },
            )
            break

    scipy.io.savemat(
        save_folder + "saved_lists.mat",
        {
            "val_loss": val_loss_list,
            "train_loss": train_loss_list,
            "w_list": w_list,
        },
    )


def main(config):
    start = time.time()

    # setup device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    print("Num devices: ", torch.cuda.device_count())
    device = helper.get_device(config["device"])
    try:
        print("Trying device: ", torch.cuda.get_device_properties(device).name)
        device = torch.device(device)
    except Exception as e:
        print(f"Failed to select device {device}: {e}")
        print("Running on CPU")
        device = "cpu"

    # init data and model
    print("Loading data...", end="")
    if not config["data_precomputed"]:
        train_loader, val_loader, _ = ds.get_data(
            config["batch_size"],
            config["data_partition"],
            config["base_data_path"],
            config["patch_size"],
            config["num_workers"],
        )
    else:
        train_loader, val_loader, _ = pre_ds.get_data_precomputed(
            config["batch_size"],
            config["data_partition"],
            config["base_data_path"],
            config["num_workers"],
            config["forward_model_params"],
        )
    print(
        f"Done! {len(train_loader)} training samples, {len(val_loader)} validation samples"
    )

    print("Loading model...", end="")
    model = get_model(config, device=device)
    print("Done!")

    # set up optimization
    loss_function = optim_utils.get_loss_function(
        config["loss_function"]["name"], config["loss_function"]["params"]
    )
    optimizer = optim_utils.get_optimizer(
        model, config["optimizer"]["name"], config["optimizer"]["params"]
    )
    lr_scheduler = optim_utils.get_lr_scheduler(
        optimizer, config["lr_scheduler"]["name"], config["lr_scheduler"]["params"]
    )

    save_folder = get_save_folder(config)

    print("Training...")
    run_training(
        model=model,
        config=config,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        save_folder=save_folder,
        device=device,
    )
    print(f"Done! Total time: {time.time() - start:.2f} s")
