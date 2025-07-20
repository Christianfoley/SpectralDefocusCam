import numpy as np
import time
import torch
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import scipy.io
import os

import sys

sys.path.append("..")

from torch.utils.tensorboard import SummaryWriter

from utils.diffuser_utils import *
import utils.early_stop_utils as stopping
import utils.helper_functions as helper
import utils.optimizer_utils as optim_utils

import dataset.dataset as ds
import dataset.precomp_dataset as pre_ds
from models.get_model import get_model


# don't delete: registering
import models.LCNF.edsr
import models.LCNF.mlp


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
    torch.cuda.empty_cache()

    val_loss = 0
    with torch.no_grad():
        for sample in tqdm.tqdm(dataloader, desc="validating", leave=0):
            output = model(sample["input"].to(device))
            loss = loss_function(output, sample["image"].to(device))
            val_loss += loss.item()

            if isinstance(output, torch.Tensor):
                trace = output.detach().cpu().numpy()
            else:
                trace = output[0].detach().cpu().numpy()
            del output

    val_loss = val_loss / dataloader.__len__()

    gt_np = sample["image"].detach().cpu().numpy()[0]
    in_np = sample["input"].detach().cpu().numpy()[0]

    model.train()
    return val_loss, in_np, trace, gt_np


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

    if model_input.shape[0] > 1:
        input_fc = helper.stack_rgb_opt_30(model_input.transpose(1, 2, 0))
        input_fc = helper.value_norm(input_fc)
    else:
        input_fc = model_input[0]

    pred_fc = helper.stack_rgb_opt_30(recon.transpose(1, 2, 0))
    pred_fc = helper.value_norm(pred_fc)

    samp_fc = helper.stack_rgb_opt_30(gt.transpose(1, 2, 0))
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
    accum_steps = config.get("grad_accumulate", 1)

    print(f"Readtime: {train_dataloader.dataset.readtime}")
    total_updates = 0
    w_list = []
    val_loss_list = []
    train_loss_list = []
    logged_graph = config.get("no_graph", False)
    for i in tqdm.tqdm(
        range(config["epochs"] - config.get("offset", 0)), desc="Epochs", position=0
    ):
        dl_time, inf_time, prop_time, train_loss = 0, 0, 0, 0
        mark = time.time()
        idx = 0
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
            if ((idx + 1) % accum_steps == 0) or (idx + 1 == len(train_dataloader)):
                optimizer.step()
                optimizer.zero_grad()
                total_updates += 1

            # Enforce a physical constraint on the blur parameters
            if model.model1.psf["optimize"]:
                model.model1.w_blur.data = torch.clip(
                    model.model1.w_blur.data, 0.0006, 1
                )
            prop_time += time.time() - mark
            mark = time.time()
            del y
            del x
            del output
            idx += 1

            # running mid-epoch validation
            # if idx % 500 == 0:
            #     val_loss, input_np, recon_np, ground_truth_np = evaluate(
            #         model, val_dataloader, loss_function, device=device
            #     )
            #     val_loss_list.append(val_loss)
            #     train_loss = train_loss / idx
            #     writer.add_scalar("train loss", val_loss, global_step=total_updates)
            #     writer.add_scalar(
            #         "validation loss", val_loss_list[-1], global_step=total_updates
            #     )
            #     writer.add_scalar(
            #         "learning rate",
            #         optim_utils.get_lr(optimizer),
            #         global_step=total_updates,
            #     )
            #     print(
            #         f"\nEpoch ({i} {idx}) losses  (train, val) : ({train_loss}, {val_loss})"
            #     )

            #     if plot:
            #         fig = generate_plot(input_np, recon_np, ground_truth_np)
            #         writer.add_figure(f"epoch_{i}_{idx}_fig", fig)

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
        writer.add_scalar("train loss", train_loss_list[-1], global_step=total_updates)
        writer.add_scalar(
            "validation loss", val_loss_list[-1], global_step=total_updates
        )
        writer.add_scalar(
            "learning rate", optim_utils.get_lr(optimizer), global_step=total_updates
        )
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
