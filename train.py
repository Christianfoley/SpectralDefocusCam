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
    fm_params = config["forward_model_params"]
    rm_params = config["recon_model_params"]
    rm_params["num_measurements"] = fm_params["stack_depth"]
    rm_params["blur_stride"] = fm_params["blur_stride"]

    # forward model
    mask = load_mask(path=config["mask_dir"], patch_size=config["patch_size"])
    forward_model = fm.Forward_Model(
        mask,
        params=fm_params,
        cuda_device=device,
        psf_dir=config["psf_dir"],
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
    model.eval()
    val_loss = 0
    sample_np = None
    for sample in tqdm.tqdm(dataloader, desc="validating", leave=0):
        sample_np = sample["image"].numpy()[0]
        output = model(sample["image"].to(device))  # Compute the output image
        loss = loss_function(output, sample["image"].to(device))  # Compute the loss
        val_loss += loss.item()

    val_loss = val_loss / dataloader.__len__()
    test_np = output.detach().cpu().numpy()[0]

    model.train()
    return val_loss, test_np, sample_np


def generate_plot(test, gt, opt_path, fc_scaling=[0.9, 0.74, 1.12]):
    """
    Generate a plot of recon next to ground truth in false color. Also plots
    a random pixel spectral response from the gt and test
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    pred_fc = helper.stack_rgb_opt_30(
        test.transpose(1, 2, 0), scaling=fc_scaling, opt=opt_path
    )
    samp_fc = helper.stack_rgb_opt_30(
        gt.transpose(1, 2, 0), scaling=fc_scaling, opt=opt_path
    )
    pred_fc = helper.value_norm(pred_fc)
    samp_fc = helper.value_norm(samp_fc)

    ax[0].imshow(pred_fc)
    ax[0].set_title("reconstructed")

    ax[1].imshow(samp_fc)
    ax[1].set_title("ground truth")

    y = np.random.randint(0, test.shape[1])
    x = np.random.randint(0, test.shape[2])
    ax[2].plot(test[:, y, x], color="red", label="prediction")
    ax[2].plot(gt[:, y, x], color="blue", label="ground_truth")
    ax[2].set_title(f"spectral response: pixel {(y,x)}")
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
        dl_time = 0
        inf_time = 0
        prop_time = 0
        train_loss = 0
        mark = time.time()
        for sample in tqdm.tqdm(train_dataloader, desc="iters", position=1, leave=0):
            dl_time += time.time() - mark
            mark = time.time()

            # Compute the output image
            output = model(sample["image"].to(device))
            inf_time += time.time() - mark
            mark = time.time()

            # Compute the loss
            loss = loss_function(output, sample["image"].to(device))
            train_loss += loss.item()

            # Update the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Enforce a physical constraint on the blur parameters
            if model.model1.optimize_blur:
                model.model1.w_blur.data = torch.clip(
                    model.model1.w_blur.data, 0.0006, 1
                )
            prop_time += time.time() - mark
            mark = time.time()
        lr_scheduler.step()

        train_loss = train_loss / train_dataloader.__len__()
        train_loss_list.append(train_loss)

        # running validation
        if i % config["validation_stride"] == 0:
            val_loss, test_np, ground_truth_np = evaluate(
                model, val_dataloader, loss_function, device=device
            )
            val_loss_list.append(val_loss)
            print(f"\nEpoch ({i}) losses  (train, val) : ({train_loss}, {val_loss})")

            if plot:
                fig = generate_plot(
                    test_np,
                    ground_truth_np,
                    opt_path=config["false_color_mat_path"],
                )
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
            writer.add_graph(model, sample["image"].to(device), verbose=False)
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
    if config["forward_model_params"]["sim_meas"]:
        train_loader, val_loader, test_loader = ds.get_data(
            config["batch_size"],
            config["data_partition"],
            config["base_data_path"],
            config["patch_size"],
            config["num_workers"],
        )
    else:
        train_loader, val_loader, test_loader = pre_ds.get_data_precomputed(
            config["batch_size"],
            config["precomp_meas_path"],
            config["patch_size"],
            config["num_workers"],
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
