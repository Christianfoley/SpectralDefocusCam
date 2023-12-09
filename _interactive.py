# %%
import os, sys, glob
import tqdm
import matplotlib.pyplot as plt

import numpy as np
from scipy.ndimage import rotate

sys.path.insert(0, "/home/cfoley_waller/defocam/SpectralDefocusCam")

import utils.psf_calibration_utils as psf_utils
import utils.helper_functions as helper
import utils.diffuser_utils as diffuser_utils
from models.rdmpy._src.util import getCircList

# %%
############### Hyperparameters ################
psf_path = "../defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/10_26/multipos/alignment_psfs_telecent25um_10_26"
crop = (400, 0, 1500, 1000)
focus_levels = 3
patch_size = (768, 768)
alignment_estimate = (880, 421)

coord_method = "conv"
# if coord_method == conv
ksizes = [7, 21, 45]
min_distance = 15
threshold = 0.7

# %%
############### Reading and locating the psfs ################
psfs = psf_utils.read_psfs(psf_path, crop=crop, patchsize=patch_size)
supimp_psfs = psf_utils.superimpose_psfs(psfs, focus_levels=focus_levels, one_norm=True)
# %%
if coord_method == "conv":
    coords = psf_utils.get_psf_coords(
        psfs,
        focus_levels,
        coord_method,
        ksizes,
        threshold,
    )
else:
    coords = psf_utils.get_psf_coords(
        supimp_psfs,
        focus_levels,
        coord_method,
        threshold=threshold,
        min_distance=min_distance,
    )
# %%
for psf_set_num in range(0, 16):
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    fig.set_dpi(60)
    cropim = lambda im, coords, size: im[
        coords[0] - size // 2 : coords[0] + size // 2,
        coords[1] - size // 2 : coords[1] + size // 2,
    ]
    for i in range(3):
        ax[i].imshow(cropim(psfs[psf_set_num * 3 + i], coords[i][psf_set_num], 64))

    plt.show()
# %%
############### Getting radial image subdivisions ################

cropped_align_est = (alignment_estimate[0] - crop[0], alignment_estimate[1] - crop[1])
maxval = diffuser_utils.get_radius(patch_size[0] // 2, patch_size[1] // 2)
subdivisions, radii = psf_utils.radial_subdivide(
    coords, cropped_align_est, maxval, True
)


# %%
############### Plotting subdivisions for each focus level ################

fig, ax = plt.subplots(1, focus_levels, figsize=(8 * focus_levels, 5))
fig.set_dpi(70)
for i in range(focus_levels):
    ax[i].scatter(radii[i], range(len(coords[i])))
    ax[i].set_xlabel("radius")
    ax[i].set_ylabel("order")
    ax[i].set_title(f"focus {i}")

    ax[i].vlines(
        subdivisions[i],
        ymin=0,
        ymax=len(subdivisions[i]),
        colors=["red"] * len(subdivisions),
    )
plt.suptitle(
    "PSF distances from alignment axis and chosen radial subdivisions", fontsize=14
)
plt.show()
# %%
############### Getting rotated psf rings ################
level = 2
rotated_psfs = psf_utils.view_patched_psf_rings(
    psfs[level::focus_levels], coords[level], dim=768
)

plt.imshow(rotated_psfs, cmap="inferno", interpolation="none")
plt.show()

# %%
############### Hyperparameters ################
import utils.psf_calibration_utils as psf_utils
import torch

psf_path = "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/11_21/multipos/alignment_psfs_cross_5_blur"
blur_levels = 5
crop_size = 256
dim = 256
alignment_estimate = (1000, 472)

coord_method = "conv"
ksizes = [7, 21, 45, 55, 65]
exposures = [1 / 662, 1 / 110, 1 / 45, 1 / 30, 1 / 21]
min_distance = 25
threshold = 0.7
psf_dim = 120
polar = True
device = torch.device("cuda:0")
torch.cuda.get_device_name(device)
# %%
psf_data = psf_utils.get_lri_psfs(
    psf_path,
    blur_levels,
    crop_size,
    dim,
    alignment_estimate,
    coord_method,
    ksizes,
    exposures,
    min_distance,
    threshold,
    psf_dim,
    polar=polar,
    device=device,
    verbose=True,
)

# %%
print(psf_data.shape)
import matplotlib.pyplot as plt
import numpy as np

if polar:
    plt.imshow(np.mean(np.real(psf_data[0, ...]), axis=1))
else:
    plt.imshow(psf_data[4, 383, :, :] + psf_data[0, 100, :, :])
plt.colorbar()
plt.show()
# %%
############## Turn mask into video ################
import imageio
import subprocess
from scipy import io
from PIL import Image, ImageDraw, ImageFont


def frames_to_video_with_numbers(frames, output_path, frame_rate, waves):
    """
    Converts a 3D matrix of frames into an MP4 video using FFmpeg,
    with frame numbers on the top left corner of each frame.

    :param frames: 3D NumPy array of frames (x, y, time)
    :param output_path: Path to save the MP4 video
    :param frame_rate: Frame rate of the output video
    """
    # Create a temporary directory to store images
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)

    # Extract frames, add numbers, and save as images
    for i in range(frames.shape[2]):
        # Convert the numpy array to an Image object
        frame_image = Image.fromarray(frames[:, :, i])
        draw = ImageDraw.Draw(frame_image)

        # Optionally set font size and type here
        # font = ImageFont.truetype("arial.ttf", size=15)  # Example
        font = ImageFont.truetype("arial.ttf", size=48)

        # Position for the text: top-left corner
        text_position = (10, 10)

        # Draw the frame number
        draw.text(text_position, f"{waves[i]} nm", fill="white", font=font)

        # Save the frame with the number
        frame_image.save(f"{temp_dir}/frame_{i:04d}.png")

    # Use FFmpeg to convert images to video
    ffmpeg_cmd = [
        "ffmpeg",
        "-r",
        str(frame_rate),  # Frame rate
        "-i",
        f"{temp_dir}/frame_%04d.png",  # Input format
        "-vcodec",
        "libx264",  # Video codec
        "-crf",
        "25",  # Constant Rate Factor (quality)
        "-pix_fmt",
        "yuv420p",  # Pixel format
        output_path,
    ]
    subprocess.run(ffmpeg_cmd)

    # Clean up temporary images
    for filename in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, filename))
    os.rmdir(temp_dir)


# %%
mask = io.loadmat(
    "/home/cfoley_waller/defocam/defocuscamdata/calibration_data/DMM_37UX178_ML_calib_data/calib_matrix_11_10_2023_2_preprocessed/calibration_matrix_390-870_60chan_stride8_avg8.mat"
)

# %%
import numpy as np

mask = mask["mask"][1000 - 384 : 1000 + 384, 2023 - 512 : 2023 + 512, :]
mask = (((mask - np.min(mask)) / np.max(mask - np.min(mask))) * 255).astype(np.uint8)
# %%
frames_to_video_with_numbers(
    mask,
    output_path="/home/cfoley_waller/defocam/calibration_matrix_390-870_60chan_stride8_avg8.mp4",
    frame_rate=10,
    waves=list(range(390, 870, 8)),
)
# %%
