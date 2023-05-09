# %%
import numpy as np
import pandas as pd
import scipy.io as io
import PIL.Image as Image
import os
import glob
import yaml
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# %%
# measurements_dir = "/Users/christian.foley/Workspaces/spectral_calibration_basler_3_23"
# save_dir = "/Users/christian.foley/Workspaces/spectral_calibration_basler_3_23/"
measurements_dir = "../defocuscamdata/calibration_data/spectral_calibration_basler_4_4"
histograms_dir = os.path.join(measurements_dir, "exposure_histograms")
save_dir = "../defocuscamdata/calibration_data/spectral_calibration_basler_4_4/"
save_name = "cropped_calibration_matrix.mat"
filepaths = glob.glob(os.path.join(measurements_dir, "*.png"))
histogram_filepaths = glob.glob(os.path.join(histograms_dir, "*.csv"))

RGB = False  # whether images are rgb or not
filepaths.sort()
histogram_filepaths.sort()


def images_to_video(image_stack, output_path, fps=30):
    # Get image dimensions
    height, width = image_stack.shape[0], image_stack.shape[1]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Write images to video
    for i in range(image_stack.shape[-1]):
        out.write(image_stack[:, :, i])

    # Release video writer
    out.release()


# %%
# ------------------ compile calibration matrix & luminance ------------------ #
calibration_matrix = []
for i, path in enumerate(filepaths):
    band_num = f"{path.split('/')[-1][:-4]}"
    print(band_num)
    image = np.array(Image.open(path))

    # sum over color channel and normalize
    if RGB:
        image = np.sum(image, axis=-1)
        image = (
            255 * (image - np.min(image)) / (np.max(image) - np.min(image))
        ).astype(np.uint8)
    calibration_matrix.append(image)

    # # histogram
    # luminance = pd.read_csv(histogram_filepaths[i]).iloc[:, 0].to_list()
    # luminance = np.array([int(l.split(";")[-1]) for l in luminance])

    # # plotting
    # fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    # ax[0].imshow(image, cmap="gray")
    # ax[1].plot(np.arange(0, 256, 1), luminance)
    # ax[1].set_xlabel("grey value")
    # ax[1].set_ylabel("count value")
    # plt.suptitle(band_num)

    # save = os.path.join(save_dir, "plt figures")
    # if not os.path.exists(save):
    #     os.makedirs(save)
    # plt.savefig(os.path.join(save, band_num + ".png"), format="png")

calibration_matrix = np.stack(calibration_matrix, axis=-1)

# %%
# ----------- visualization scan ------------#
plt_paths = glob.glob(os.path.join(measurements_dir, "plt figures", "*.png"))
plt_paths.sort()
plt_img_stack = np.stack(
    [np.array(Image.open(path).convert("L")) for path in plt_paths], axis=-1
)
print(plt_img_stack.shape)
# %%
images_to_video(
    plt_img_stack, "/Users/christian.foley/Workspaces/spectral_matrix_scan.mp4", fps=10
)


# %%
# ------------------ save full matrix ------------------#
with open(os.path.join(measurements_dir, "parameters.yml"), "r") as f:
    parameters = yaml.safe_load(f)

full_parameters = parameters.copy()
full_parameters["calibration_matrix"] = calibration_matrix
full_parameters["dtype"] = str(calibration_matrix.dtype)

io.savemat(os.path.join(save_dir, "full_" + save_name), full_parameters)

# %%
# ----------- visualization scan ------------#
images_to_video(
    calibration_matrix, "/Users/christian.foley/Workspaces/spectral_matrix_scan.mp4"
)

# %%
# ----------- crop to usable filter array shape ------------#
usable_region = ((1850, 500), 1250, 700)

# Make sure the usable region is what u want
fig, ax = plt.subplots(1, 1)
ax.imshow(calibration_matrix[:, :, 43])
rect = patches.Rectangle(*usable_region, linewidth=1, edgecolor="r", facecolor="none")
ax.add_patch(rect)
plt.show()

# crop to specified region
cols = usable_region[0][0], usable_region[0][0] + usable_region[1]
rows = usable_region[0][1], usable_region[0][1] + usable_region[2]
cropped_calibration_matrix = calibration_matrix[rows[0] : rows[1], cols[0] : cols[1]]
plt.imshow(cropped_calibration_matrix[:, :, 43])
plt.show()
# %%
# ------------------ save cropped segment and indices ------------------#
cropped_parameters = parameters.copy()
cropped_parameters["calibration_matrix"] = cropped_calibration_matrix
cropped_parameters["dtype"] = str(cropped_calibration_matrix.dtype)
cropped_parameters["crop"] = usable_region

io.savemat(os.path.join(save_dir, "cropped_" + save_name), parameters)

# %%
# ----------- visualization scan ------------#
images_to_video(
    cropped_calibration_matrix,
    "/Users/christian.foley/Workspaces/cropped_spectral_matrix_scan.mp4",
)

# %%
