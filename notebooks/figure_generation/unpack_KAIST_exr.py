# %%
import OpenEXR as exr
import Imath
import matplotlib.pyplot as plt
import numpy as np

import sys

sys.path.insert(0, "../..")

import utils.helper_functions as helper


def exr_get_channels(exrfile):
    header = exrfile.header()
    all_channels = header["channels"]
    spectral_channels = []
    for key in all_channels.keys():
        if key not in ["R", "G", "B"]:
            spectral_channels.append(key)
    # sort the spectral channels
    spectral_channels.sort()
    return spectral_channels


def exr_to_array(exrfile, channel="R"):
    raw_bytes = exrfile.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
    depth_vector = np.frombuffer(raw_bytes, dtype=np.float32)
    height = (
        exrfile.header()["displayWindow"].max.y
        + 1
        - exrfile.header()["displayWindow"].min.y
    )
    width = (
        exrfile.header()["displayWindow"].max.x
        + 1
        - exrfile.header()["displayWindow"].min.x
    )
    depth_map = np.reshape(depth_vector, (height, width))
    return depth_map


def gen_rgb_image(filepath):
    exrfile = exr.InputFile(filepath)
    channels = ["R", "G", "B"]
    image = []
    for channel in channels:
        image.append(exr_to_array(exrfile, channel))
    image = np.stack(image, axis=-1)
    return image


def gen_spectral_image(filepath):
    exrfile = exr.InputFile(filepath)
    channels = exr_get_channels(exrfile)
    image = []
    for channel in channels:
        image.append(exr_to_array(exrfile, channel))
    image = np.stack(image, axis=-1)
    wavelengths = [float(channel.split("w")[-1].split("n")[0]) for channel in channels]
    return image, wavelengths


# %%
# Path to your .exr file
exr_file_path = (
    "/home/cfoley/defocuscamdata/sample_data/kaistdata/scene03_reflectance.exr"
)
# %%
# Generate an RGB image from the .exr file
# rgb_image = gen_rgb_image(exr_file_path)

# Plot the RGB image
plt.figure(figsize=(5, 5))
plt.imshow(rgb_image)
plt.axis("off")
plt.title("RGB Image")
plt.show()

# Print the wavelengths of the spectral image
print("Wavelengths:", wavelengths)
# %%


# Generate a spectral image from the .exr file
spectral_image, wavelengths = gen_spectral_image(exr_file_path)
# %%
import scipy.io as io

io.savemat(
    "/home/cfoley/defocuscamdata/sample_data/kaistdata/scene03_reflectance.mat",
    {"ref": spectral_image, "wavs": wavelengths},
)
# %%
plt.imshow(helper.select_and_average_bands(spectral_image))
# %%
print(wavelengths)
# %%
