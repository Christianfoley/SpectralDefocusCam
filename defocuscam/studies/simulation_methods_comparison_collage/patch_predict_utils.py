import numpy as np
import math
import itertools

import torch

from utils import diffuser_utils
from models.ensemble import SSLSimulationModel

# ----------------------------------------------------#
# Utils for patchwise prediction on large images for #
# learned model prediction. On simulated data, we    #
# patchwise predict to extend the results of our     #
# best models, trained on spectral filter masks of   #
# size 256x256, to larger, higher resolution images  #
# by effectively tiling our spectral filter          #
# ----------------------------------------------------#


def stitch_patches(patches, centers):
    # Assuming all patches are of the same size
    patch_height, patch_width, channels = patches[0].shape

    # Calculate canvas size
    min_x = min([center[0] - patch_width // 2 for center in centers])
    max_x = max([center[0] + patch_width // 2 for center in centers])
    min_y = min([center[1] - patch_height // 2 for center in centers])
    max_y = max([center[1] + patch_height // 2 for center in centers])

    height = max_x - min_x
    width = max_y - min_y

    # Create an empty canvas
    canvas = np.zeros((height, width, channels))
    contributions = np.zeros((height, width, channels), dtype=np.int)

    # Stitch patches
    for patch, center in zip(patches, centers):
        x, y = center
        x_start = x - patch_width // 2 - min_x
        x_end = x_start + patch_width
        y_start = y - patch_height // 2 - min_y
        y_end = y_start + patch_height

        canvas[x_start:x_end, y_start:y_end] += patch
        contributions[x_start:x_end, y_start:y_end] += 1

    # Avoid division by zero
    contributions[contributions == 0] = 1

    # Average the overlapping sections
    stitched_image = canvas / contributions

    return stitched_image


def get_overlapping_positions(
    image_center: tuple, image_shape: tuple, patch_shape: tuple, min_overlap=30
):
    """
    Gets center positions of overlapping prediction patches in image given the minimum overlap
    of min_overlap.
    NOTE: greater image may ALSO be a "subset" cropped out of an even larger super-image

    - greater_center: tuple (y, x) of center of greater image
    - greater_shape: tuple (y, x) of greater image patch shape
    - lesser_shape: tuple (y, x) of smaller image patch shape
    """
    im_center_y, im_center_x = image_center
    im_y, im_x = image_shape
    patch_y, patch_x = patch_shape

    # Beginning with minimum possible coverage, increasing until we meet min overlap
    n_patches_y = math.ceil(im_y / patch_y)
    n_patches_x = math.ceil(im_x / patch_x)

    while (n_patches_y * patch_y) < im_y + (n_patches_y - 1) * min_overlap:
        n_patches_y += 1
    while (n_patches_x * patch_x) < im_x + (n_patches_x - 1) * min_overlap:
        n_patches_x += 1

    # Locate centers for each patch along each dimension
    y_off = im_center_y - im_y // 2 + patch_y // 2
    x_off = im_center_x - im_x // 2 + patch_x // 2
    y_step = (im_y - patch_y) / (n_patches_y - 1)
    x_step = (im_x - patch_x) / (n_patches_x - 1)

    y_centers = [int(y_off + y_step * i) for i in range(n_patches_y)]
    x_centers = [int(x_off + x_step * i) for i in range(n_patches_x)]

    return list(itertools.product(y_centers, x_centers))


def patchwise_predict_image_learned(
    model: SSLSimulationModel, image: torch.Tensor, min_overlap=64
):
    """
    Predict an entire large image using a learned model by breaking it into overlapping patches
    and predicting each patch separately, then stiching & blending the results.

    Parameters
    ----------
    model : torch.nn.Module
        The learned model to use for prediction, which should have a forward method that accepts
        the input image tensor.
    image : torch.Tensor
        The input image tensor of shape (1, 1, height, width, channels).
    min_overlap : int, optional
        The minimum overlap between patches in pixels, by default 64. Small values may lead to
        edge artifacts in the stitched prediction.

    Returns
    -------
    np.ndarray
        The predicted image as a numpy array of shape (height, width, channels).

    """
    patchy, patchx = model.model1.psfs.shape[-2:]
    patch_centers = get_overlapping_positions(
        (image.shape[-2] // 2, image.shape[-1] // 2),
        image.shape[-2:],
        (patchy, patchx),
        min_overlap=min_overlap,  # The higher this is, the less edge artifacts may show up
    )

    prediction = np.zeros(image.squeeze().shape)
    contributions_mask = np.zeros(image.shape[-2:])
    for i, (ceny, cenx) in enumerate(patch_centers):
        reg = [
            ceny - patchy // 2,
            ceny + patchy // 2,
            cenx - patchx // 2,
            cenx + patchx // 2,
        ]
        patch_gt = image[..., reg[0] : reg[1], reg[2] : reg[3]]
        sim = model.model1(patch_gt.to(model.model1.device))
        pred = model.model2((sim - sim.mean()) / sim.std()).detach().cpu().numpy()
        pred = pred * patch_gt.std().numpy() + patch_gt.mean().numpy()

        # ------------ REMOVE NON IMAGE-BORDERING PATCH EDGE ARTIFACTS ----------- #
        crop_width = pred.shape[-1] // 10  # assuming patch is square

        # Crop patch edges that are not bording an image edge
        bordering_top = ceny - patchy // 2 == 0
        bordering_bottom = ceny + patchy // 2 == image.shape[-2]
        bordering_right = cenx + patchx // 2 == image.shape[-1]
        bordering_left = cenx - patchx // 2 == 0
        if not bordering_top:
            pred, reg[0] = pred[..., crop_width:, :], reg[0] + crop_width
        if not bordering_bottom:
            pred, reg[1] = pred[..., :-crop_width, :], reg[1] - crop_width
        if not bordering_left:
            pred, reg[2] = pred[..., :, crop_width:], reg[2] + crop_width
        if not bordering_right:
            pred, reg[3] = pred[..., :, :-crop_width], reg[3] - crop_width

        # Insert the cropped patch into the prediction array
        prediction[..., reg[0] : reg[1], reg[2] : reg[3]] += pred.squeeze()
        contributions_mask[reg[0] : reg[1], reg[2] : reg[3]] += 1
    prediction = prediction / contributions_mask
    return np.maximum(0, prediction).transpose(1, 2, 0)


def prep_image(image, crop_shape, patch_shape):
    """
    Our sample images are all (H X W X C), but of different sizes.
    This helper function stantardizes these image shapes, normalizes them, and prepares them to be used as input
    to the model.
    """
    image = np.stack(
        [
            diffuser_utils.pyramid_down(
                image[: crop_shape[0], : crop_shape[1], i], patch_shape
            )
            for i in range(image.shape[-1])
        ],
        0,
    )

    image = image - max(0.0, np.min(image))
    image = image / np.max(image)
    image = torch.tensor(image)[None, None, ...]
    return image
