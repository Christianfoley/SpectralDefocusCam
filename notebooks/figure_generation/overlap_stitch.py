# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import itertools


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


# %%
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


# %%
if __name__ == "__main__":
    get_overlapping_positions((1200, 1970), (1260, 1860), (768, 768))

    img = np.zeros((1260, 1860))

    for i, j in get_overlapping_positions(
        (1260 // 2, 1860 // 2), (1260, 1860), (768, 768)
    ):
        img[i - 10 : i + 10, j - 10 : j + 10] = np.ones((20, 20))

    plt.imshow(img)
# %%
