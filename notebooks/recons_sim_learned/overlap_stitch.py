# %%

import numpy as np
import matplotlib.pyplot as plt


def overlap_prediction(image, model, y, x, k):
    """
    Predicts on overlapping regions of an image and stitches them back together.

    Parameters:
    - image: Input image of shape (Y, X)
    - model: Prediction model
    - y: Height of the regions
    - x: Width of the regions
    - k: Minimum number of pixels to overlap on each edge

    Returns:
    - Predicted image of shape (Y, X)
    """

    Y, X = image.shape

    # Calculate the stride for overlapping
    stride_y = y - k
    stride_x = x - k

    # Calculate the number of regions in each dimension
    num_regions_y = (Y - y) // stride_y + 1
    num_regions_x = (X - x) // stride_x + 1

    # Initialize arrays to hold predictions and counts for averaging
    predictions = np.zeros_like(image, dtype=np.float64)
    counts = np.zeros_like(image, dtype=np.int32)

    # Iterate through each region
    for i in range(num_regions_y):
        for j in range(num_regions_x):
            # Calculate the starting and ending indices of the region
            start_y = i * stride_y
            start_x = j * stride_x
            end_y = start_y + y
            end_x = start_x + x

            # Extract the region from the input image
            region = image[start_y:end_y, start_x:end_x]

            # Make prediction on the region
            prediction = model.predict(region)

            # Update predictions and counts
            predictions[start_y:end_y, start_x:end_x] += prediction
            counts[start_y:end_y, start_x:end_x] += 1

    # Divide the accumulated predictions by the counts to average
    averaged_predictions = predictions / counts

    return averaged_predictions


# Generate synthetic image
np.random.seed(0)
image = np.random.rand(256, 256)


# Define a simple model that just returns the mean of the input
class SimpleModel:
    def predict(self, x):
        return np.mean(x)


# Function to visualize the results
def visualize(image, predicted_image):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_image, cmap="gray")
    plt.title("Predicted Image")
    plt.show()


# Test the overlap_prediction function
model = SimpleModel()
result = overlap_prediction(image, model, y=64, x=64, k=16)
visualize(image, result)
# %%
