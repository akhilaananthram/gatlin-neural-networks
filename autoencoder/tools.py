import math
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image # Pillow

def plot_features(black_and_white, features, filename):
    """
    Feature points are in 109x109
    Black and white is 60x60
    """
    shape = black_and_white.shape
    if len(shape) == 3 and shape[0] <= 3:
        black_and_white = np.swapaxes(black_and_white, 1, 2)
        black_and_white = np.swapaxes(black_and_white, 0, 2)
    black_and_white = black_and_white.astype(float)
    features = (features * len(black_and_white) / 109).astype(int)
    implot = plt.imshow(black_and_white)
    # Separate x and y
    plt.scatter(features[::2], features[1::2])
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, frameon=False)

def plot_reconstruction(reconstruction, filename):
    MM = len(reconstruction)
    reconstruction = np.reshape(reconstruction, (math.sqrt(MM), math.sqrt(MM)))
    img = Image.fromarray(reconstruction)
    img = img.convert('RGB')
    img.save(filename)
