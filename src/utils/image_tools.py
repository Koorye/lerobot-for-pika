import imageio.v3 as imageio
import numpy as np


def load_image(image_path):
    if isinstance(image_path, np.ndarray):
        return image_path
    return imageio.imread(image_path)