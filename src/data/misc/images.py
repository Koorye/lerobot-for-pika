import imageio.v3 as imageio


def load_image(path):
    """
    Load an image from the given path.
    
    Args:
        path (str): The file path to the image.
    
    Returns:
        numpy.ndarray: The loaded image as a NumPy array.
    """
    return imageio.imread(path)