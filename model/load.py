from pathlib import Path
from typing import List

import cv2
import numpy as np
from tqdm import tqdm

from .consts import FLOAT_T
from .impute import ImageImputer


def load_image(img_path: str | Path, width=224, height=224):
    """Load a single image from a Path or a string.

    The image is cropped to be square and is resized to the specified width and
    height.

    Args:
        img_path (str | Path): Path to the image file.
        width (int, optional): Desired width of the image. Defaults to 224.
        height (int, optional): Desired height of the image. Defaults to 224.

    Returns:
        np.ndarray: The loaded and processed image.
    """

    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    imputer = ImageImputer(im_width=width, im_height=height)
    return imputer.forward(img)


def load_images(img_paths: List[str | Path], width=224, height=224, channels=3):
    """
    Loads multiple images from file paths and converts them to a numpy array.
    This function takes a list of image file paths, loads each image, and converts them
    to a standardized size. The resulting images are stored in a numpy array.

    Args:
        img_paths (List[str | Path]): List of file paths to images to be loaded
        width (int, optional): Width to resize images to. Defaults to 224.
        height (int, optional): Height to resize images to. Defaults to 224.
        channels (int, optional): Number of color channels. Defaults to 3.

    Returns:
        np.ndarray: A numpy array of shape (len(img_paths), width, height, channels) containing the loaded images
    """

    images = np.zeros((len(img_paths), width, height, channels), dtype=FLOAT_T)
    for i, img_path in enumerate(tqdm(img_paths)):
        images[i] = load_image(img_path, width, height)
    return images
