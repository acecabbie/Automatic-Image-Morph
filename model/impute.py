from abc import ABC, abstractmethod

import cv2
import numpy as np

from .consts import FLOAT_T


class Imputer(ABC):
    """Abstract base class for data imputation transformations."""

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Transform input data from original to model-compatible format."""
        pass

    @abstractmethod
    def backward(self, x: np.ndarray) -> np.ndarray:
        """Transform data from model-compatible format back to original format."""
        pass


class ImageImputer(Imputer):
    im_width: int
    im_height: int
    im_max: float
    crop: bool
    resize: bool

    def __init__(
        self,
        im_width=224,
        im_height=224,
        im_max=255.0,
        crop=True,
        resize=True,
    ):
        self.im_width = int(im_width)
        self.im_height = int(im_height)
        self.im_max = float(im_max)
        self.crop = crop
        self.resize = resize

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.crop:
            min_dim = min(x.shape[:-1])
            x = x[:min_dim, :min_dim]

        if self.resize:
            kernel_width = int(self.im_width / x.shape[1]) // 2 * 2 + 1
            if len(x.shape) in {2, 3}:
                x = cv2.GaussianBlur(x, (kernel_width, kernel_width), 0)
                x = cv2.resize(
                    x,
                    (self.im_width, self.im_height),
                    interpolation=cv2.INTER_AREA,
                )
            elif len(x.shape) == 4:
                for i in range(x.shape[0]):
                    x[i] = cv2.GaussianBlur(x[i], (kernel_width, kernel_width), 0)
                    x[i] = cv2.resize(
                        x[i],
                        (self.im_width, self.im_height),
                        interpolation=cv2.INTER_AREA,
                    )
            else:
                raise ValueError(
                    f"Invalid image shape: {x.shape}. Expected 2D, 3D, or 4D array."
                )

        return x.astype(FLOAT_T) / self.im_max

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * self.im_max


class KeypointImputer(Imputer):
    im_width: float
    im_height: float

    def __init__(self, native_im_width=512.0, native_im_height=512.0):
        self.im_width = float(native_im_width)
        self.im_height = float(native_im_height)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x / self.im_width

    def backward(self, x: np.ndarray) -> np.ndarray:
        return x * self.im_width
