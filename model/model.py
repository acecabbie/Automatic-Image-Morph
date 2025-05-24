from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .impute import ImageImputer, KeypointImputer


class KeypointModel:
    def __init__(
        self,
        model_file: Path | str,
        image_imputer: Optional[ImageImputer] = None,
        keypoint_imputer: Optional[KeypointImputer] = None,
    ):
        self.model_file = model_file
        self.model: tf.keras.Model = tf.keras.models.load_model(str(self.model_file))
        self.image_imputer: ImageImputer = image_imputer or ImageImputer()
        self.keypoint_imputer: KeypointImputer = keypoint_imputer or KeypointImputer()

    def predict(self, im: np.ndarray, show: bool = False) -> np.ndarray:
        im = im.reshape(
            -1, self.image_imputer.im_width, self.image_imputer.im_height, 3
        )
        pred_raw = self.model.predict(im, verbose=0)
        pred = self.keypoint_imputer.backward(pred_raw).reshape(-1, 2, 68)

        if show and im.shape[0] == 1:
            h_scale = self.image_imputer.im_width / self.keypoint_imputer.im_width
            w_scale = self.image_imputer.im_height / self.keypoint_imputer.im_height

            plt.figure()
            plt.imshow(im[0])
            for i, (x, y) in enumerate(zip(pred[0][0], pred[0][1])):
                plt.scatter(x * h_scale, y * w_scale, c="r", s=10)
                plt.text(x * h_scale, y * w_scale, str(i), color="white", fontsize=8)
            plt.axis("off")
            plt.show()

        return pred
