from typing import Tuple

import numpy as np
from nptyping import ndarray

from coolcnn.layers.base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, input_layer: ndarray) -> ndarray:
        return input_layer.flatten()

    def backpropagate(self, input_layer: ndarray, output_layer: ndarray, d_error_d_out: ndarray) -> ndarray:
        return d_error_d_out.reshape(input_layer.shape)

    def _get_output_shape(self) -> Tuple:
        return (np.prod(self._input_shape), )
