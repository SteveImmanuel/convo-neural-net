from typing import Tuple

import numpy as np
from nptyping import ndarray

from coolcnn.layers.base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def process(self, input_layer: ndarray) -> ndarray:
        return input_layer.flatten()

    def _validate_weight(self) -> bool:
        return True

    def _get_output_shape(self) -> Tuple:
        return (np.prod(self._input_shape), )
