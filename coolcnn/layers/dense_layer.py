import numpy as np
from nptyping import ndarray

from coolcnn.activation import *
from coolcnn.layers.base_layer import BaseLayer
from typing import Tuple


class Dense(BaseLayer):
    def __init__(self, n_nodes=int, activator: ActivationType = ActivationType.SIGMOID, **kwargs):
        super().__init__(**kwargs)
        self.__n_nodes = n_nodes
        self.__activator = activator

    def _validate_weight(self):
        return True

    def process(self, input_layer: ndarray) -> ndarray:
        out_layer = np.array([], dtype=float)
        for weight in self._weights:
            calculated_value = np.sum(input_layer * weight) + self._bias
            out_layer = np.append(out_layer, Activation.process(self.__activator, calculated_value))

        return out_layer

    def _generate_weight(self):
        self._weights = np.array(
            [np.random.rand(self._input_shape[-1]) for _ in range(self.__n_nodes)]
        )

        self._bias = 0

    def _get_output_shape(self) -> Tuple:
        return (self.__n_nodes, )
