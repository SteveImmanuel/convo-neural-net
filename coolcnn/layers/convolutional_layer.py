from coolcnn.layers.kernel_layer import KernelLayer
from coolcnn.activation import *

from nptyping import ndarray
from typing import Union, Tuple, List

import numpy as np


class Convolutional(KernelLayer):
    def __init__(self, activator: ActivationType = ActivationType.RELU, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__activator = activator

    def _on_feature_map_entry(self, value: float) -> float:
        return Activation.process(self.__activator, value)

    def _generate_weight(self):
        n_channel = self._input_shape[-1]
        w, h = self._kernel_shape

        total_filter_element = w * h * n_channel
        self._weights = np.array([], dtype=float)
        for i in range(self._n_kernel):
            random_weight = np.random.rand(total_filter_element)
            self._weights = np.concatenate((self._weights, random_weight))

        self._weights = np.reshape(self._weights, (self._n_kernel, w, h, n_channel))
        self._bias = np.random.rand(self._n_kernel)
