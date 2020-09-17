from typing import List, Tuple, Union

import numpy as np
from nptyping import ndarray

from coolcnn.activation import *
from coolcnn.layers.kernel_layer import KernelLayer


class Convolutional(KernelLayer):
    def __init__(self, activator: ActivationType = ActivationType.RELU, **kwargs) -> None:
        super().__init__(**kwargs)
        self.__activator = activator

    def _on_receptive_field(
        self,
        receptive_field: ndarray,
        feature_map: ndarray,
        output_row: int,
        output_col: int,
    ) -> None:
        idx = 0
        for kernel, bias in zip(self._weights, self._bias):
            receptive_field *= kernel
            summed_receptive_field = np.sum(receptive_field)
            summed_receptive_field += bias

            feature_map[output_row][output_col][idx] = Activation.process(
                self.__activator, summed_receptive_field
            )
            idx += 1

    def _generate_weight(self):
        n_channel = self._input_shape[-1]
        row, col = self._kernel_shape

        total_filter_element = row * col * n_channel
        self._weights = np.array([], dtype=float)
        for i in range(self._n_kernel):
            random_weight = np.random.rand(total_filter_element)
            self._weights = np.concatenate((self._weights, random_weight))

        self._weights = np.reshape(self._weights, (self._n_kernel, row, col, n_channel))
        self._bias = np.random.rand(self._n_kernel)

    def _get_trainable_params(self) -> int:
        trainable_params = np.prod(self._weights.shape) + np.prod(self._bias.shape)
        return trainable_params
