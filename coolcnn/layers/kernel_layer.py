from coolcnn.layers.base_layer import BaseLayer
from abc import abstractmethod

from nptyping import ndarray
from typing import Union, Tuple, List

import numpy as np


class KernelLayer(BaseLayer):
    def __init__(
        self,
        n_kernel: int,
        kernel_shape: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: int = 0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape, kernel_shape)

        self._n_kernel = n_kernel
        self._padding = ((padding, padding), (padding, padding), (0, 0))
        self._kernel_shape = kernel_shape
        self._strides = strides
        self._bias = np.zeros(self._n_kernel)

        self._generate_weight()

    def _add_padding(self, arr: ndarray) -> ndarray:
        return np.pad(arr, self._padding)

    def process(self, input_layer: ndarray) -> ndarray:
        if input_layer.ndim == 2:
            input_layer = np.reshape(input_layer, (*input_layer.shape, 1))
        elif input_layer.ndim != 3:
            raise ValueError('Invalid input layer')

        padded_input = self._add_padding(input_layer)

        w_strides, h_strides = self._strides

        kernel_size = self._kernel_shape
        input_size = padded_input.shape

        row_size = (input_size[0] - kernel_size[0]) // h_strides + 1
        col_size = (input_size[1] - kernel_size[1]) // w_strides + 1

        feature_map = np.zeros((self._n_kernel, row_size, col_size))

        for output_row in range(row_size):
            for output_col in range(col_size):
                anchor_left = output_col * w_strides
                anchor_top = output_row * h_strides

                receptive_field = padded_input[anchor_top:anchor_top + kernel_size[0],
                                               anchor_left:anchor_left + kernel_size[1]].copy()

                self._on_receptive_field(receptive_field, feature_map, output_row, output_col)

        return feature_map

    def _on_receptive_field(
        self,
        receptive_field: ndarray,
        feature_map: ndarray,
        output_row: int,
        output_col: int,
    ) -> None:
        pass

    def _validate_weight(self) -> bool:
        return True

    @abstractmethod
    def _generate_weight(self) -> None:
        pass