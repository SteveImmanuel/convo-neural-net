from abc import abstractmethod
from typing import List, Tuple, Union

import numpy as np
from nptyping import ndarray

from coolcnn.layers.base_layer import BaseLayer


class KernelLayer(BaseLayer):
    def __init__(self, n_kernel: int, kernel_shape: Union[int, Tuple[int, int]], strides: Union[int, Tuple[int, int]] = (1, 1), padding: int = 0, **kwargs) -> None:
        super().__init__(**kwargs)
        if isinstance(strides, int):
            strides = (strides, strides)
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape, kernel_shape)

        self._n_kernel = n_kernel
        self._padding = padding
        self._kernel_shape = kernel_shape
        self._strides = strides
        self._bias = np.zeros(self._n_kernel)

    def _add_padding(self, arr: ndarray) -> ndarray:
        padding = ((self._padding, self._padding), (self._padding, self._padding), (0, 0))
        return np.pad(arr, padding)

    def process(self, input_layer: ndarray) -> ndarray:
        if input_layer.ndim == 2:
            input_layer = np.reshape(input_layer, (*input_layer.shape, 1))
        elif input_layer.ndim != 3:
            raise ValueError('Invalid input layer')

        padded_input = self._add_padding(input_layer)

        w_strides, h_strides = self._strides
        kernel_size = self._kernel_shape
        row_size, col_size, _ = self.output_shape

        feature_map = np.zeros((row_size, col_size, self._n_kernel))

        for output_row in range(row_size):
            for output_col in range(col_size):
                anchor_left = output_col * w_strides
                anchor_top = output_row * h_strides

                receptive_field = padded_input[anchor_top:anchor_top + kernel_size[0], anchor_left:anchor_left + kernel_size[1]].copy()

                self._on_receptive_field(receptive_field, feature_map, output_row, output_col)

        return feature_map

    @abstractmethod
    def _on_receptive_field(
        self,
        receptive_field: ndarray,
        feature_map: ndarray,
        output_row: int,
        output_col: int,
    ) -> None:
        pass

    def _get_output_shape(self) -> Tuple:
        kernel_size = self._kernel_shape
        w_strides, h_strides = self._strides

        row_size = (self._input_shape[0] - kernel_size[0] + 2 * self._padding) // h_strides + 1
        col_size = (self._input_shape[1] - kernel_size[1] + 2 * self._padding) // w_strides + 1
        return (row_size, col_size, self._n_kernel)
