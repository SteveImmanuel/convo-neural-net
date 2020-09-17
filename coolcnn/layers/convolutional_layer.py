from coolcnn.layers.base_layer import BaseLayer
from nptyping import ndarray
from typing import Union, Tuple, List

import numpy as np


class Convolutional(BaseLayer):
    def __init__(
        self,
        n_kernel: int,
        kernel_shape: Union[int, Tuple[int, int]],
        strides: Union[int, Tuple[int, int]] = (1, 1),
        padding: int = 0,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.__n_kernel = n_kernel
        self.__padding = ((padding, padding), (padding, padding), (0, 0))
        if isinstance(kernel_shape, int):
            kernel_shape = (kernel_shape, kernel_shape)
        self.__kernel_shape = kernel_shape
        if isinstance(strides, int):
            strides = (strides, strides)
        self.__strides = strides

        self.generate_weight()

    def _add_padding(self, arr: ndarray) -> ndarray:
        return np.pad(arr, self.__padding)

    def process(self, input_layer: ndarray) -> ndarray:
        if input_layer.ndim == 2:
            input_layer = np.reshape(input_layer, (*input_layer.shape, 1))
        elif input_layer.ndim != 3:
            raise ValueError('Invalid input layer')

        padded_input = self._add_padding(input_layer)

        w_strides, h_strides = self.__strides

        kernel_size = self.__kernel_shape
        input_size = padded_input.shape

        row_size = (input_size[0] - kernel_size[0]) // h_strides + 1
        col_size = (input_size[1] - kernel_size[1]) // w_strides + 1

        feature_map = np.zeros((self.__n_kernel, row_size, col_size))

        for output_row in range(row_size):
            for output_col in range(col_size):
                anchor_left = output_col * w_strides
                anchor_top = output_row * h_strides

                receptive_field = padded_input[anchor_top:anchor_top + kernel_size[0],
                                               anchor_left:anchor_left + kernel_size[1]].copy()

                idx = 0
                for kernel, bias in zip(self._weights, self.__bias):
                    receptive_field *= kernel
                    summed_receptive_field = np.sum(receptive_field)
                    summed_receptive_field += bias

                    feature_map[idx][output_row][output_col] = summed_receptive_field
        return feature_map

    def _validate_weight(self):
        return True

    def generate_weight(self):
        n_channel = self._input_shape[-1]
        w, h = self.__kernel_shape

        total_filter_element = w * h * n_channel
        self._weights = np.array([], dtype=float)
        for i in range(self.__n_kernel):
            random_weight = np.random.rand(total_filter_element)
            self._weights = np.concatenate((self._weights, random_weight))

        self._weights = np.reshape(self._weights, (self.__n_kernel, w, h, n_channel))
        self.__bias = np.random.rand(self.__n_kernel)
