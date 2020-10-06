from enum import Enum
from typing import List, Tuple, Union

import numpy as np
from nptyping import ndarray

from coolcnn.layers.kernel_layer import KernelLayer


class PoolMode(Enum):
    MAX = 'max'
    AVG = 'avg'


class Pooling(KernelLayer):
    def __init__(self, mode: PoolMode = PoolMode.MAX, **kwargs) -> None:
        super().__init__(n_kernel=1, **kwargs)
        self.__mode = mode

    def backpropagate(self, input_layer: ndarray, output_layer: ndarray, d_error_d_out: ndarray) -> ndarray:
        d_error_d_out_prev = np.zeros(input_layer.shape)

        w_strides, h_strides = self._strides

        for receptive_field, output_row, output_col in self._gen_receptive_field(input_layer):
            anchor_left = output_col * w_strides
            anchor_top = output_row * h_strides

            for idx in range(self._n_kernel):
                pool_idx = 0
                value = 0

                channel_matrix = receptive_field[:, :, idx]
                if self.__mode == PoolMode.MAX:
                    pool_idx = np.argmax(channel_matrix)
                    pool_idx_top, pool_idx_left = np.unravel_index(pool_idx, self._kernel_shape)
                    value = d_error_d_out[output_row][output_col][idx]

                    pool_idx_top += anchor_top
                    pool_idx_left += anchor_left
                else:
                    raise RuntimeError('Backprop not implemented for average mode')

                d_error_d_out_prev[pool_idx_top][pool_idx_left][idx] += value

        return d_error_d_out_prev

    def _on_receptive_field(
        self,
        receptive_field: ndarray,
        feature_map: ndarray,
        output_row: int,
        output_col: int,
    ) -> None:
        for idx in range(self._n_kernel):
            value = 0
            channel_matrix = receptive_field[:, :, idx]
            if self.__mode == PoolMode.MAX:
                value = channel_matrix.max()
            else:
                value = channel_matrix.mean()
            feature_map[output_row][output_col][idx] = value

    def _generate_weight(self):
        self._n_kernel = self._input_shape[-1]
