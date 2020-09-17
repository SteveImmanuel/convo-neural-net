from coolcnn.layers.kernel_layer import KernelLayer
from enum import Enum
from typing import Union, Tuple, List

from nptyping import ndarray

import numpy as np


class PoolMode(Enum):
    MAX = 'max'
    AVG = 'avg'


class Pooling(KernelLayer):
    def __init__(self, mode: PoolMode = PoolMode.MAX, **kwargs) -> None:
        super().__init__(n_kernel=1, **kwargs)
        self.__mode = mode

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
