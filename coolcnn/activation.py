from enum import Enum
from nptyping import ndarray

import math
import numpy as np


class ActivationType(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    NONE = 'none'


class Activation():
    @staticmethod
    def process(activation_type: ActivationType, value: int) -> int:
        # print('before activation:', value)
        if activation_type == ActivationType.SIGMOID:
            return 1 / (1 + np.exp(-value))

        if activation_type == ActivationType.RELU:
            return np.maximum(0, value)

        return value

    @staticmethod
    def differential(activation_type: ActivationType, value: ndarray) -> ndarray:
        if activation_type == ActivationType.SIGMOID:
            return value * (1 - value)

        if activation_type == ActivationType.RELU:
            return np.maximum(np.zeros(value.shape), value)

        return value
