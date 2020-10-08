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
            return 1 / (1 + np.exp(-1 * value))

        if activation_type == ActivationType.RELU:
            return np.maximum(0, value)

        return value

    @staticmethod
    def differential(activation_type: ActivationType, value: ndarray) -> ndarray:
        if activation_type == ActivationType.SIGMOID:
            return value * (1 - value)

        if activation_type == ActivationType.RELU:
            # output = np.maximum(np.zeros(value.shape), value)
            value[value > 0] = 1
            return value

        return value

    @staticmethod
    def init_weight(activation_type: ActivationType, fan_in: int, fan_out: int) -> ndarray:
        if activation_type == ActivationType.SIGMOID:
            return np.random.uniform(-1, 1, size=(fan_in, fan_out)) * math.sqrt(6. / (fan_in + fan_out))

        if activation_type == ActivationType.RELU:
            return np.random.normal(0, 1, size=(fan_in, fan_out)) * math.sqrt(2. / fan_in)

        return np.random.normal(0, 0.08, size=(fan_in, fan_out))
