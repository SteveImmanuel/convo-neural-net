from enum import Enum
import math


class ActivationType(Enum):
    RELU = 'relu'
    SIGMOID = 'sigmoid'
    NONE = 'none'


class Activation():
    @staticmethod
    def process(activation_type: ActivationType, value: int) -> int:
        if activation_type == ActivationType.SIGMOID:
            return 1 / (1 + math.exp(-value))

        if activation_type == ActivationType.RELU:
            return max(0, value)

        return value