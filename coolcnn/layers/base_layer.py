from abc import ABC, abstractmethod
from typing import Tuple, Union

from nptyping import ndarray


class BaseLayer(ABC):
    def __init__(self, input_shape: Tuple = None) -> None:
        self._input_shape = input_shape  # (h, w, c)

    @abstractmethod
    def _validate_weight(self) -> bool:
        pass

    @property
    def weight(self) -> ndarray:
        return self._weights

    @weight.setter
    def weight(self, new_weight: ndarray) -> None:
        if (not self._validate_weight(new_weight)):
            raise ValueError('Invalid weight input')
        self._weights = weight

    @property
    def input_shape(self) -> Tuple:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: Tuple) -> None:
        self._input_shape = input_shape
        self._generate_weight()

    def _generate_weight(self):
        pass

    @abstractmethod
    def process(self, input_layer: ndarray) -> ndarray:
        pass

    @property
    def output_shape(self) -> Tuple:
        return self._get_output_shape()

    @abstractmethod
    def _get_output_shape(self) -> Tuple:
        pass
