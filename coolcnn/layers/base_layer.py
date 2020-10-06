from abc import ABC, abstractmethod
from typing import Tuple, Union

from nptyping import ndarray


class BaseLayer(ABC):
    def __init__(self, input_shape: Tuple = None) -> None:
        self._input_shape = input_shape  # (h, w, c)
        self._weights_delta = None
        self._prev_weights_delta = None
        self._activator = None

    @property
    def weight(self) -> ndarray:
        return self._weights

    @weight.setter
    def weight(self, new_weight: ndarray) -> None:
        self._weights = new_weight

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

    @abstractmethod
    def backpropagate(self, input_layer: ndarray, output_layer: ndarray, d_error_d_out: ndarray) -> ndarray:
        return d_error_d_out

    @abstractmethod
    def update_weight(self) -> None:
        pass

    @property
    def output_shape(self) -> Tuple:
        return self._get_output_shape()

    @abstractmethod
    def _get_output_shape(self) -> Tuple:
        pass

    @property
    def trainable_params(self) -> int:
        return self._get_trainable_params()

    def _get_trainable_params(self) -> int:
        return 0
