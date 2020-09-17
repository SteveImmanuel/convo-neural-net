from abc import ABC, abstractmethod
from nptyping import ndarray
from typing import Tuple


class BaseLayer(ABC):
    def __init__(self, input_shape: Tuple[int, int, int]) -> None:
        self._input_shape = input_shape  # (w, h, c)

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

    @abstractmethod
    def process(self, input_layer: ndarray) -> ndarray:
        pass
