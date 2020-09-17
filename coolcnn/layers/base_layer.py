from abc import ABC, abstractmethod
from nptyping import ndarray


class BaseLayer(ABC):
    def __init__(self, weight: ndarray) -> None:
        if (not self._validate_weight(weight)):
            raise ValueError('Invalid weight input')
        self._weight = weight

    @abstractmethod
    def _validate_weight(self) -> bool:
        pass

    @property
    def weight(self) -> ndarray:
        return self._weight

    @weight.setter
    def weight(self, new_weight: ndarray) -> None:
        if (not self._validate_weight(new_weight)):
            raise ValueError('Invalid weight input')
        self._weight = weight

    @abstractmethod
    def process(self, input_layer: ndarray) -> ndarray:
        pass
