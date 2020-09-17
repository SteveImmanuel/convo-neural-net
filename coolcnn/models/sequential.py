from coolcnn.layers.base_layer import BaseLayer
from nptyping import ndarray
from typing import List


class Sequential():
    _layers: List[BaseLayer] = []

    def __init__(self) -> None:
        pass

    def add_layer(self, layer: BaseLayer) -> None:
        self._layers.append(layer)

    def run(self, input_array: ndarray) -> ndarray:
        for layer in self._layers:
            input_array = layer.process(input_array)

        return input_array