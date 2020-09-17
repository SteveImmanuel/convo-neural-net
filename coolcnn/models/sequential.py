from typing import List, Tuple

from nptyping import ndarray

from coolcnn.layers.base_layer import BaseLayer


class Sequential():
    def __init__(self, layers: List[BaseLayer] = []) -> None:
        self._layers = layers

    def add_layer(self, layer: BaseLayer) -> None:
        self._layers.append(layer)

    def compile(self):
        input_shape = self._layers[0].input_shape
        for layer in self._layers:
            layer.input_shape = input_shape
            input_shape = layer.output_shape

    def run(self, input_array: ndarray) -> ndarray:
        for layer in self._layers:
            input_array = layer.process(input_array)

        return input_array
