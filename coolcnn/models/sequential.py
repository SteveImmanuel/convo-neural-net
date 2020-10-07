from typing import List, Tuple
from nptyping import ndarray
import numpy as np

from coolcnn.layers.base_layer import BaseLayer


class Sequential():
    def __init__(self, layers: List[BaseLayer] = []) -> None:
        self._layers = layers
        self._compiled = False
        self._input_array_list = []

    def add_layer(self, layer: BaseLayer) -> None:
        self._layers.append(layer)

    def compile(self):
        self._compiled = True
        input_shape = self._layers[0].input_shape
        for layer in self._layers:
            layer.input_shape = input_shape
            input_shape = layer.output_shape

    def fit(
        self,
        input_array: ndarray,  # list of instance
        result_array: ndarray,
        epoch: int = 10,
        mini_batch: int = 2,
        learning_rate: float = 0.5,
        momentum: float = 0,
    ):
        if not self._compiled:
            raise RuntimeError('Error, please compile model first')

        step = 0

        while step < (epoch * len(input_array)):
            print('step', step)
            data_idx = step % len(input_array)
            res = self.run(input_array[data_idx])
            # print('res, target', res, result_array[data_idx])
            self._backpropagate(result_array[data_idx])

            if (step + 1) % mini_batch == 0:
                print('minibatch')
                # +1 because of mod and step start with 0
                for layer in self._layers:
                    layer.update_weight(momentum, learning_rate)
            step += 1

    def summary(self):
        if not self._compiled:
            raise RuntimeError('Error, please compile model first')

        total = 0
        for idx, layer in enumerate(self._layers):
            print('{:<5}{:<20}Output Shape: {:<20}Trainable Params: {:<20}'.format(
                str(idx + 1) + '.',
                type(layer).__name__,
                str(layer.output_shape),
                layer.trainable_params,
            ))
            total += layer.trainable_params
        print('=' * 92)
        print('Total trainable params:', total)

    def run(self, input_array: ndarray) -> ndarray:
        if not self._compiled:
            raise RuntimeError('Error, please compile model first')

        for layer in self._layers:
            self._input_array_list.append(input_array)
            input_array = layer.process(input_array)

        self._input_array_list.append(input_array)
        return input_array

    def _backpropagate(self, target_array: ndarray) -> None:
        predicted_array = self._input_array_list[-1]
        d_error_d_out = -(target_array - predicted_array)
        print('outmost derrordout', d_error_d_out)
        print(predicted_array, target_array)
        print('mse', np.sum((target_array - predicted_array)**2))

        for layer, input_array, output_array in zip(self._layers[::-1], self._input_array_list[-2::-1], self._input_array_list[::-1]):
            d_error_d_out = layer.backpropagate(input_array, output_array, d_error_d_out)

        self._input_array_list = []