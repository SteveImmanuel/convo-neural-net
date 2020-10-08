from __future__ import annotations
from typing import List, Tuple
from nptyping import ndarray

import numpy as np
import pickle
import time

from datetime import datetime
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

        self.history = []

        step = 0
        mse = 0.
        acc = 0.

        start_time = time.time()
        start = datetime.now()
        while step < (epoch * len(input_array)):
            # print('Step', step, ':', end=' ')
            print(
                step // len(input_array) + 1,
                step % len(input_array) + 1,
                '/',
                len(input_array),
                end='\r'
            )
            data_idx = step % len(input_array)
            res = self.run(input_array[data_idx]).copy()

            mse += np.sum((result_array - res)**2)
            res[res < 0.5] = 0
            res[res >= 0.5] = 1
            acc += np.sum(res == result_array[data_idx])

            self._backpropagate(result_array[data_idx])

            if (step + 1) % mini_batch == 0:
                # +1 because of mod and step start with 0
                for layer in self._layers:
                    layer.update_weight(momentum, learning_rate)

            if step % len(input_array) == 0 and step != 0:
                print(
                    '{}/{} epoch in {:.2f}s'.format(
                        (step + 1) // len(input_array), epoch,
                        time.time() - start_time
                    )
                )
                start_time = time.time()
                # one epoch
                print()
                print('elapsed', datetime.now() - start)
                print('mse', mse / len(input_array))
                print('acc', acc / len(input_array))
                self.history.append((mse, acc))
                start = datetime.now()
                mse = 0.
                acc = 0.
            step += 1

    def summary(self):
        if not self._compiled:
            raise RuntimeError('Error, please compile model first')

        total = 0
        for idx, layer in enumerate(self._layers):
            print(
                '{:<5}{:<20}Output Shape: {:<20}Trainable Params: {:<20}'.format(
                    str(idx + 1) + '.',
                    type(layer).__name__,
                    str(layer.output_shape),
                    layer.trainable_params,
                )
            )
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
        # print(predicted_array, target_array)
        # print('mse', np.sum((target_array - predicted_array)**2))

        for layer, input_array, output_array in zip(
            self._layers[::-1], self._input_array_list[-2::-1], self._input_array_list[::-1]
        ):
            d_error_d_out = layer.backpropagate(input_array, output_array, d_error_d_out)

        self._input_array_list = []

    def save(self, save_path: str) -> None:
        with open(save_path, 'wb') as model_out:
            pickle.dump(self, model_out)

    @staticmethod
    def load(save_path: str) -> Sequential:
        with open(save_path, 'rb') as model_saved:
            return pickle.load(model_saved)
