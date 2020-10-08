import numpy as np
from nptyping import ndarray

from coolcnn.activation import *
from coolcnn.layers.base_layer import BaseLayer
from typing import Tuple


class Dense(BaseLayer):
    def __init__(self, n_nodes=int, activator: ActivationType = ActivationType.SIGMOID, **kwargs):
        super().__init__(**kwargs)
        self.__n_nodes = n_nodes
        self._activator = activator

    def process(self, input_layer: ndarray) -> ndarray:
        calculated_value = np.multiply(self._weights, input_layer)
        summed_value = np.sum(calculated_value, axis=-1) + self._bias
        out_array = Activation.process(self._activator, summed_value)

        return out_array

    def update_weight(self, momentum: float, learning_rate: float) -> None:
        # print('layer dense')
        # print('delta weight', self._weights_delta)
        # print('delta bias', self._bias_delta)
        self._weights = self._weights - learning_rate * self._weights_delta - momentum * self._prev_weights_delta
        self._prev_weights_delta = self._weights_delta
        self._weights_delta = np.zeros(self._weights.shape)

        self._bias = self._bias - learning_rate * self._bias_delta - momentum * self._prev_bias_delta
        self._prev_bias_delta = self._bias_delta
        self._bias_delta = 0

    def backpropagate(self, input_layer: ndarray, output_layer: ndarray, d_error_d_out: ndarray) -> ndarray:
        # weight = (n_nodes, n_input)
        # input_layer = (n_input, )
        # d_error_d_out = (n_nodes, )
        # d_error_d_net = (n_nodes, )input_layer
        # out = (n_input, )
        # bias = (n_nodes, )

        d_error_d_net = Activation.differential(self._activator, output_layer) * d_error_d_out
        d_error_d_weight = np.matmul(d_error_d_net.reshape(d_error_d_net.shape[0], 1), input_layer.reshape(1, input_layer.shape[0]))

        self._weights_delta += d_error_d_weight
        self._bias_delta += np.sum(d_error_d_net)

        d_error_d_out_prev = np.matmul(d_error_d_net.reshape(1, d_error_d_net.shape[0]), self._weights)
        return d_error_d_out_prev.flatten()

    def _generate_weight(self):
        self._weights = np.random.normal(0, 0.08, size=(self.__n_nodes, self._input_shape[-1]))
        self._weights_delta = np.zeros(self._weights.shape)
        self._prev_weights_delta = np.zeros(self._weights.shape)

        self._bias = 0
        self._bias_delta = 0
        self._prev_bias_delta = 0

    def _get_output_shape(self) -> Tuple:
        return (self.__n_nodes, )

    def _get_trainable_params(self) -> int:
        return (self._input_shape[0] + 1) * self.__n_nodes
