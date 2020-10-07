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
        out_layer = np.array([], dtype=float)
        for node_idx, weight in enumerate(self._weights):
            calculated_value = np.sum(input_layer * weight) + self._bias[node_idx]
            out_layer = np.append(out_layer, Activation.process(self._activator, calculated_value))

        return out_layer

    def update_weight(self, momentum: float, learning_rate: float) -> None:
        # print('layer dense')
        # print('delta weight', self._weights_delta)
        # print('delta bias', self._bias_delta)
        self._weights = self._weights - learning_rate * self._weights_delta - momentum * self._prev_weights_delta
        self._prev_weights_delta = self._weights_delta
        self._weights_delta = np.zeros(self._weights.shape)

        self._bias = self._bias - learning_rate * self._bias_delta - momentum * self._prev_bias_delta
        self._prev_bias_delta = self._bias_delta
        self._bias_delta = np.zeros(self.__n_nodes)

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
        self._bias_delta += d_error_d_net

        d_error_d_out_prev = np.matmul(d_error_d_net.reshape(1, d_error_d_net.shape[0]), self._weights)
        return d_error_d_out_prev.flatten()

    def _generate_weight(self):
        self._weights = np.array([np.random.normal(scale=0.1, size=self._input_shape[-1]) for _ in range(self.__n_nodes)])
        self._weights_delta = np.zeros(self._weights.shape)
        self._prev_weights_delta = np.zeros(self._weights.shape)

        self._bias = np.random.normal(scale=0.1, size=self.__n_nodes)
        self._bias_delta = np.zeros(self.__n_nodes)
        self._prev_bias_delta = np.zeros(self.__n_nodes)

    def _get_output_shape(self) -> Tuple:
        return (self.__n_nodes, )

    def _get_trainable_params(self) -> int:
        return (self._input_shape[0] + 1) * self.__n_nodes
