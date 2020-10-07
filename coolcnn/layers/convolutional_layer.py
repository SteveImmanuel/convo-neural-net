from typing import List, Tuple, Union

import numpy as np
from nptyping import ndarray

from coolcnn.activation import *
from coolcnn.layers.kernel_layer import KernelLayer


class Convolutional(KernelLayer):
    def __init__(self, activator: ActivationType = ActivationType.RELU, **kwargs) -> None:
        super().__init__(**kwargs)
        self._activator = activator

    def backpropagate(self, input_layer: ndarray, output_layer: ndarray, d_error_d_out: ndarray) -> ndarray:
        """
        derr/dnet = dout/dnet * derror/dout
        output_layer = (ft_map_size, n_kernel)
        d_error_d_out = (ft_map_size, n_kernel)
        d_error_d_out_prev = (input_size) # input size itu udah 3d
        weights = (n_kernel, kernel_shape, n_input_channel) # kernel shape 2d
        bias = (n_kernel,)
        """
        d_error_d_net = Activation.differential(self._activator, output_layer) * d_error_d_out
        # d_error_d_out_prev = np.zeros((*output_layer.shape[:-1], input_layer.shape[-1]))
        d_error_d_out_prev = np.zeros(input_layer.shape)
        d_error_d_out_prev = self._add_padding(d_error_d_out_prev)
        w_strides, h_strides = self._strides

        # for receptive_field, output_row, output_col in self._gen_receptive_field(input_layer, d_error_d_out.shape, self._weights.shape[1:]):
        for receptive_field, output_row, output_col in self._gen_receptive_field(input_layer):
            anchor_left = output_col * w_strides
            anchor_top = output_row * h_strides

            kernel_w, kernel_h = self._kernel_shape

            for idx in range(self._n_kernel):
                d_error_d_in = self._weights[idx] * d_error_d_net[output_row, output_col, idx]
                # print(d_error_d_in.shape)
                # print(d_error_d_in)
                # print(d_error_d_in.shape)
                # print(d_error_d_out_prev[anchor_left:anchor_left + kernel_w, anchor_top:anchor_top + kernel_h, :].shape)
                # print(anchor_left, kernel_w, anchor_top, kernel_h)

                d_error_d_out_prev[anchor_left:anchor_left + kernel_w, anchor_top:anchor_top + kernel_h] += d_error_d_in
                # print(receptive_field.shape)
                # print(d_error_d_net[output_row, output_col, idx].shape)
                # print(d_error_d_net[output_row, output_col, idx])
                # print(self._weights_delta.shape)

                self._weights_delta[idx] += receptive_field * d_error_d_net[output_row, output_col, idx]
                self._bias_delta[idx] += d_error_d_net[output_row, output_col, idx]

                # receptive_field *= d_error_d_net[:, :, idx].reshape(d_error_d_net.shape[0], d_error_d_net.shape[1], 1)
                # summed_receptive_field = np.sum(receptive_field, axis=(0, 1))

                # self._weights_delta[idx][output_row][output_col] = summed_receptive_field

        # print(d_error_d_out_prev.shape)
        if self._padding == 0:
            return d_error_d_out_prev
        else:
            return d_error_d_out_prev[self._padding:-self._padding, self._padding:-self._padding]

    def update_weight(self, momentum: float, learning_rate: float) -> None:
        # print('layer convo')
        # print('delta weight', self._weights_delta)
        # print('delta bias', self._bias_delta)
        self._weights = self._weights - learning_rate * self._weights_delta - momentum * self._prev_weights_delta
        self._prev_weights_delta = self._weights_delta
        self._weights_delta = np.zeros(self._weights_delta.shape)

        self._bias = self._bias - learning_rate * self._bias_delta - momentum * self._prev_bias_delta
        self._prev_bias_delta = self._bias_delta
        self._bias_delta = np.zeros(self._bias_delta.shape)

    def _on_receptive_field(
        self,
        receptive_field: ndarray,
        feature_map: ndarray,
        output_row: int,
        output_col: int,
    ) -> None:
        summed_receptive_field = np.sum(self._weights * receptive_field, axis=(1, 2, 3)) + self._bias
        feature_map[output_row][output_col] = Activation.process(self._activator, summed_receptive_field)

    def _generate_weight(self):
        n_channel = self._input_shape[-1]
        row, col = self._kernel_shape

        total_filter_element = row * col * n_channel
        self._weights = np.array([], dtype=float)
        # for i in range(self._n_kernel):
        self._weights = np.random.normal(0, 0.08, size=(self._n_kernel, row, col, n_channel))
        # self._weights = np.concatenate((self._weights, random_weight))

        # self._weights = np.reshape(self._weights, (self._n_kernel, row, col, n_channel))
        self._weights_delta = np.zeros(self._weights.shape)
        self._prev_weights_delta = np.zeros(self._weights.shape)

        # self._bias = np.random.rand(self._n_kernel)
        self._bias = np.zeros(self._n_kernel)
        self._bias_delta = np.zeros(self._n_kernel)
        self._prev_bias_delta = np.zeros(self._n_kernel)

    def _get_trainable_params(self) -> int:
        trainable_params = np.prod(self._weights.shape) + np.prod(self._bias.shape)
        return trainable_params
