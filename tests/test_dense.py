import numpy as np

from coolcnn.activation import *
from coolcnn.layers import Dense


def test_dense_1():
    input_layer = np.array([1, 2, 3, 4, 5, 6, 7])

    layer = Dense(n_nodes=3)
    layer.input_shape = input_layer.shape
    layer._generate_weight()

    result = layer.process(input_layer)

    assert result.shape == (3, )


def test_dense_2():
    input_layer = np.array([5, 3, 4])

    output_layer = np.array([24, 0, 23])

    layer = Dense(n_nodes=3, activator=ActivationType.RELU)
    layer.input_shape = input_layer.shape
    layer._weights = np.array([[1, 2, 3], [-3, -2, 1], [1, 3, 2]])
    layer._bias = 1

    result = layer.process(input_layer)

    assert np.all(result == output_layer)


def test_dense_3():
    input_layer = np.array([5, 3, 4])

    output_layer_before_sigmoid = [3.3, -0.7, 3.2]
    output_layer = np.array(
        list(
            map(
                lambda x: Activation.process(ActivationType.SIGMOID, x),
                output_layer_before_sigmoid,
            )
        )
    )

    layer = Dense(n_nodes=3)
    layer.input_shape = input_layer.shape
    layer._weights = np.array([[0.1, 0.2, 0.3], [-0.3, -0.2, 0.1], [0.1, 0.3, 0.2]])
    layer._bias = 1

    result = layer.process(input_layer)
    assert np.sum(result - output_layer) < 0.0001