import numpy as np

from coolcnn.activation import *
from coolcnn.layers import Dense
from coolcnn.models import Sequential


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
    output_layer = np.array(list(map(
        lambda x: Activation.process(ActivationType.SIGMOID, x),
        output_layer_before_sigmoid,
    )))

    layer = Dense(n_nodes=3)
    layer.input_shape = input_layer.shape
    layer._weights = np.array([[0.1, 0.2, 0.3], [-0.3, -0.2, 0.1], [0.1, 0.3, 0.2]])
    layer._bias = 1

    result = layer.process(input_layer)
    assert np.sum(result - output_layer) < 0.0001


def test_dense_backprop():
    # Calculation from slide

    layer_1 = Dense(n_nodes=2, activator=ActivationType.SIGMOID, input_shape=(2, ))
    layer_2 = Dense(n_nodes=2, activator=ActivationType.SIGMOID)
    model = Sequential([layer_1, layer_2])

    model.compile()

    layer_1.weight = np.array([[0.15, 0.2], [0.25, 0.3]])
    layer_2.weight = np.array([[0.4, 0.45], [0.5, 0.55]])

    layer_1._bias = 0.35
    layer_2._bias = 0.6

    model.fit(np.array([[0.05, 0.1]]), np.array([[0.01, 0.99]]), epoch=1, mini_batch=1, learning_rate=0.5)

    expected_layer_1_weight = np.array([[0.14978072, 0.19956143], [0.24975114, 0.29950229]])
    expected_layer_2_weight = np.array([[0.35891648, 0.40866619], [0.51130127, 0.56137012]])
    expected_layer_1_bias = -0.35 + 0.34561432 + 0.34561432
    expected_layer_2_bias = -0.6 + 0.53075072 + 0.61904912

    assert np.sum(layer_1.weight - expected_layer_1_weight) < 0.00001
    assert np.sum(layer_2.weight - expected_layer_2_weight) < 0.00001
    assert np.sum(expected_layer_1_bias - layer_1._bias) < 0.001
    assert np.sum(expected_layer_2_bias - layer_2._bias) < 0.001
