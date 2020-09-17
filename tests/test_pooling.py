from coolcnn.layers import Pooling, PoolMode

import numpy as np

from unittest import TestCase


def test_pooling_1():
    input_array = np.array(
        [
            [[1], [1], [2], [4]],
            [[5], [6], [7], [8]],
            [[3], [2], [1], [0]],
            [[1], [2], [3], [4]],
        ]
    )

    layer = Pooling(kernel_shape=2, strides=2)
    layer.input_shape = input_array.shape

    result = layer.process(input_array)

    assert result.shape == (2, 2, 1)
    assert result[0][0][0] == 6
    assert result[0][1][0] == 8
    assert result[1][0][0] == 3
    assert result[1][1][0] == 4


def test_pooling_2():
    input_array = np.array(
        [
            [[1], [1], [2], [2]],
            [[9], [9], [4], [4]],
            [[4], [8], [3], [6]],
            [[8], [4], [3], [6]],
        ]
    )

    layer = Pooling(kernel_shape=2, strides=2, mode=PoolMode.AVG)
    layer.input_shape = input_array.shape

    result = layer.process(input_array)

    assert result.shape == (2, 2, 1)
    assert result[0][0][0] == 5
    assert result[0][1][0] == 3
    assert result[1][0][0] == 6
    assert result[1][1][0] == 4.5


def test_pooling_3():
    input_array = np.array(
        [
            [[1, 2], [1, 2], [2, 3], [2, 3]],
            [[9, 2], [9, 2], [4, 2], [4, 2]],
            [[4, 3], [8, 3], [3, 5], [6, 5]],
            [[8, 5], [4, 5], [3, 9], [6, 9]],
        ]
    )

    output_array = np.array([
        [[5, 2], [3, 2.5]],
        [[6, 4], [4.5, 7]],
    ])

    layer = Pooling(kernel_shape=2, strides=2, mode=PoolMode.AVG)
    layer.input_shape = input_array.shape

    result = layer.process(input_array)

    assert result.shape == (2, 2, 2)
    assert np.all(result == output_array)


def test_pooling_4():
    input_array = np.array(
        [
            [[1, 2], [1, 2], [2, 3], [2, 3]],
            [[9, 2], [9, 2], [4, 2], [4, 1]],
            [[4, 3], [8, 3], [3, 11], [6, 12]],
            [[8, 5], [4, 5], [3, 14], [6, 13]],
        ]
    )

    output_array = np.array([
        [[9, 2], [4, 3]],
        [[8, 5], [6, 14]],
    ])

    layer = Pooling(kernel_shape=2, strides=2)
    layer.input_shape = input_array.shape

    result = layer.process(input_array)

    assert result.shape == (2, 2, 2)
    assert np.all(result == output_array)
