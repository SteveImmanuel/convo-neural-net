from coolcnn.layers import Pooling, PoolMode

import numpy as np


def test_pooling_1():
    input_array = np.array(
        [
            [[1], [1], [2], [4]],
            [[5], [6], [7], [8]],
            [[3], [2], [1], [0]],
            [[1], [2], [3], [4]],
        ]
    )

    layer = Pooling(kernel_shape=2, strides=2, input_shape=(4, 4, 1))

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

    layer = Pooling(kernel_shape=2, strides=2, input_shape=(4, 4, 1), mode=PoolMode.AVG)

    result = layer.process(input_array)

    assert result.shape == (2, 2, 1)
    assert result[0][0][0] == 5
    assert result[0][1][0] == 3
    assert result[1][0][0] == 6
    assert result[1][1][0] == 4.5