from coolcnn.layers import Convolutional
from coolcnn.activation import *

from PIL import Image

import numpy as np


def test_convolution_1():
    input_layer = np.array(
        [
            [[2], [2], [3], [3], [3]],
            [[0], [1], [3], [0], [3]],
            [[2], [3], [0], [1], [3]],
            [[3], [3], [2], [1], [2]],
            [[3], [3], [0], [2], [3]],
        ]
    )

    conv_layer = Convolutional(
        n_kernel=1, kernel_shape=3, strides=2, padding=1, input_shape=(5, 5, 1)
    )

    conv_layer._weights = np.array([[
        [[2], [0], [1]],
        [[1], [0], [0]],
        [[0], [1], [1]],
    ]])
    conv_layer._Convolutional__bias = [0]

    result = conv_layer.process(input_layer)

    assert result.shape == (1, 3, 3)
    assert result.ndim == 3
    assert result[0][0][0] == 1
    assert result[0][0][1] == 5
    assert result[0][0][2] == 6
    assert result[0][1][0] == 7
    assert result[0][1][1] == 8
    assert result[0][1][2] == 3
    assert result[0][2][0] == 3
    assert result[0][2][1] == 10
    assert result[0][2][2] == 4


def test_convolution_2():
    input_layer = np.array(
        [
            [[252], [251], [246], [207], [90]],
            [[250], [242], [236], [144], [41]],
            [[252], [244], [228], [102], [43]],
            [[250], [243], [214], [59], [52]],
            [[248], [243], [201], [44], [54]],
        ]
    )

    conv_layer = Convolutional(
        n_kernel=1, kernel_shape=3, strides=1, padding=0, input_shape=(5, 5, 1)
    )

    conv_layer._weights = np.array([[
        [[1], [0], [-1]],
        [[1], [0], [-1]],
        [[1], [0], [-1]],
    ]])
    conv_layer._Convolutional__bias = [0]

    result = conv_layer.process(input_layer)

    assert result.shape == (1, 3, 3)
    assert result.ndim == 3
    assert result[0][0][0] == 44
    assert result[0][0][1] == 284
    assert result[0][0][2] == 536
    assert result[0][1][0] == 74
    assert result[0][1][1] == 424
    assert result[0][1][2] == 542
    assert result[0][2][0] == 107
    assert result[0][2][1] == 525
    assert result[0][2][2] == 494


def test_convolution_image():
    input_layer = np.array(Image.open('tests/test.jpg').convert('RGB'), dtype=np.float64)

    conv_layer = Convolutional(
        n_kernel=1, kernel_shape=100, strides=50, padding=0, input_shape=input_layer.shape
    )

    result = conv_layer.process(input_layer)
    assert result.ndim == 3
