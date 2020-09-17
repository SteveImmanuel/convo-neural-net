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
    conv_layer._bias = [0]

    result = conv_layer.process(input_layer)

    assert result.shape == (3, 3, 1)
    assert result.ndim == 3
    assert result[0][0][0] == 1
    assert result[0][1][0] == 5
    assert result[0][2][0] == 6
    assert result[1][0][0] == 7
    assert result[1][1][0] == 8
    assert result[1][2][0] == 3
    assert result[2][0][0] == 3
    assert result[2][1][0] == 10
    assert result[2][2][0] == 4


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
    conv_layer._bias = [0]

    output_layer = np.array([
        [[44], [284], [536]],
        [[74], [424], [542]],
        [[107], [525], [494]],
    ])

    result = conv_layer.process(input_layer)

    assert result.shape == (3, 3, 1)
    assert result.ndim == 3
    assert np.all(result == output_layer)


def test_convolution_3():
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
        [[-1], [0], [-2]],
        [[-1], [0], [-2]],
        [[-1], [0], [-2]],
    ]])
    conv_layer._bias = [0]

    result = conv_layer.process(input_layer)

    assert result.shape == (3, 3, 1)
    assert result.ndim == 3
    assert result[0][0][0] == 0
    assert result[0][1][0] == 0
    assert result[0][2][0] == 0
    assert result[1][0][0] == 0
    assert result[1][1][0] == 0
    assert result[1][2][0] == 0
    assert result[2][0][0] == 0
    assert result[2][1][0] == 0
    assert result[2][2][0] == 0


def test_convolution_image():
    input_layer = np.array(Image.open('tests/test.jpg').convert('RGB'), dtype=np.float64)

    conv_layer = Convolutional(
        n_kernel=1, kernel_shape=100, strides=50, padding=0, input_shape=input_layer.shape
    )
    conv_layer._generate_weight()

    result = conv_layer.process(input_layer)
    assert result.ndim == 3


def test_convolution_4():
    input_layer = np.array(
        [
            [[2], [2], [3], [3], [3]],
            [[0], [1], [3], [0], [3]],
            [[2], [3], [0], [1], [3]],
            [[3], [3], [2], [1], [2]],
            [[3], [3], [0], [2], [3]],
        ]
    )

    output_array = np.zeros((3, 3, 1))

    conv_layer = Convolutional(
        n_kernel=1, kernel_shape=3, strides=1, padding=0, input_shape=(5, 5, 1)
    )

    conv_layer._weights = np.array([[
        [[-1], [0], [-2]],
        [[-1], [0], [-2]],
        [[-1], [0], [-2]],
    ]])
    conv_layer._bias = [0]

    result = conv_layer.process(input_layer)

    assert result.shape == (3, 3, 1)
    assert result.ndim == 3
    assert np.all(result == output_array)


def test_convolution_5():
    input_layer = np.array(
        [
            [[252, 252], [251, 251], [246, 246], [207, 207], [90, 90]],
            [[250, 250], [242, 242], [236, 236], [144, 144], [41, 41]],
            [[252, 252], [244, 244], [228, 228], [102, 102], [43, 43]],
            [[250, 250], [243, 243], [214, 214], [59, 59], [52, 52]],
            [[248, 248], [243, 243], [201, 201], [44, 44], [54, 54]],
        ]
    )

    conv_layer = Convolutional(
        n_kernel=1, kernel_shape=3, strides=1, padding=0, input_shape=(5, 5, 2)
    )

    conv_layer._weights = np.array([[
        [[1], [0], [-1]],
        [[1], [0], [-1]],
        [[1], [0], [-1]],
    ]])
    conv_layer._bias = [0, 0]

    result = conv_layer.process(input_layer)

    output_array = np.array(
        [
            [[44], [284], [536]],
            [[74], [424], [542]],
            [[107], [525], [494]],
        ]
    ) * 2

    assert result.shape == (3, 3, 1)
    assert result.ndim == 3
    assert np.all(result == output_array)


def test_convolution_6():
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
        n_kernel=2, kernel_shape=3, strides=1, padding=0, input_shape=(5, 5, 1)
    )

    conv_layer._weights = np.array(
        [
            [
                [[1], [0], [-1]],
                [[1], [0], [-1]],
                [[1], [0], [-1]],
            ], [
                [[-1], [0], [-1]],
                [[-1], [0], [-1]],
                [[-1], [0], [-1]],
            ]
        ]
    )
    conv_layer._bias = [0, 0]

    result = conv_layer.process(input_layer)

    output_array = np.array(
        [
            [[44, 0], [284, 0], [536, 0]],
            [[74, 0], [424, 0], [542, 0]],
            [[107, 0], [525, 0], [494, 0]],
        ]
    )

    assert result.shape == (3, 3, 2)
    assert result.ndim == 3
    assert np.all(result == output_array)