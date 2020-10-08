import numpy as np
import tempfile
import os
import uuid

from coolcnn.activation import *
from coolcnn.layers import *
from coolcnn.models import Sequential


def test_sequential_1():
    input_layer = np.arange(27).reshape((3, 3, 3)).astype(float)

    model = Sequential(
        [
            Convolutional(n_kernel=5, kernel_shape=2, strides=1, padding=1, input_shape=(3, 3, 3)),
            Pooling(kernel_shape=2, strides=2),
            Flatten(),
            Dense(n_nodes=3),
        ]
    )

    model.compile()
    result = model.run(input_layer)


def test_sequential_2():
    input_layer = np.arange(625 * 3).reshape((25, 25, 3)).astype(float)

    model = Sequential(
        [
            Convolutional(
                n_kernel=5, kernel_shape=2, strides=1, padding=1, input_shape=(25, 25, 3)
            ),
            Pooling(kernel_shape=(2, 2), strides=2),
            Convolutional(n_kernel=10, kernel_shape=2, strides=1, padding=1),
            Pooling(kernel_shape=(2, 2), strides=2),
            Convolutional(n_kernel=15, kernel_shape=2, strides=1, padding=1),
            Pooling(kernel_shape=(2, 2), strides=2),
            Flatten(),
            Dense(n_nodes=100),
            Dense(n_nodes=50),
            Dense(n_nodes=20),
            Dense(n_nodes=1),
        ]
    )

    model.compile()
    model.summary()
    result = model.run(input_layer)
    print(result)
    assert result.shape == (1, 1)


def test_sequential_3():
    input_layer = np.array(
        [
            [[252, 252], [251, 251], [246, 246], [207, 207], [90, 90]],
            [[250, 250], [242, 242], [236, 236], [144, 144], [41, 41]],
            [[252, 252], [244, 244], [228, 228], [102, 102], [43, 43]],
            [[250, 250], [243, 243], [214, 214], [59, 59], [52, 52]],
            [[248, 248], [243, 243], [201, 201], [44, 44], [54, 54]],
        ],
        dtype=np.float64
    )

    conv_layer_1 = Convolutional(
        n_kernel=2,
        kernel_shape=3,
        strides=1,
        padding=0,
        input_shape=input_layer.shape,
    )

    pool_layer_1 = Pooling(kernel_shape=3, strides=1)

    dense_layer_1 = Dense(n_nodes=2)

    model = Sequential([conv_layer_1, pool_layer_1, Flatten(), dense_layer_1])

    model.compile()

    conv_layer_1._weights = np.array(
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
    conv_layer_1._bias = np.zeros((2, ))

    # dense_layer_1._weights = np.array([[0, 2], [0.001, 0]])
    dense_layer_1._weights = np.array([[0, 0.001], [2, 0]])

    result = model.run(input_layer)

    assert result.shape == (1, 2)
    assert result[0][0] == Activation.process(ActivationType.SIGMOID, 0)
    assert result[0][1] == Activation.process(ActivationType.SIGMOID, 1084 / 1000)

def test_sequential_save_load():
    model = Sequential(
        [
            Convolutional(
                n_kernel=5, kernel_shape=2, strides=1, padding=1, input_shape=(25, 25, 3)
            ),
            Pooling(kernel_shape=(2, 2), strides=2),
            Flatten(),
            Dense(n_nodes=100),
            Dense(n_nodes=1),
        ]
    )
    model.compile()

    temp_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    model.save(temp_path)
    model_2 = Sequential.load(temp_path)

    assert np.all(model._layers[0].weight == model_2._layers[0].weight)

