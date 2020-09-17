import numpy as np

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