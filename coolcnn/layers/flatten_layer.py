from nptyping import ndarray

from coolcnn.layers.base_layer import BaseLayer


class Flatten(BaseLayer):
    def __init__(self, **kwargs):
        super.__init__(kwargs)

    def process(self, input_layer: ndarray) -> ndarray:
        return input_layer.flatten()
