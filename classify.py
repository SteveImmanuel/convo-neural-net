from coolcnn.layers import Convolutional, Pooling, Flatten, Dense
from coolcnn.models import Sequential
from coolcnn.activation import *

from PIL import Image
import os
import numpy as np
import random

dataset_dir = 'tests/sample'
cat_dir = dataset_dir + '/cats'
dog_dir = dataset_dir + '/dogs'
cat_files = os.listdir(cat_dir)
dog_files = os.listdir(dog_dir)

DOG_TARGET = 0
CAT_TARGET = 1
IMAGE_SIZE = 100

total_data = len(cat_files) + len(dog_files)
correct_classification = 0

model = Sequential(
    [
        Convolutional(n_kernel=5, kernel_shape=(3, 3), strides=1, padding=0, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activator=ActivationType.RELU),
        Pooling(kernel_shape=(2, 2), strides=2),
        Convolutional(n_kernel=10, kernel_shape=(3, 3), strides=2, padding=0, activator=ActivationType.RELU),
        Pooling(kernel_shape=(3, 3), strides=3),
        Convolutional(n_kernel=15, kernel_shape=(3, 3), strides=1, padding=0, activator=ActivationType.RELU),
        Pooling(kernel_shape=(2, 2), strides=2),
        Flatten(),
        Dense(n_nodes=100, activator=ActivationType.RELU, input_shape=(IMAGE_SIZE * IMAGE_SIZE * 3, )),
        Dense(n_nodes=1, activator=ActivationType.SIGMOID),
    ]
)

model.compile()
model.summary()

input_array = []
target_array = []

for cat in cat_files:
    path = os.path.join(cat_dir, cat)
    image = Image.open(path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    input_layer = np.array(image, dtype=np.float64) / 255.0

    input_array.append(input_layer.flatten())
    target_array.append(np.array([CAT_TARGET]))

for dog in dog_files:
    path = os.path.join(dog_dir, dog)
    image = Image.open(path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    input_layer = np.array(image, dtype=np.float64) / 255.0

    input_array.append(input_layer.flatten())
    target_array.append(np.array([DOG_TARGET]))

shuffle_list = list(zip(input_array, target_array))
random.shuffle(shuffle_list)
model.fit(*zip(*shuffle_list), mini_batch=10, learning_rate=0.1)

# print('Total classification:', total_data)
# print('Correct classification:', correct_classification)
