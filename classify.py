from coolcnn.layers import Convolutional, Pooling, Flatten, Dense
from coolcnn.models import Sequential
from coolcnn.activation import *

from sklearn.model_selection import train_test_split

from PIL import Image
import os
import numpy as np
import random

dataset_dir = 'tests/data/train'
cat_dir = dataset_dir + '/cats'
dog_dir = dataset_dir + '/dogs'
cat_files = os.listdir(cat_dir)
dog_files = os.listdir(dog_dir)

DOG_TARGET = 0
CAT_TARGET = 1
IMAGE_SIZE = 100

total_data = len(cat_files) + len(dog_files)

# m_keras =

# model = Sequential(
#     [
# Convolutional(n_kernel=5, kernel_shape=(3, 3), strides=1, padding=2, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activator=ActivationType.RELU),
# Pooling(kernel_shape=(2, 2), strides=2),
# Convolutional(n_kernel=10, kernel_shape=(3, 3), strides=2, padding=2, activator=ActivationType.RELU),
# Pooling(kernel_shape=(3, 3), strides=2),
# Flatten(),
# Dense(n_nodes=50, activator=ActivationType.RELU, input_shape=(IMAGE_SIZE * IMAGE_SIZE * 3, )),
# Dense(n_nodes=1, activator=ActivationType.SIGMOID),

# Convolutional(n_kernel=32, kernel_shape=(3, 3), strides=1, padding=0, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activator=ActivationType.RELU),
# Pooling(kernel_shape=(2, 2), strides=2),
# Convolutional(n_kernel=64, kernel_shape=(3, 3), strides=2, padding=0, activator=ActivationType.RELU),
# Pooling(kernel_shape=(2, 2), strides=2),
# Convolutional(n_kernel=128, kernel_shape=(3, 3), strides=1, padding=0, activator=ActivationType.RELU),
# Pooling(kernel_shape=(2, 2), strides=2),
# Flatten(),
# Dense(n_nodes=128, activator=ActivationType.SIGMOID, input_shape=(IMAGE_SIZE * IMAGE_SIZE * 3, )),
# Dense(n_nodes=64, activator=ActivationType.SIGMOID),
# Dense(n_nodes=32, activator=ActivationType.SIGMOID),
# Dense(n_nodes=16, activator=ActivationType.SIGMOID),
# Dense(n_nodes=8, activator=ActivationType.SIGMOID),
# Dense(n_nodes=4, activator=ActivationType.SIGMOID),
# Dense(n_nodes=2, activator=ActivationType.SIGMOID),
# Dense(n_nodes=100, activator=ActivationType.RELU),
# Dense(n_nodes=1, activator=ActivationType.SIGMOID),
#     ]
# )

model = Sequential(
    [
        Convolutional(n_kernel=32, kernel_shape=(3, 3), strides=1, padding=2, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), activator=ActivationType.RELU),
        Pooling(kernel_shape=(2, 2), strides=2),
        Flatten(),
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

    input_array.append(input_layer)
    target_array.append(np.array([CAT_TARGET]))

for dog in dog_files:
    path = os.path.join(dog_dir, dog)
    image = Image.open(path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    input_layer = np.array(image, dtype=np.float64) / 255.0

    input_array.append(input_layer)
    target_array.append(np.array([DOG_TARGET]))

train_input, validation_input, train_target, validation_target = train_test_split(input_array, target_array, test_size=0.1)

model.fit(train_input, train_target, mini_batch=10, learning_rate=0.0001, epoch=100)

correct_classification = 0

for input_entry, target_entry in zip(validation_input, validation_target):
    result = model.run(input_entry)
    correct_classification += result[0] == target_entry[0]

print('Accuracy with val data: {:.2f}'.format(correct_classification / len(validation_input)))
