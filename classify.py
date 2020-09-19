from coolcnn.layers import Convolutional, Pooling, Flatten, Dense
from coolcnn.models import Sequential
from coolcnn.activation import *

from PIL import Image
import os
import numpy as np

cat_dir = 'dataset/cats'
dog_dir = 'dataset/dogs'
cat_files = os.listdir(cat_dir)
dog_files = os.listdir(dog_dir)

total_data = len(cat_files) + len(dog_files)
correct_classification = 0

model = Sequential([
    Convolutional(n_kernel=5, kernel_shape=(3,3), strides=1, padding=2, input_shape=(250,250,3), activator=ActivationType.RELU),
    Pooling(kernel_shape=(2, 2), strides=2),
    Convolutional(n_kernel=10, kernel_shape=(3,3), strides=2, padding=0, activator=ActivationType.RELU),
    Pooling(kernel_shape=(3, 3), strides=3),
    Convolutional(n_kernel=15, kernel_shape=(3,3), strides=1, padding=0, activator=ActivationType.RELU),
    Pooling(kernel_shape=(2, 2), strides=2),
    Flatten(),
    Dense(n_nodes=50, activator=ActivationType.RELU),
    Dense(n_nodes=20, activator=ActivationType.RELU),
    Dense(n_nodes=1, activator=ActivationType.SIGMOID),
])

model.compile()
model.summary()
    
for cat in cat_files:
    path = os.path.join(cat_dir, cat)
    image = Image.open(path).convert('RGB')
    image = image.resize((250, 250))
    input_layer = np.array(image, dtype=np.float64) / 255.0

    result = model.run(input_layer)
    if (result[0] >= .5):
        print(cat, '=>', 'cat')
        correct_classification += 1
    else:
        print(cat, '=>', 'dog')

for dog in dog_files:
    path = os.path.join(dog_dir, dog)
    image = Image.open(path).convert('RGB')
    image = image.resize((250, 250))
    input_layer = np.array(image, dtype=np.float64) / 255.0

    result = model.run(input_layer)
    if (result[0] >= .5):
        print(dog, '=>', 'cat')
    else:
        print(dog, '=>', 'dog')
        correct_classification += 1
    

print('Total classification:', total_data)
print('Correct classification:', correct_classification)
