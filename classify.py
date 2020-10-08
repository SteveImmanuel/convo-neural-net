from coolcnn.layers import Convolutional, Pooling, Flatten, Dense
from coolcnn.models import Sequential
from coolcnn.activation import *

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from PIL import Image

import os
import numpy as np
import random
import sys
import argparse

DOG_TARGET = 0
CAT_TARGET = 1
IMAGE_SIZE = 100

LEARNING_RATE = 0.0005
EPOCH = 10
BATCH_SIZE = 32

RANDOM_STATE = 1

MODEL_SAVE_PATH = 'cnn.model'


def count_score(model, test_input, test_target, data_name):
    correct_classification = 0
    test_result = []

    for input_entry, target_entry in zip(test_input, test_target):
        result = model.run(input_entry)
        result = 1 if result[0] > 0.5 else 0
        test_result.append(result)
        correct_classification += result == target_entry[0]

    print('Accuracy with {}: {:.2f}'.format(data_name, correct_classification / len(test_target)))
    print('Confusion matrix: ')
    print(confusion_matrix(test_target, test_result))
    print()


def process_image(path):
    image = Image.open(path).convert('RGB')
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    input_layer = np.array(image, dtype=np.float64) / 255.0
    return input_layer


def load_dataset(root_path, target, input_array, target_array):
    for item in os.listdir(root_path):
        path = os.path.join(root_path, item)

        input_array.append(process_image(path))
        target_array.append(np.array(target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform cat dog classification')
    parser.add_argument('-l', '--load', action='store_true', help='Load stored model in default path (cnn.model)')
    parser.add_argument('-c', '--crossval', action='store_true', help='Perform 10 folds cross validation')
    args = parser.parse_args()
    # print(args.accumulate(args.integers))

    dataset_dir = 'tests/data/train'
    dataset_test_dir = 'tests/data/test'

    cat_dir = os.path.join(dataset_dir, 'cats')
    dog_dir = os.path.join(dataset_dir, 'dogs')
    cat_test_dir = os.path.join(dataset_test_dir, 'cats')
    dog_test_dir = os.path.join(dataset_test_dir, 'dogs')

    if args.load:
        model = Sequential.load(MODEL_SAVE_PATH)
    else:
        model = Sequential(
            [
                Convolutional(
                    n_kernel=32,
                    kernel_shape=(3, 3),
                    strides=1,
                    padding=2,
                    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                    activator=ActivationType.RELU
                ),
                Pooling(kernel_shape=(2, 2), strides=2),
                Flatten(),
                Dense(n_nodes=1, activator=ActivationType.SIGMOID),
            ]
        )
        model.compile()

    model.summary()

    input_array = []
    target_array = []

    test_input_array = []
    test_target_array = []

    # Load train image
    load_dataset(cat_dir, [CAT_TARGET], input_array, target_array)
    load_dataset(dog_dir, [DOG_TARGET], input_array, target_array)

    # Load test image
    load_dataset(cat_test_dir, [CAT_TARGET], test_input_array, test_target_array)
    load_dataset(dog_test_dir, [DOG_TARGET], test_input_array, test_target_array)

    if args.crossval == '--crossval':
        kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
        input_array = np.array(input_array)
        target_array = np.array(target_array)
        for train, validate in kf.split(input_array):
            train_input = input_array[train]
            train_target = target_array[train]
            validate_input = input_array[validate]
            validate_target = target_array[validate]

            model.fit(
                train_input,
                train_target,
                mini_batch=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                epoch=EPOCH
            )

            count_score(model, validate_input, validate_target, 'fold val data')

        count_score(model, test_input_array, test_target_array, 'test data')

        model.save(MODEL_SAVE_PATH)

    else:
        train_input, validation_input, train_target, validation_target = train_test_split(
            input_array, target_array, test_size=0.1, random_state=RANDOM_STATE
        )

        model.fit(
            train_input,
            train_target,
            mini_batch=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            epoch=EPOCH
        )

        count_score(model, validation_input, validation_target, 'val data')

        count_score(model, test_input_array, test_target_array, 'test data')

        model.save(MODEL_SAVE_PATH)
