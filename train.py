#!/usr/bin/python3

# This code is heavily based on https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential

def img_augumentation(rotation_factor=0.15, height_factor=0.1, width_factor=0.1, contrast_factor=0.1):
    return Sequential(
        [
            layers.RandomRotation(factor=rotation_factor),
            layers.RandomTranslation(height_factor=height_factor, width_factor=width_factor),
            layers.RandomFlip(),
            layers.RandomContrast(factor=contrast_factor)
        ],
        name="img_augumentation"
    )

