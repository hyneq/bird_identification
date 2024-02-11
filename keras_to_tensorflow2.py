#!/usr/bin/env python3

import os
import argparse

import tensorflow as tf

def keras_to_tensorflow(keras_path, tf_path=None):
    if tf_path is None:
        tf_path = os.path.splitext(keras_name[0]) + ".pb"

    # based on https://stackoverflow.com/a/60186155
    model = tf.keras.models.load_model(keras_path)
    tf.saved_model.save(model, tf_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("keras_path", help="Path to Keras model file")
    parser.add_argument("tf_path", nargs="?", default=None, help="Path to Tensorflow model file")
    args = parser.parse_args()

    keras_to_tensorflow(args.keras_path, args.tf_path)