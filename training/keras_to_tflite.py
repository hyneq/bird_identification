#!/usr/bin/env python3

import argparse

import tensorflow as tf

def convert(in_path: str, out_path: str, quantize: bool=False, ):
    """
    Converts a Keras model to a TF Lite model
    """

    keras_model = tf.keras.models.load_model(in_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(out_path, 'wb') as f:
        f.write(tflite_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="path to the Keras model to convert")
    parser.add_argument("out_path", help="path where to save the converted TF Lite model")
    parser.add_argument("-q", "--quantize", action="store_true",
        help="quantize the converted model"
    )
    args = parser.parse_args()

    convert(args.in_path, args.out_path, quantize=args.quantize)


if __name__ == "__main__":
    main()
