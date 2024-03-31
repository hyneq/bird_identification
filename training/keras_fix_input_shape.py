#!/usr/bin/env python3

"""
Fixes input shape for models with some dimensions not specified

This allows models to be later quantized and compiled for fixed-shape environments
"""

from typing import Optional

import argparse

import keras

# Load keras_cv, if possible, to register their custom objects
try:
    import keras_cv
except ImportError:
    pass


def convert(in_path: str, out_path: list[int], input_shape: Optional[list[int]]=None, batch_size: int=1):
    model = keras.models.load_model(in_path)

    input_shape = input_shape or model.input.shape[1:]

    new_input = keras.Input(shape=input_shape, batch_size=batch_size)

    new_output = model(new_input)

    new_model = keras.Model(name=model.name, inputs=new_input, outputs=new_output)

    new_model.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="path to the Keras model to convert")
    parser.add_argument("out_path", help="path where to save the converted TF Lite model")
    parser.add_argument("-s", "--input_shape", type=int, nargs="+", help="the new input shape")
    parser.add_argument("-z", "--batch_size", type=int, default=1, help="the batch size to set")
    args = parser.parse_args()

    convert(args.in_path, args.out_path, input_shape=args.input_shape, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
