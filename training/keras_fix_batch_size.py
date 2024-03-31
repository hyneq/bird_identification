#!/usr/bin/env python3

"""
Fixes batch size for models with unspecified batch size
"""

from typing import Optional

import sys
import argparse

import keras

def convert(in_path: str, out_path: str, batch_size: int=1):
    model = keras.models.load_model(in_path)

    new_inputs = []
    for input in model.inputs:
        new_inputs.append(keras.Input(shape=input.shape[1:], batch_size=batch_size))

    new_outputs = model(new_inputs)

    new_model = keras.Model(inputs=new_inputs, outputs=new_outputs)

    new_model.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="path to the Keras model to convert")
    parser.add_argument("out_path", help="path where to save the converted TF Lite model")
    parser.add_argument("-s", "--batch_size", type=int, default=1, help="The batch size to set")
    args = parser.parse_args()

    convert(args.in_path, args.out_path, batch_size=args.batch_size)


if __name__ == "__main__":
    main()