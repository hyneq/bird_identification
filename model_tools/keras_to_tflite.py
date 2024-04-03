#!/usr/bin/env python3

from typing import Optional

import argparse

import tensorflow as tf

# Load keras_cv, if possible, to register their custom objects
try:
    import keras_cv
except ImportError:
    pass

def convert(in_path: str, out_path: str, *args, optimization: Optional[str]=None, **kwargs):
    """
    Converts a Keras model to a TF Lite model
    """

    keras_model = tf.keras.models.load_model(in_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    if optimization:
        OPTIMIZATIONS[optimization](converter, keras_model, *args, **kwargs)

    tflite_model = converter.convert()

    with open(out_path, 'wb') as f:
        f.write(tflite_model)


def optimization_dynamic_range_quant(converter: tf.lite.TFLiteConverter, keras_model: tf.keras.Model, *_, **__):

    # from https://www.tensorflow.org/lite/performance/post_training_quant
    converter.optimizations = [tf.lite.Optimize.DEFAULT]


def optimization_float_fallback_quant(converter: tf.lite.TFLiteConverter, keras_model: tf.keras.Model, representative_data_path: str, *args, **kwargs):

    # from https://www.tensorflow.org/lite/performance/post_training_integer_quant

    optimization_dynamic_range_quant(converter, keras_model, *args, **kwargs)

    image_size = keras_model.input.shape[1:3]
    def representative_data_gen():
        for input_value in (
            tf.keras.preprocessing.image_dataset_from_directory(
                representative_data_path, labels=None, batch_size=1, image_size=image_size
            ).take(100)
        ):
            yield [input_value]

    converter.representative_dataset = representative_data_gen


def optimization_integer_only_quant(converter: tf.lite.TFLiteConverter, *args, **kwargs):

    # from https://www.tensorflow.org/lite/performance/post_training_integer_quant

    optimization_float_fallback_quant(converter, *args, **kwargs)

    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8


OPTIMIZATIONS = {name.removeprefix("optimization_"): obj for name, obj in globals().items() if name.startswith("optimization_")}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("in_path", help="path to the Keras model to convert")
    parser.add_argument("out_path", help="path where to save the converted TF Lite model")
    parser.add_argument("-o", "--optimization",
        help="The optimization to use", choices=OPTIMIZATIONS.keys(), required=False
    )
    parser.add_argument("-d", "--representative-data-path", help="The path to directory with representative data")
    args = parser.parse_args()

    convert(args.in_path, args.out_path,
        optimization=args.optimization, representative_data_path=args.representative_data_path
    )


if __name__ == "__main__":
    main()
