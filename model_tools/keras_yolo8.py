#!/usr/bin/env python3
"""
Creates a model from a keras_cv.models.YOLOV8Detector preset
"""

import argparse

import keras_cv

DEFAULT_PRESET_NAME = "yolo_v8_m_pascalvoc"

def get_yolov8_model(preset_name=DEFAULT_PRESET_NAME) -> keras_cv.models.YOLOV8Detector:
    """
    Creates a keras_cv.models.YOLOV8Detector from a preset
    """

    return keras_cv.models.YOLOV8Detector.from_preset(preset_name)


def main():
    """
    Creates a Keras model file from a keras_cv.models.YOLOV8Detector preset
    """

    parser = argparse.ArgumentParser(
        description="Create a model file from a keras_cv.models.YOLOV8Detector preset"
    )
    parser.add_argument(
        "--preset-name",
        default=DEFAULT_PRESET_NAME,
        help="The preset name to initialize the model with"
    )
    parser.add_argument(
        "model_path",
        help="The path to model file to save"
    )

    args = parser.parse_args()

    model = get_yolov8_model(preset_name=args.preset_name)
    model.save(args.model_path)


if __name__ == "__main__":
    main()
