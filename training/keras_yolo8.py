#!/usr/bin/env python3
"""
Creates a model from a keras_cv.models.YOLOV8Detector preset
"""

import argparse

import keras_cv

DEFAULT_PRESET_NAME = "yolo_v8_m_pascalvoc"
DEFAULT_NUM_CLASSES = 20

def get_yolov8_model(preset_name=DEFAULT_PRESET_NAME, num_classes=DEFAULT_NUM_CLASSES) -> keras_cv.models.YOLOV8Detector:
    """
    Creates a keras_cv.models.YOLOV8Detector from a preset
    """

    return keras_cv.models.YOLOV8Detector.from_preset(preset_name, num_classes=num_classes, bounding_box_format="xywh")


def main():
    """
    Creates a Keras model file from a keras_cv.models.YOLOV8Detector preset
    """

    parser = argparse.ArgumentParser(
        description="Create a model file from a keras_cv.models.YOLOV8Detector preset"
    )
    parser.add_argument(
        "--preset", "-p",
        default=DEFAULT_PRESET_NAME,
        help="The preset name to initialize the model with"
    )
    parser.add_argument(
        "--num-classes", "-n",
        default=DEFAULT_NUM_CLASSES,
        help="The number of classes to initialize the model with",
        type=int
    )
    parser.add_argument(
        "model_path",
        help="The path to model file to save"
    )

    args = parser.parse_args()

    model = get_yolov8_model(preset_name=args.preset, num_classes=args.num_classes)
    model.save(args.model_path)


if __name__ == "__main__":
    main()
