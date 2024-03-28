#!/usr/bin/env python3

"""
Train KerasCV-based YOLOv8 model on a specified dataset

This code is heavily based on https://keras.io/guides/keras_cv/object_detection_keras_cv/#train-a-custom-object-detection-model
and https://keras.io/examples/vision/yolov8/
"""

from dataclasses import dataclass
import os
import sys
import csv
from pprint import pprint
import argparse

import tensorflow as tf
from tensorflow import keras
import keras_cv
from keras_cv import visualization
from keras_cv import bounding_box

import cv2
import numpy as np

from matplotlib import pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from bird_identification.prediction.classes import ClassNames

SPLIT_RATIO = 0.2
BATCH_SIZE = 4
LEARNING_RATE = 0.001
GLOBAL_CLIPNORM = 10.0
TARGET_SIZE = (640, 640)
MODEL_BACKBONE = "resnet50_imagenet"
BOUNDING_BOX_FORMAT = "xywh"
DATASET_BOUNDING_BOX_FORMAT = "rel_xywh"

@dataclass
class Dataset:
    train: tf.data.Dataset
    val: tf.data.Dataset
    class_names: tf.data.Dataset


def visualize_dataset(inputs, class_names, value_range=(0, 255), rows=2, cols=2, bounding_box_format=BOUNDING_BOX_FORMAT):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_names,
    )
    plt.show()


def plot_history(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def load_image(image_path):
    """
    Loads an image

    Based on https://keras.io/examples/vision/yolov8/
    """

    image = tf.io.read_file(image_path)
    return tf.image.decode_jpeg(image, channels=3)


def parse_YOLO_annotations(annot_path: str):
    """
    Parses a YOLO annotation file

    Based on https://keras.io/examples/vision/yolov8/
    """

    boxes: list[tuple] = []
    classes: list[int] = []

    with open(annot_path, newline="", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter=" ")
        for row in reader:

            # Convert bounding box coords from center to top-left
            box_center_x, box_center_y, box_w, box_h = tuple(float(val) for val in row[1:5])
            box_x = box_center_x - box_w/2
            box_y = box_center_y - box_h/2
            boxes.append((box_x, box_y, box_w, box_h))

            classes.append(int(row[0]))

    return boxes, classes


def load_YOLO_dataset(annots_dir: str, imgs_dir: str, class_names_path: str, split_ratio: float=SPLIT_RATIO):
    """
    Loads a dataset in YOLO format

    Based on https://keras.io/examples/vision/yolov8/
    """

    image_paths = []
    boxes = []
    classes = []
    for annot_file in filter(lambda p: p.endswith(".txt"), os.listdir(annots_dir)):
        basename = os.path.splitext(annot_file)[0]
        found_img = False
        for ext in [".jpg", ".jpeg"]:
            img_path = os.path.join(imgs_dir, basename + ext)
            if os.path.exists(img_path):
                found_img = True
                break

        if not found_img:
            continue

        annot_boxes, annot_classes = parse_YOLO_annotations(os.path.join(annots_dir, annot_file))

        image_paths.append(img_path)
        boxes.append(annot_boxes)
        classes.append(annot_classes)

    n_images = len(image_paths)

    image_paths = tf.ragged.constant(image_paths)
    boxes = tf.ragged.constant(boxes)
    classes = tf.ragged.constant(classes)
    data = tf.data.Dataset.from_tensor_slices((image_paths, classes, boxes))

    num_val = int(n_images * split_ratio)

    val_data = data.take(num_val)
    train_data = data.skip(num_val)

    def load_dataset(image_path, classes, boxes):
        image = load_image(image_path)
        boxes = keras_cv.bounding_box.convert_format(
            boxes.to_tensor(),
            images=image,
            source=DATASET_BOUNDING_BOX_FORMAT,
            target=BOUNDING_BOX_FORMAT
        )
        bounding_boxes = {
            "classes": tf.cast(classes, dtype=tf.float32),
            "boxes": boxes,
        }
        return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}

    train_ds = train_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_data.map(load_dataset, num_parallel_calls=tf.data.AUTOTUNE)

    class_names = ClassNames.load_from_file(class_names_path)

    return Dataset(
        train=train_ds,
        val=val_ds,
        class_names=dict(enumerate(class_names.class_names))
    )


def get_augmentation():
    """
    Returns an augmentation layer

    Based on https://keras.io/guides/keras_cv/object_detection_keras_cv/#data-augmentation
    and 
    """

    return keras.Sequential(
        layers=[
            keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format=BOUNDING_BOX_FORMAT),
            keras_cv.layers.RandomFlip(mode="vertical", bounding_box_format=BOUNDING_BOX_FORMAT),
            #keras_cv.layers.RandomShear( # corrupts bounding box locations
            #    x_factor=0.2, y_factor=0.2, bounding_box_format=BOUNDING_BOX_FORMAT
            #),
            keras_cv.layers.JitteredResize(
                target_size=TARGET_SIZE, scale_factor=(0.75, 1.3), bounding_box_format=BOUNDING_BOX_FORMAT
            ),
        ],
        name="augmentation"
    )


def get_resize():
    """
    Returns a resize layer

    Based on https://keras.io/guides/keras_cv/object_detection_keras_cv/#data-augmentation
    """

    return keras_cv.layers.Resizing(
        *TARGET_SIZE, bounding_box_format=BOUNDING_BOX_FORMAT, pad_to_aspect_ratio=True
    )


def input_tuple(inputs):
    """
    Convert input to format suitable for model

    Based on https://keras.io/guides/keras_cv/object_detection_keras_cv/#data-augmentation
    """


    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )


def prepare_input_data(dataset: Dataset, take=None, show_data=False):
    """
    Prepares input data, given a dataset
    """

    train_ds = dataset.train
    val_ds = dataset.val
    class_names = dataset.class_names

    train_ds = train_ds.shuffle(BATCH_SIZE * 4)
    train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    if show_data:
        visualize_dataset(
            train_ds,
            value_range=(0, 255),
            rows=2,
            cols=2,
            class_names=class_names
        )

    val_ds = val_ds.shuffle(BATCH_SIZE * 4)
    val_ds = val_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
    if show_data:
        visualize_dataset(val_ds, class_names=class_names)

    train_ds = train_ds.map(get_augmentation(), num_parallel_calls=tf.data.AUTOTUNE)
    if show_data:
        visualize_dataset(train_ds, class_names=class_names)

    val_ds = val_ds.map(get_resize(), num_parallel_calls=tf.data.AUTOTUNE)
    if show_data:
        visualize_dataset(train_ds, class_names=class_names)

    train_ds = train_ds.map(input_tuple, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(input_tuple, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    if take:
        train_ds = train_ds.take(take)
        val_ds = val_ds.take(take)

    return train_ds, val_ds


def get_model(num_classes):
    """
    Returns a YOLOv8Detector model
    """

    return keras_cv.models.YOLOV8Detector.from_preset(
        MODEL_BACKBONE,
        bounding_box_format=BOUNDING_BOX_FORMAT,
        num_classes=num_classes,
    )


def get_optimizer():
    """
    Returns a SGD optimizer
    """

    return keras.optimizers.legacy.SGD(
        learning_rate=LEARNING_RATE, momentum=0.9, global_clipnorm=GLOBAL_CLIPNORM
    )


def get_coco_metrics(val_ds):
    """
    Returns a PyCOCOCallback
    """

    return keras_cv.callbacks.PyCOCOCallback(
        val_ds, bounding_box_format=BOUNDING_BOX_FORMAT
    )

def train_model(
    model,
    train_ds,
    val_ds=None,
    epochs=1,
    optimizer=None,
    callbacks=None,
    verbose=True
):
    """
    Trains a model on a specified dataset
    """

    optimizer = optimizer or get_optimizer()

    callbacks = callbacks or [get_coco_metrics(val_ds)]

    # Compile model for training
    model.compile(
        classification_loss="binary_crossentropy",
        box_loss="ciou",
        optimizer=optimizer,
    )

    # Train on the dataset
    hist = model.fit(
        train_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    if verbose:
        pprint(hist.history)

    #if plot_hist:
    #    plot_history(hist)


def build_and_train_model(
        class_names_path,
        dataset_annots_dir=None,
        dataset_imgs_dir=None,
        dataset=None,
        save_path=None,
        take=None,
        epochs=25,
        verbose=True,
        show_data=False
):
    """
    Sets up the training pipeline and trains the model
    on a specified dataset

    Based on https://github.com/hyneq/bird_identification/blob/main/training/train_keras_classification.py
    """

    # Prepare dataset
    if not (dataset_annots_dir and dataset_imgs_dir) or dataset:
        raise ValueError("No dataset path or dataset given, cannot continue")

    if dataset is None:
        dataset = load_YOLO_dataset(
            dataset_annots_dir,
            dataset_imgs_dir,
            class_names_path,
        )
        if verbose:
            print(f"Loaded annotations from {dataset_annots_dir}")
            print(f"Loaded images from {dataset_imgs_dir}")

    num_classes = len(dataset.class_names)

    if verbose:
        print(f"Dataset contains {len(dataset.class_names)} classes")


    # Prepare data from the dataset
    train_ds, val_ds = prepare_input_data(dataset, take=take, show_data=show_data)


    # Get model
    model = get_model(num_classes)


    # Train model
    train_model(
        model,
        train_ds,
        val_ds,
        epochs=epochs,
        verbose=verbose
    )

    # Save the model
    if save_path:
        model.save(save_path)

        if verbose:
            print(f"Saved model to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Creates an EfficientNetB0-based model with ImageNet weights and fine-tunes it on a custom dataset",
        epilog="Most of the code is based on https://keras.io/guides/keras_cv/object_detection_keras_cv"
    )
    parser.add_argument("class_names_path", help="CSV file containing class names")
    parser.add_argument("dataset_annots_dir", help="directory containing the dataset annotations")
    parser.add_argument("dataset_imgs_dir", help="directory containing the dataset images")
    parser.add_argument("save_path", help="save path of the model")
    parser.add_argument("--epochs",
        type=int, default=25, help="number of epochs for training the model"
    )
    parser.add_argument("--take",
        type=int, required=False, help="Take only a specific number of batches from the dataset"
    )
    parser.add_argument("-d", "--show-data",
        action='store_true', help="show first data from the dataset in various preprocess phases"
    )

    args=parser.parse_args()

    build_and_train_model(
        class_names_path=args.class_names_path,
        dataset_annots_dir=args.dataset_annots_dir,
        dataset_imgs_dir=args.dataset_imgs_dir,
        save_path=args.save_path,
        epochs=args.epochs,
        take=args.take,
        show_data=args.show_data
    )

if __name__ == "__main__":
    main()
