#!/usr/bin/python3

# This code is heavily based on https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

import os, sys
import argparse
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from bird_identification.classification import load_classes

SEED = 13579
IMG_SIZE = 224 # for EfficientNetB0
BATCH_SIZE = 32

@dataclass()
class Dataset:
    ds_train: any
    ds_test: any
    class_names: any

# based on https://keras.io/api/data_loading/
def get_dataset_from_directory(path, validation_split=0.4, class_names=None, batch_size=BATCH_SIZE, seed=SEED, prefetch=True):
    kwargs = {
        'directory': path,
        'labels': 'inferred',
        'batch_size': batch_size,
        'seed': seed,
        'image_size': (IMG_SIZE, IMG_SIZE),
        'validation_split': validation_split,
        'class_names': class_names
    }

    ds_train = keras.utils.image_dataset_from_directory(subset='training', **kwargs)
    ds_test = keras.utils.image_dataset_from_directory(subset='validation', **kwargs)

    class_names = ds_train.class_names

    if prefetch:
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)
        ds_test = ds_train.prefetch(tf.data.AUTOTUNE)

    return Dataset(ds_train, ds_test, class_names)

def get_img_augmentation(rotation_factor=0.15, height_factor=0.1, width_factor=0.1, contrast_factor=0.1):
    return Sequential(
        [
            layers.RandomRotation(factor=rotation_factor),
            layers.RandomTranslation(height_factor=height_factor, width_factor=width_factor),
            layers.RandomFlip(),
            layers.RandomContrast(factor=contrast_factor)
        ],
        name="img_augmentation"
    )

def build_model(model_name, num_classes, img_augmentation=None):
    if img_augmentation is None: img_augmentation = get_img_augmentation()

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = img_augmentation(inputs)
    model = EfficientNetB0(include_top=False, input_tensor=x, weights="imagenet")

    # Freeze the pretrained weights
    model.trainable = False
    
    # Rebuild top
    x = layers.GlobalAveragePooling2D(name="avg_pool")(model.output)
    x = layers.BatchNormalization()(x)

    top_dropout_rate = 0.2
    x = layers.Dropout(top_dropout_rate, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    # Compile
    model = tf.keras.Model(inputs, outputs, name=model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

def build_and_train_model(
        model_name,
        class_names_path=None,
        class_names=None,
        dataset_path=None,
        dataset=None,
        save_path=None,
        strategy=None,
        epochs_top=25,
        epochs_ft=10,
        img_augmentation=None,
        verbose=True,
        show_data=False,
        show_augm=False,
        plot_hist=False
    ):

    # Prepare strategy

    #if strategy is None:
    #    strategy = tf.distribute.MirroredStrategy()


    # Prepare class names

    if (not class_names) and class_names_path:
        class_names = list(load_classes(class_names_path).values())


    # Prepare dataset

    if dataset_path is None and dataset is None:
        raise ValueError("No dataset path nor dataset given, cannot continue")

    if dataset is None:
        dataset = get_dataset_from_directory(dataset_path)
        if verbose: print("Loaded dataset from {}".format(dataset_path))

    if verbose:
        print("Dataset contains {} classes".format(len(dataset.class_names)))

    if show_data: show_dataset(dataset)


    # Prepare image augmentation

    if img_augmentation is None:
        img_augmentation = get_img_augmentation()
    
    if verbose:
        print("Created image augmentation layer")
    
    if show_augm:
        #with strategy.scope():
        show_augmentation(img_augmentation, dataset)


    # Build model based on EfficientNetB0

    model = build_model(model_name, len(class_names),img_augmentation=img_augmentation)

    if verbose:
        print("Built an EfficientNetB0 model with ImageNet weights, with inner layers frozen")


    # Train the top layer on our dataset

    if verbose:
        print("Training top layers with larger learning rate")

    hist_top = model.fit(dataset.ds_train, epochs=epochs_top, validation_data=dataset.ds_test, verbose=2)

    if plot_hist: plot_history(hist_top)


    # Unfreeze and fine-tune the model

    if verbose:
        print("Unfreezing model and fine-tuning top 20 internal layers with smaller learning rate")

    unfreeze_model(model)

    hist_ft = model.fit(dataset.ds_train, epochs=epochs_ft, validation_data=dataset.ds_test, verbose=2)

    if plot_hist: plot_history(hist_ft)


    # Save the model to the specified path
    
    if save_path:
        model.save(save_path)
    
    if verbose:
        print("Saved model to {}".format(save_path))


# based on https://www.tensorflow.org/tutorials/load_data/images
def show_dataset(ds):
    for images, labels in ds.ds_train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(ds.class_names[labels[i]])
            plt.axis("off")

def show_augmentation(img_augmentation, ds):
    for image, label in ds.ds_train.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = img_augmentation(image)
            plt.imshow(aug_img[0].numpy().astype("uint8"))
            plt.title("{}".format(ds.class_names[1]))
            plt.axis("off")

def plot_history(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        prog="Transfer-learning EfficientNetB0",
        description="Creates an EfficientNetB0-based model with ImageNet weights and fine-tunes it on a custom dataset",
        epilog="Most of the code is based on https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/"
    )
    parser.add_argument("model_name", help="name of the model")
    parser.add_argument("class_names_path", help="CSV file containing class names")
    parser.add_argument("dataset_path", help="directory containing the training data, grouped in subdirectories by class names")
    parser.add_argument("save_path", help="save path of the model")
    parser.add_argument("--epochs-top", type=int, default=25, help="number of epochs for training the top layer")
    parser.add_argument("--epochs-ft", type=int, default=10, help="number of epochs for fine-tuning the EfficienNet layers")
    parser.add_argument("--log-level", type=str, default="error", choices=["fatal","error","warn","info","debug"], help="TensorFlow log level")
    parser.add_argument("-d", "--show-data", action='store_true', help="show first 9 images from the dataset")
    parser.add_argument("-a", "--show-augmentation", action='store_true', help="show first 9 variations of first augumented image")
    parser.add_argument("-t", "--plot-history", action='store_true', help="plot charts of training history")

    args=parser.parse_args()

    tf.compat.v1.logging.set_verbosity(args.log_level.upper())

    build_and_train_model(
        args.model_name,
        dataset_path=args.dataset_path,
        class_names_path=args.class_names_path,
        save_path=args.save_path,
        epochs_top=args.epochs_top,
        epochs_ft=args.epochs_ft,
        show_data=args.show_data,
        show_augm=args.show_augmentation,
        plot_hist=args.plot_history
    )

if __name__ == "__main__":
    main()