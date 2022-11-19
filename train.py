#!/usr/bin/python3

# This code is heavily based on https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning/

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.applications import EfficientNetB0
import matplotlib as plt

import argparse

from classification import load_classes

IMG_SIZE = 224 # for EfficientNetB0

def get_class_name(dataset, i):
    return dataset.class_names[i]

# based on https://keras.io/api/data_loading/
def get_dataset_from_directory(path, validation_split=0.4, class_names=None):
    return keras.utils.image_dataset_from_directory(
        directory=path,
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        image_size=(IMG_SIZE, IMG_SIZE),
        validation_split=validation_split,
        subset="both",
        class_names=class_names
    )

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
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model

def unfreeze_model(model):
    # We unfreeze the top 20 layers while leaving BatchNorm layers frozen
    for layer in model.layers[-20:]:
        if not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
    )

def build_and_train_model(
        model_name,
        class_names_path=None,
        class_names=None,
        dataset_path=None,
        dataset=None,
        save_path=None,
        epochs_top=25,
        epochs_ft=10,
        img_augmentation=None,
        verbose=2,
        show_data=False,
        show_augm=False,
        plot_hist=False
    ):

    # Prepare class names
    if not class_names and class_names_path:
        class_names = list(load_classes().values())

    # Prepare dataset

    if dataset_path is None and dataset is None:
        raise ValueError("No dataset path nor given, cannot continue")

    if dataset is None:
        dataset = get_dataset_from_directory(dataset_path)
        if verbose: print("Loaded dataset from {}".format(dataset_path))
    
    if verbose:
        print("Dataset contains {} classes".format(len(dataset.class_names)))
    
    (ds_train, ds_test) = dataset

    if show_data: show_dataset(ds_train)


    # Prepare image augmentation

    if img_augmentation is None:
        img_augmentation = get_img_augmentation()
    
    if verbose:
        print("Created image augmentation layer")
    
    if show_augm: show_augmentation(img_augmentation, ds_train)


    # Build model based on EfficientNetB0

    model = build_model(model_name, len(dataset.class_names),img_augmentation=img_augmentation)

    if verbose:
        print("Built an EfficientNetB0 model with ImageNet weights, with inner layers frozen")


    # Train the top model on our dataset

    if verbose:
        print("Training top layers with higher learning rate")

    hist_top = model.fit(ds_train, epochs=epochs_top, validation_data=ds_test, verbose=verbose)

    if plot_hist: plot_history(hist_top)


    # Unfreeze and fine-tune the model

    if verbose:
        print("Unfreezing model and fine-tuning top 20 internal layers")

    unfreeze_model(model)

    hist_ft = model.fit(ds_train, epochs=epochs_ft, validation_data=ds_test, verbose=verbose)

    if plot_hist: plot_history(hist_ft)


    # Save the model to the specified path
    
    if save_path:
        model.save(save_path)
    
    if verbose:
        print("Saved model to {}".format(save_path))


def show_dataset(ds):
    for i, (image, label) in enumerate(ds.take(9)):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title("{}".format(get_class_name(ds, label)))
        plt.axis("off")

def show_augmentation(img_augmentation, ds):
    for image, label in ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            aug_img = img_augmentation(tf.expand_dims(image, axis=0))
            plt.imshow(aug_img[0].numpy().astype("uint8"))
            plt.title("{}".format(get_class_name(ds, label)))
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
    parser.add_argument("dataset_path", help="directory containing the training data, grouped in subdirectories by class names")
    parser.add_argument("class_name_path", help="CSV file containing class names")
    parser.add_argument("save_path", help="save path of the model")
    parser.add_argument("--epochs-top", type=int, default=25, help="number of epochs for training the top layer")
    parser.add_argument("--epochs-ft", type=int, default=10, help="number of epochs for fine-tuning the EfficienNet layers")
    parser.add_argument("-v", "--verbose", action='count', help="the level of output verbosity")
    parser.add_argument("-d", "--show-data", action='store_true', help="show first 9 images from the dataset")
    parser.add_argument("-a", "--show-augmentation", action='store_true', help="show first 9 variations of first augumented image")
    parser.add_argument("-t", "--plot-history", action='store_true', help="plot charts of training history")

    args=parser.parse_args()

    build_and_train_model(
        args.model_name,
        dataset_path=args.dataset_path,
        class_names_path=args.class_names_path,
        save_path=args.save_path,
        epochs_top=args.epochs_top,
        epochs_ft=args.epochs_ft,
        verbose=args.verbose,
        show_data=args.show_data,
        show_augm=args.show_augmentation,
        plot_hist=args.plot_history
    )

if __name__ == "__main__":
    main()