#!/usr/bin/python3

import os,sys
import csv
from dataclasses import dataclass

import tensorflow as tf
from tensorflow import keras
import numpy as np
#import PIL

@dataclass()
class Model:
    model: any
    class_names: list

class Result:
    class_name: str
    confidence: float

model_path = os.path.join(os.path.dirname(__file__),"models","jizbirds","jizbirds.h5")
classes_path = os.path.join(os.path.dirname(__file__),"models","jizbirds","classes.csv")

# based on the code from https://prasanshasatpathy.medium.com/deploying-image-classification-model-using-the-saved-model-in-the-format-of-tflite-file-and-h5-file-92bcaf299181
def predict(img_path):

    classes = load_classes(classes_path)

    model = keras.models.load_model(model_path)

    image = keras.utils.load_img(img_path, target_size=(224,224))
    blob = keras.utils.img_to_array(image)
    blob = np.expand_dims(blob,0)

    # image = PIL.Image.open(img_path)
    # image = image.resize((200,200))
    # blob = np.asarray(image)
    # blob = np.expand_dims(blob, axis=0)

    predictions = model.predict(blob)
    scores = predictions[0]

    return (classes[np.argmax(scores)], np.max(scores)*100)


def load_classes(classes_path):
    with open(classes_path, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        return {int(row[0]): row[1] for row in reader}

if __name__ == "__main__":
    (class_name, confidence) = predict(sys.argv[1])
    print("{} ({}% confidence)".format(class_name, confidence.round(3)))
