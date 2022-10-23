#!/usr/bin/python3

# This code is heavily based on https://medium.com/analytics-vidhya/object-detection-using-yolov3-d48100de2ebb

import os, sys

import cv2
import numpy as np

model_path = os.path.join(os.path.dirname(__file__),"models","YOLOv3-COCO")

config_path = os.path.join(model_path, "yolov3.cfg")
weights_path = os.path.join(model_path, "yolov3.weights")
labels_path = os.path.join(model_path, "coco.names")

def load_model(config_path, weights_path, labels_path):
    with open(labels_path) as f:
        # Getting labels reading every line
        # and putting them into the list
        labels = [line.strip() for line in f]
    
    network = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    # Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

    # Getting only output layers' names that we need from YOLO v3 algorithm
    # with function that returns indexes of layers with unconnected outputs
    layers_names_output = \
        [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]
    
    return (network, layers_names_output, labels)


def detect_objects(image, network, layer_names_output, labels):
    if type(image) == str:
        image = cv2.imread(image)
    
    probability_minimum = 0.5

    # blob from image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    
    network.setInput(blob)

    output = network.forward(layer_names_output)

    # Going through all output layers after feed forward pass
    for result in output:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            if confidence_current > probability_minimum:
                print(labels[class_current], confidence_current)
        
if __name__ == "__main__":
    model = load_model(config_path, weights_path, labels_path)
    detect_objects(sys.argv[1], *model)