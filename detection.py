#!/usr/bin/python3

# This code is heavily based on https://medium.com/analytics-vidhya/object-detection-using-yolov3-d48100de2ebb

import os, sys, glob, re
from dataclasses import dataclass

import cv2
import numpy as np

model_path = os.path.join(os.path.dirname(__file__),"models","YOLOv3-COCO")

config_path = os.path.join(model_path, "yolov3.cfg")
weights_path = os.path.join(model_path, "yolov3.weights")
labels_path = os.path.join(model_path, "coco.names")

@dataclass()
class Model:
    network: any
    layer_names_output: any
    labels: any

@dataclass()
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

@dataclass()
class Result:
    label: str
    bounding_box: BoundingBox
    confidence: any


def load_model(config_path=config_path, weights_path=weights_path, labels_path=labels_path):
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
    
    return Model(network, layers_names_output, labels)


def detect_objects(image, model=None):
    if model is None:
        model = load_model()


    if type(image) == str:
        image = cv2.imread(image)

    height, width = image.shape[0:2]
    
    probability_minimum = 0.5

    threshold = 0.3

    # blob from image
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    
    model.network.setInput(blob)

    output = model.network.forward(model.layer_names_output)

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

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
                #print(labels[class_current], confidence_current)

                box_current = detected_objects[0:4] * np.array([width, height, width, height])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)


    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    filtered = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                            probability_minimum, threshold)
    
    results = [Result(model.labels[class_numbers[i]], BoundingBox(*bounding_boxes[i]), confidences[i]) for i in filtered]

    return results

if __name__ == "__main__":
    results = detect_objects(sys.argv[1])

    for result in results:
        print("{} at {} with {} % confidence".format(result.label, result.bounding_box, round(result.confidence*100,2)))