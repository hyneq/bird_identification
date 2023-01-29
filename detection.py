#!/usr/bin/python3

# This code is heavily based on https://medium.com/analytics-vidhya/object-detection-using-yolov3-d48100de2ebb

import os, sys, glob, re
from typing import Optional, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod, ABCMeta
from threading import Lock

import cv2
import numpy as np

import prediction
from classes import ClassSelector, DEFAULT_CLASS_SELECTOR, ClassificationMode, get_class_selector

YOLO_COCO_PATH = os.path.join(os.path.dirname(__file__),"models","YOLOv3-COCO")

DEFAULT_PROBABILITY_MINIMUM = 0.5
DEFAULT_NMS_THRESHOLD = 0.5

@dataclass()
class DetectionModelConfig(prediction.PredictionModelConfig):
    config_path: str
    weights_path: str

DEFAULT_MODEL_CONFIG = DetectionModelConfig(
    classes_path=os.path.join(YOLO_COCO_PATH, "coco.names"),
    config_path=os.path.join(YOLO_COCO_PATH, "yolov3.cfg"),
    weights_path=os.path.join(YOLO_COCO_PATH, "yolov3.weights")
)

@dataclass()
class DetectionModelOutput:
    raw_output: np.ndarray
    width: int
    height: int

    def get_box(self, obj: np.ndarray):
        box = obj[0:4] * np.array([self.width, self.height, self.width, self.height])

        # From YOLO data format, we can get top left corner coordinates
        # that are x_min and y_min
        x_center, y_center, box_width, box_height = box
        x_min = int(x_center - (box_width / 2))
        y_min = int(y_center - (box_height / 2))

        return [x_min, y_min, int(box_width), int(box_height)]
    
    def get_scores(self, obj: np.ndarray):
        return obj[5:]
    
    def __iter__(self):
        return DetectionModelOutputIter(self.raw_output)    

class DetectionModelOutputIter:
    __slots__: tuple

    raw_output: tuple[np.ndarray, np.ndarray, np.ndarray]

    _layer_len: int
    _layer_index: int = 0

    _object_len: int
    _object_index: int = 0


    def __init__(self, raw_output: np.ndarray):
        self.raw_output = raw_output

        self._layer_len = len(raw_output)
        self._object_len = raw_output[0].shape[0]
    
    def __next__(self) -> np.ndarray:
        if self._layer_index < self._layer_len:
            obj = self.raw_output[self._layer_index][self._object_index]

            self._object_index += 1
            if self._object_index == self._object_len:
                self._layer_index += 1
                self._object_index = 0
        
            return obj
        
        raise StopIteration

class DetectionModel(prediction.PredictionModel[DetectionModelConfig, np.ndarray, DetectionModelOutput]):
    __slots__: tuple

    network: any
    layer_names_output: any
    labels: any
    lock: Lock

    def __init__(self, cfg: DetectionModelConfig):
        self.network = network = cv2.dnn.readNetFromDarknet(cfg.config_path, cfg.weights_path)

        # Getting list with names of all layers from YOLO v3 network
        layers_names_all = network.getLayerNames()

        # Getting only output layers' names that we need from YOLO v3 algorithm
        # with function that returns indexes of layers with unconnected outputs
        self.layers_names_output = \
            [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

        super().__init__(cfg)
    
    def predict(self, input: np.ndarray) -> DetectionModelOutput:
        height, width = input.shape[0:2]

        # blob from image
        blob = cv2.dnn.blobFromImage(input, 1 / 255.0, (416, 416),
                                    swapRB=True, crop=False)
        
        self.network.setInput(blob)
        raw_output = self.network.forward(self.layers_names_output)

        return DetectionModelOutput(raw_output, width, height)

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

class DetectionProcessor(prediction.PredictionProcessorWithCS[DetectionModel, DetectionModelOutput, Result]):
    __slots__: tuple

    NMS_threshold = DEFAULT_NMS_THRESHOLD

    # Lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes: list
    confidences: list
    classes: list
    
    def add_detected_object(self, bounding_box, confidence, class_number):
        self.bounding_boxes.append(bounding_box)
        self.confidences.append(confidence)
        self.classes.append(class_number)

    def process_object(self, obj):

        scores = self.output.get_scores(obj)

        classes = self.cs.get_filtered_classes(scores)

        if len(classes):
            box = self.output.get_box(obj)

            # Adding results into prepared lists
            self.add_detected_object(
                box,
                scores[classes],
                classes
            )
    
    def NMSBoxes(self):
        # Implementing non-maximum suppression of given bounding boxes
        # With this technique we exclude some of bounding boxes if their
        # corresponding confidences are low or there is another
        # bounding box for this region with higher confidence

        return cv2.dnn.NMSBoxes(self.bounding_boxes, [confidence[0] for confidence in self.confidences], self.cs.min_confidence, self.NMS_threshold)
    
    def get_results(self, filtered):
        return [Result(self.model.class_names.get_names(self.classes[i]), BoundingBox(*self.bounding_boxes[i]), self.confidences[i]) for i in filtered]
    
    def process(self):
        self.bounding_boxes = []
        self.confidences = []
        self.classes = []

        for obj in self.output:
            self.process_object(obj)

        filtered = self.NMSBoxes()

        return self.get_results(filtered)

class ObjectDetector(prediction.PredictorWithCS[DetectionModel, DetectionModelConfig, DetectionProcessor, np.ndarray, Result]):
    __slots__: tuple

    model_cls = DetectionModel

    prediction_processor = DetectionProcessor

class FileObjectDetector(prediction.FileImagePredictor[ObjectDetector]):
    __slots__: tuple

    predictor_cls = ObjectDetector

get_object_detector = prediction.get_predictor_factory(
        "get_object_detector",
        ObjectDetector,
        DetectionModel,
        DetectionModelConfig,
        DEFAULT_MODEL_CONFIG
    )

def detect_objects(
        images: Union[list[str], list[np.ndarray], str, np.ndarray],
        *args,
        detector: Optional[Union[type[ObjectDetector],ObjectDetector]]=None,
        **kwargs
    ):

    if type(images) is not list:
        images = [images]
    
    if not detector:
        if type(images[0]) is str:
            detector = FileObjectDetector
        else:
            detector = ObjectDetector
    
    detector_type = type(detector)
    if detector_type is type or detector_type is ABCMeta:
        detector = get_object_detector(*args, predictor=detector, **kwargs)
    
    return [detector.predict(image) for image in images]

if __name__ == "__main__":
    results = detect_objects(sys.argv[1])

    for result in results[0]:
        print("{} at {} with {} % confidence".format(result.label, result.bounding_box, np.round(result.confidence*100,2)))