#!/usr/bin/python3

# based on the code from https://prasanshasatpathy.medium.com/deploying-image-classification-model-using-the-saved-model-in-the-format-of-tflite-file-and-h5-file-92bcaf299181

import os,sys
import csv
from typing import Optional, Union
from dataclasses import dataclass
from threading import Lock
from abc import ABCMeta
import argparse
from enum_actions import enum_action

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL.Image import Image, open as open_image

import prediction
from classes import ClassSelector, DEFAULT_CLASS_SELECTOR, ClassificationMode, get_class_selector

@dataclass()
class ClassificationModelConfig(prediction.PredictionModelConfig):
    model_path: str

    @classmethod
    def from_dir(cls, path: str):
        return cls(model_path=os.path.join(path, "model.h5"), classes_path=os.path.join(path, "classes.csv"))

class ClassificationModel(prediction.PredictionModel[ClassificationModelConfig, Image, np.ndarray]):
    __slots__: tuple

    model: keras.Model
    model_lock: Lock

    def __init__(self, cfg: ClassificationModelConfig):
        self.model = keras.models.load_model(cfg.model_path)
        self.model_lock = Lock()
        super().__init__(cfg)
    
    def predict(self, input: Image) -> np.ndarray:
        image = input.resize((224,224))
        blob = keras.utils.img_to_array(image)
        blob = np.expand_dims(blob,0)

        with self.model_lock:
            predictions = self.model.predict(blob)
        
        return predictions[0]
    

@dataclass
class Result:
    class_name: str
    confidence: float

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "czbirds")

DEFAULT_MODEL_CONFIG = ClassificationModelConfig.from_dir(DEFAULT_MODEL_PATH)

class ClassificationProcessor(prediction.PredictionProcessorWithCS[np.ndarray, Result]):
    __slots__: tuple

    def get_results(self, classes) -> list:
        return [Result(class_, self.scores[class_]) for class_ in classes]

    def process(self) -> list:
        self.scores = self.output

        classes = self.cs.get_filtered_classes(self.scores)
        
        return self.get_results(classes)

class ImageClassifier(prediction.PredictorWithCS[ClassificationModel, ClassificationModelConfig, ClassificationProcessor, Image, Result]):
    __slots__: tuple

    model_cls = ClassificationModel

    prediction_processor = ClassificationProcessor

class FileImageClassifier(prediction.FileImagePredictor[ImageClassifier]):
    __slots__: tuple

    predictor_cls = ImageClassifier

get_image_classifier = prediction.get_predictor_factory(
    "get_image_classifier",
    ImageClassifier,
    ClassificationModel,
    ClassificationModelConfig,
    DEFAULT_MODEL_CONFIG
)

def classify_images(
        images: Union[list[str], list[Image], str, Image],
        *args,
        classifier: Optional[Union[type[ImageClassifier],ImageClassifier]]=None,
        **kwargs
    ):

    if type(images) is not list:
        images = [images]
    
    if not classifier:
        if type(images[0]) is str:
            classifier = FileImageClassifier
        else:
            classifier = ImageClassifier
    
    classifier_type = type(classifier)
    if classifier_type is type or classifier_type is ABCMeta:
        classifier = get_image_classifier(*args, predictor=classifier, **kwargs)
    
    return [classifier.predict(image) for image in images]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--mode", action=enum_action(ClassificationMode), help="The classification mode to use")
    parser.add_argument("--min-confidence", type=int, choices=range(0,100), help="Minimum confidence to find")
    parser.add_argument("-c", "--class", type=str, nargs='*', help="class(es) to find", dest="classes")
    parser.add_argument("images", type=str, nargs='+', help="The image file(s) to process")

    args = parser.parse_args()

    results = classify_images(
        images=args.images,
        mode=args.mode,
        min_confidence=args.min_confidence,
        classes=args.classes,
    )

    for result in results:
        print(result)

if __name__ == "__main__":
    main()
