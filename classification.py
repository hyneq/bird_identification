#!/usr/bin/python3

# based on the code from https://prasanshasatpathy.medium.com/deploying-image-classification-model-using-the-saved-model-in-the-format-of-tflite-file-and-h5-file-92bcaf299181

import os,sys
import csv
from typing import Optional, Union
from dataclasses import dataclass
from threading import Lock
from abc import ABC, abstractmethod
import argparse
from enum_actions import enum_action

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL.Image import Image, open as open_image

from abstract_classification import AbstractClassificationProcessor as ACP, DEFAULT_ACP, ClassificationMode, get_acp

@dataclass()
class ClassificationModel:
    model: any
    class_names: list[str]
    model_lock: Lock

@dataclass()
class ClassificationModelConfig:
    model_path: str
    classes_path: str

    @classmethod
    def from_dir(cls, path: str):
        return cls(model_path=os.path.join(path, "model.h5"), classes_path=os.path.join(path, "classes.csv"))

@dataclass
class Result:
    class_name: str
    confidence: float

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "czbirds")

DEFAULT_MODEL_CONFIG = ClassificationModelConfig.from_dir(DEFAULT_MODEL_PATH)

class ClassificationProcessor:
    __slots__ = [
        "output"
    ]

    acp: ACP

    def __init__(self, output):
        self.output = output

    def get_results(self, classes) -> list:
        return [Result(class_, self.scores[class_]) for class_ in classes]

    def process(self) -> list:
        self.scores = self.output

        classes = self.acp.get_filtered_classes(self.scores)
        
        return self.get_results(classes)
    
    @classmethod
    def _class_copy(cls):
        class ClassificationProcessorWithACP(cls):
            pass
        
        return ClassificationProcessorWithACP
    
    @classmethod
    def with_acp(cls, acp: ACP):
        cls = cls._class_copy()
        cls.acp = acp

        return cls

class ImageClassifier:
    __slots__ = [
        "classification_processor",
        "model"
    ]

    classification_processor_cls: type[ClassificationProcessor] = ClassificationProcessor

    classification_processor: ClassificationProcessor

    model: ClassificationModel

    def __init__(
            self, model_config: ClassificationModelConfig=DEFAULT_MODEL_CONFIG,
            model: Optional[ClassificationModel]=None,
            acp: Optional[ACP]=None,
        ):

        if not model:
            model = self.load_model(model_config)

        self.model = model

        if not acp:
            acp = DEFAULT_ACP()

        self.classification_processor = self.classification_processor_cls.with_acp(acp)

    @classmethod
    def load_model(cls, cfg: ClassificationModelConfig) -> ClassificationModel:
        return ClassificationModel(keras.models.load_model(cfg.model_path), cls.load_classes(cfg.classes_path), Lock())
    
    @staticmethod
    def load_classes(classes_path: str):
        with open(classes_path, newline='') as f:
            reader = csv.reader(f, delimiter=',')
            return {int(row[0]): row[1] for row in reader}
    
    def classify(self, image: Image) -> list:
        image = image.resize((224,224))
        blob = keras.utils.img_to_array(image)
        blob = np.expand_dims(blob,0)

        with self.model.model_lock:
            predictions = self.model.model.predict(blob)

        scores = predictions[0]

        return self.classification_processor(scores).process()

class FileImageClassifier(ImageClassifier):
    def classify(self, im_path: str):
        return super().classify(open_image(im_path))

def get_image_classifier(
        model_config: ClassificationModelConfig=DEFAULT_MODEL_CONFIG,
        model: Optional[ClassificationModel]=None,
        classifier: type[ImageClassifier]=ImageClassifier,
        acp: Optional[ACP]= None,
        **acp_kwargs,
    ) -> ImageClassifier:
    
    if not model:
        model = classifier.load_model(model_config)

    if not acp:
        acp = get_acp(class_names=model.class_names, **acp_kwargs)
    
    return classifier(model=model, acp=acp)

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
    
    if type(classifier) is type:
        classifier = get_image_classifier(*args, classifier=classifier, **kwargs)
    
    return [classifier.classify(image) for image in images]

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
