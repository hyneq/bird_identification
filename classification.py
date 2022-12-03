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

from util import ClassificationMode, ClassRequiredForModeException, get_class_numbers

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

DEFAULT_MIN_CONFIDENCE = 0.5


class ClassificationProcessor(ABC):
    __slots__ = [
        "output"
    ]

    min_confidence : float = DEFAULT_MIN_CONFIDENCE

    def __init__(self, output):
        self.output = output
    
    @abstractmethod
    def get_classes(self):
        pass

    def is_enough_confidence(self, score: np.array) -> bool:
        return (score > self.min_confidence)
    
    def process(self) -> list:
        self.scores = scores = self.output
        
        classes_filtered = []
        for class_ in self.get_classes():
            score = scores[class_]
            if self.is_enough_confidence(score):
                classes_filtered.append(class_)
        
        return [Result(class_, scores[class_]) for class_ in classes_filtered]
    
    @classmethod
    def _class_copy(cls):
        class ClassificationProcessorWithArgs(cls):
            pass
        
        return ClassificationProcessorWithArgs
    
    @classmethod
    def with_args(cls, min_confidence: Optional[float] = None):
        cls = cls._class_copy()
        if min_confidence:
            cls.min_confidence = min_confidence

        return cls


class FixedClassClassificationProcessor(ClassificationProcessor):
    __slots__ = [
        "classes"
    ]

    classes: list

    def get_classes(self) -> list:
        return self.classes

    @classmethod
    def with_args(cls, classes: list[int], **kwargs):
        cls = cls.__bases__[0].with_args(**kwargs)
        cls.classes = classes

        return cls


class MaxClassClassificationProcessor(ClassificationProcessor):

    def get_classes(self) -> list:
        return [np.argmax(self.scores)]

class SortClassClassificationProcessor(ClassificationProcessor):

    def get_classes(self) -> list:
        return np.argsort(self.scores)[::-1]


class ImageClassifier:
    __slots__ = [
        "model",
        "classification_processor"
    ]

    model: ClassificationModel

    def __init__(
            self, model_config: ClassificationModelConfig=DEFAULT_MODEL_CONFIG,
            model: Optional[ClassificationModel]=None,
            classification_processor=MaxClassClassificationProcessor
        ):

        if not model:
            model = self.load_model(model_config)

        self.model = model
        self.classification_processor = classification_processor

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

CLASSIFICATION_MODE_PROCESSORS = {
    ClassificationMode.FIXED: FixedClassClassificationProcessor,
    ClassificationMode.MAX: MaxClassClassificationProcessor,
    ClassificationMode.SORTED: SortClassClassificationProcessor
}

def get_image_classifier(
        classes: Optional[Union[list[str],list[int], str, int]]=None,
        mode: ClassificationMode=None,
        min_confidence: Optional[Union[float,int]]=None,
        model_config: ClassificationModelConfig=DEFAULT_MODEL_CONFIG,
        model: Optional[ClassificationModel]=None,
        classifier: type[ImageClassifier]=ImageClassifier,
        classification_processor: Optional[type[ClassificationProcessor]]=None
    ) -> ImageClassifier:

    if not mode:
        if classes:
            mode = ClassificationMode.FIXED
        else:
            mode = ClassificationMode.MAX
    
    if type(min_confidence) is int:
        min_confidence = min_confidence/100
    
    if not model:
        model = classifier.load_model(model_config)

    if not classification_processor:

        classification_processor = CLASSIFICATION_MODE_PROCESSORS[mode]

        if mode.classes_needed:
            if not classes:
                raise ClassRequiredForModeException(mode)
            
            if type(classes) is not list:
                classes = [classes]
            
            if type(classes[0]) is str:
                classes = get_class_numbers(classes, model.class_names)
            
            classification_processor = classification_processor.with_args(classes=classes, min_confidence=min_confidence)
        else:
            classification_processor = classification_processor.with_args(min_confidence=min_confidence)
    
    return classifier(model=model)

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
