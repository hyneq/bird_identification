from enum import Enum
import numpy as np
from typing import Optional, Union
from abc import ABC, abstractmethod

DEFAULT_MIN_CONFIDENCE = 0.5

class AbstractClassificationProcessor(ABC):
    __slots__ = ("min_confidence",)

    min_confidence: float

    def __init__(self, min_confidence=None):
        if not min_confidence:
            min_confidence = DEFAULT_MIN_CONFIDENCE

        self.min_confidence = min_confidence

    @abstractmethod
    def get_classes(self, scores):
        pass

    def get_filtered_classes(self, scores) -> list:
        classes_filtered = []
        for class_ in self.get_classes(scores):
            if scores[class_] > self.min_confidence:
                classes_filtered.append(class_)
        
        return classes_filtered


class FixedClassAbstractClassificationProcessor(AbstractClassificationProcessor):
    __slots__: tuple

    classes: list

    def __init__(self, classes, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.classes = classes

    def get_classes(self, scores) -> list:
        return self.classes

class MaxClassAbstractClassificationProcessor(AbstractClassificationProcessor):
    __slots__: tuple

    def get_classes(self, scores) -> list:
        return [np.argmax(scores)]

class SortClassAbstractClassificationProcessor(AbstractClassificationProcessor):
    __slots__: tuple

    def get_classes(self, scores) -> list:
        return np.argsort(scores)[::-1]

DEFAULT_ACP = MaxClassAbstractClassificationProcessor

class ClassificationMode(Enum):
    FIXED = ("Fixed class list", True, FixedClassAbstractClassificationProcessor)
    MAX = ("Single class with maximum confidence", False, MaxClassAbstractClassificationProcessor)
    SORTED = ("All classes, sorted by confidence", False, SortClassAbstractClassificationProcessor)

    def __init__(self, description: str, classes_needed: bool, acp: AbstractClassificationProcessor):
        self.description = description
        self.classes_needed = classes_needed
        self.acp = acp


class ClassRequiredForModeException(ValueError):
    def __init__(mode: ClassificationMode):
        super().__init__("Class must be specified for classification mode '{}'".format(mode.description))

def get_acp(
        mode: ClassificationMode, 
        min_confidence: Optional[float]=None,
        classes: Optional[Union[list[int],list[str]]]=None,
        class_names: Optional[list[str]]=None
    ):
    if mode.classes_needed:
        if not classes:
            raise ClassRequiredForModeException(mode)
        
        if type(classes) is not list:
            classes = [classes]
        
        if type(classes[0]) is str:
            classes = get_class_numbers(classes, class_names)
    
        return mode.acp(classes,min_confidence)
    else:
        return mode.acp(min_confidence)

def get_class_numbers(classes: list[str], model_class_names: list[str]) -> list[int]:
    return [model_class_names.index(class_name) for class_name in classes]