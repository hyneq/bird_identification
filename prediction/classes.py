from enum import Enum
import numpy as np
from typing import Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

DEFAULT_MIN_CONFIDENCE = 0.5

ClassList = Union[list[int],list[str], str, int]

class ClassSelector(ABC):
    __slots__: tuple

    min_confidence: float

    def __init__(self, min_confidence=None):
        if not min_confidence:
            min_confidence = DEFAULT_MIN_CONFIDENCE

        self.min_confidence = min_confidence

    @abstractmethod
    def get_classes(self, scores) -> list:
        pass

    def get_filtered_classes(self, scores) -> list:
        classes_filtered = []
        for class_ in self.get_classes(scores):
            if scores[class_] > self.min_confidence:
                classes_filtered.append(class_)
        
        return classes_filtered


class FixedClassSelector(ClassSelector):
    __slots__: tuple

    classes: list

    def __init__(self, classes, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.classes = classes

    def get_classes(self, scores) -> list:
        return self.classes

class MaxClassSelector(ClassSelector):
    __slots__: tuple

    def get_classes(self, scores) -> list:
        return [np.argmax(scores)]

class SortClassSelector(ClassSelector):
    __slots__: tuple

    def get_classes(self, scores) -> list:
        return list(np.argsort(scores)[::-1])


class ClassificationMode(Enum):
    FIXED = ("Fixed class list", True, FixedClassSelector)
    MAX = ("Single class with maximum confidence", False, MaxClassSelector)
    SORTED = ("All classes, sorted by confidence", False, SortClassSelector)

    def __init__(self, description: str, classes_needed: bool, cs: ClassSelector):
        self.description = description
        self.classes_needed = classes_needed
        self.cs = cs

DEFAULT_CLASSIFICATION_MODE = ClassificationMode.MAX

DEFAULT_CLASS_SELECTOR = DEFAULT_CLASSIFICATION_MODE.cs

class ClassRequiredForModeException(ValueError):
    def __init__(self, mode: ClassificationMode):
        super().__init__("Class must be specified for classification mode '{}'".format(mode.description))

@dataclass
class ClassSelectorConfig:
    mode: ClassificationMode = None
    min_confidence: Optional[float] = None
    classes: Optional[ClassList] = None

@dataclass
class ClassNames:
    class_names: list[str]

    def get_name(self, num: int):
        return self.class_names[num]
    
    def get_names(self, nums: Union[list[int],np.ndarray]):
        return [self.get_name(num) for num in nums]

    def get_number(self, name: str):
        return self.class_names.index(name)
    
    def get_numbers(self, names: list[str]):
        return [self.get_number(name) for name in names]
    
    @classmethod
    def load_from_file(cls, path: str):
        with open(path, newline='') as f:
            return cls(f.read().splitlines())

def get_class_selector(
        mode: Optional[ClassificationMode]=None, 
        min_confidence: Optional[float]=None,
        classes: Optional[Union[list[int],list[str], str, int]]=None,
        model_class_names: Optional[ClassNames]=None
    ):

    if not mode:
        if classes:
            mode = ClassificationMode.FIXED
        else:
            mode = ClassificationMode.MAX

    if mode.classes_needed:
        if not classes:
            raise ClassRequiredForModeException(mode)
        
        if type(classes) is not list:
            classes = [classes]
        
        if type(classes[0]) is str:
            classes = [model_class_names.get_number(class_name) for class_name in classes]
    
        return mode.cs(classes,min_confidence)
    else:
        return mode.cs(min_confidence)