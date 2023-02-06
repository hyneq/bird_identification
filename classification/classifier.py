from typing import Optional, Union
from dataclasses import dataclass

import numpy as np

from prediction.predictor import PredictionProcessorWithCS, PredictorWithCS, FileImagePredictor, get_predictor_factory
from .models import ClassificationModelConfig, ClassificationModel

from defaults.classification import DEFAULT_MODEL_CONFIG

@dataclass
class Result:
    class_names: list[str]
    confidences: list[float]

class ClassificationProcessor(PredictionProcessorWithCS[ClassificationModel, np.ndarray, Result]):
    __slots__: tuple

    def get_results(self, classes) -> list:
        return Result(self.model.class_names.get_names(classes), list(self.scores[classes]))

    def process(self) -> list:
        self.scores = self.output

        classes = self.cs.get_filtered_classes(self.scores)
        
        return self.get_results(classes)

class ImageClassifier(PredictorWithCS[ClassificationModel, ClassificationModelConfig, ClassificationProcessor, np.ndarray, np.ndarray, Result]):
    __slots__: tuple

    model_cls = ClassificationModel

    prediction_processor = ClassificationProcessor

class FileImageClassifier(FileImagePredictor[ImageClassifier, np.ndarray]):
    __slots__: tuple

    predictor_cls = ImageClassifier

get_image_classifier = get_predictor_factory(
    "get_image_classifier",
    ImageClassifier,
    ClassificationModel,
    ClassificationModelConfig,
    DEFAULT_MODEL_CONFIG
)

def classify_images(
        images: Union[list[str], list[np.ndarray], str, np.ndarray],
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

    if isinstance(classifier, type):
        classifier = get_image_classifier(*args, predictor=classifier, **kwargs)
    
    return [classifier.predict(image) for image in images]