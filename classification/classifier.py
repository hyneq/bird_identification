from typing import Optional, Union
from dataclasses import dataclass

import numpy as np

from prediction.predictor import PredictionInputT_cls, PredictorConfig, PredictionProcessorWithClasses, PredictionProcessorWithClassesFactory, PredictorWithClasses, PredictorWithClassesFactory
from prediction.classes import Scores
from prediction.image_utils import Image
from .models import ClassificationModelConfig, ClassificationModelOutput, ClassificationModel, classification_model_factory

@dataclass
class ClassificationResult:
    class_names: list[str]
    confidences: list[float]

class ClassificationProcessor(PredictionProcessorWithClasses[ClassificationModelOutput, ClassificationResult]):
    __slots__: tuple

    def get_results(self, classes: list, scores: Scores) -> ClassificationResult:
        return ClassificationResult(self.class_names.get_names(classes), list(scores[classes]))

    def process(self, model_output: ClassificationModelOutput) -> ClassificationResult:
        scores = model_output

        classes = self.cs.get_filtered_classes(scores)
        
        return self.get_results(classes, scores)

class ClassificationProcessorFactory(PredictionProcessorWithClassesFactory[ClassificationModelOutput, ClassificationResult]):
    def __init__(self):
        super().__init__(ClassificationProcessor)

class ImageClassifier(PredictorWithClasses[PredictionInputT_cls, Image, ClassificationModelOutput, ClassificationResult]):
    __slots__: tuple

    model_cls = ClassificationModel

    prediction_processor = ClassificationProcessor

@dataclass
class ClassifierConfig(PredictorConfig[ClassificationModelConfig]):
    pass

image_classifier_factory = PredictorWithClassesFactory(
    predictor=ImageClassifier,
    predictor_config=ClassifierConfig,
    model_factory=classification_model_factory,
    prediction_processor_factory=ClassificationProcessorFactory()
)

get_image_classifier = image_classifier_factory.get_predictor

def classify_images(
        images: Union[list[str], list[np.ndarray], str, np.ndarray],
        *args,
        classifier: Optional[Union[type[ImageClassifier],ImageClassifier]]=None,
        **kwargs
    ):

    if type(images) is not list:
        images = [images]
    
    if not classifier:
        classifier = ImageClassifier

    if isinstance(classifier, type):
        classifier = get_image_classifier(*args, predictor=classifier, **kwargs)
    
    return [classifier.predict(image) for image in images]