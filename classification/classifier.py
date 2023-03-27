from typing import Optional, Union
from dataclasses import dataclass

import numpy as np

from prediction.predictor import PredictorConfig, PredictionProcessorWithCS, PredictorWithCS, FileImagePredictor, PredictorFactory, WrapperPredictorFactory
from prediction.models import Image
from .models import ClassificationModelConfig, ClassificationModelOutput, ClassificationModel, model_factory

@dataclass
class ClassificationResult:
    class_names: list[str]
    confidences: list[float]

class ClassificationProcessor(PredictionProcessorWithCS[ClassificationModel, ClassificationModelOutput, ClassificationResult]):
    __slots__: tuple

    def get_results(self, classes) -> ClassificationResult:
        return ClassificationResult(self.model.class_names.get_names(classes), list(self.scores[classes]))

    def process(self) -> ClassificationResult:
        self.scores = self.output

        classes = self.cs.get_filtered_classes(self.scores)
        
        return self.get_results(classes)

class ImageClassifier(PredictorWithCS[ClassificationModel, ClassificationModelConfig, ClassificationProcessor, Image, ClassificationModelOutput, ClassificationResult]):
    __slots__: tuple

    model_cls = ClassificationModel

    prediction_processor = ClassificationProcessor

class FileImageClassifier(FileImagePredictor[ImageClassifier, ClassificationResult]):
    __slots__: tuple

    predictor_cls = ImageClassifier

@dataclass
class ClassifierConfig(PredictorConfig[ClassificationModelConfig]):
    pass

image_classifier_factory = PredictorFactory(
    predictor=ImageClassifier,
    predictor_config=ClassifierConfig,
    model_factory=model_factory
)

file_image_classifier_factory = WrapperPredictorFactory(
    wrapper_predictor_cls = FileImageClassifier,
    wrapped_predictor_factory = image_classifier_factory
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
        if type(images[0]) is str:
            classifier = FileImageClassifier
        else:
            classifier = ImageClassifier

    if isinstance(classifier, type):
        classifier = get_image_classifier(*args, predictor=classifier, **kwargs)
    
    return [classifier.predict(image) for image in images]