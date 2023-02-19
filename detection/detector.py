from typing import Optional, Union
from dataclasses import dataclass

import cv2
import numpy as np

from prediction.predictor import PredictorConfig, PredictionProcessorWithCS, PredictorWithCS, FileImagePredictor, get_predictor_factory
from prediction.models import Image
from .models import DetectionModelConfig, DetectionModelOutput, DetectionModel

from defaults.detection import DEFAULT_MODEL_CONFIG, DEFAULT_MODEL_CLS, DEFAULT_NMS_THRESHOLD

@dataclass()
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def range(self) -> tuple[np.arange, np.arange]:
        return (
            np.arange(self.y, self.y + self.height),
            np.arange(self.x, self.x + self.width)
        )

@dataclass()
class Result:
    label: str
    bounding_box: BoundingBox
    confidence: any


class DetectionProcessor(PredictionProcessorWithCS[DetectionModel, DetectionModelOutput, Result]):
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

class ObjectDetector(PredictorWithCS[DetectionModel, DetectionModelConfig, DetectionProcessor, Image, DetectionModelOutput, Result]):
    __slots__: tuple

    model_cls = DetectionModel

    prediction_processor = DetectionProcessor

class FileObjectDetector(FileImagePredictor[ObjectDetector, Result]):
    __slots__: tuple

    predictor_cls = ObjectDetector

@dataclass
class DetectorConfig(PredictorConfig[ObjectDetector]):
    pass

get_object_detector = get_predictor_factory(
    name="get_object_detector",
    predictor=ObjectDetector,
    predictor_config_cls=DetectorConfig,
    DEFAULT_MODEL_CLS=DEFAULT_MODEL_CLS,
    DEFAULT_MODEL_CONFIG=DEFAULT_MODEL_CONFIG
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
    
    
    if isinstance(detector, type):
        detector = get_object_detector(*args, predictor=detector, **kwargs)
    
    return [detector.predict(image) for image in images]