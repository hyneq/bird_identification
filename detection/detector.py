from typing import Optional, Union
from dataclasses import dataclass

import cv2
import numpy as np

from prediction.predictor import PredictionInputT_cls, PredictorConfig, PredictionProcessorWithClasses, PredictionProcessorWithClassesFactory, PredictorWithClasses, PredictorWithClassesFactory
from prediction.image_utils import Image
from .models import DetectionModelConfig, DetectionModelOutput, DetectionModel, model_factory

from defaults.detection import DEFAULT_NMS_THRESHOLD

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
class DetectionResult:
    bounding_box: BoundingBox
    labels: list[str]
    confidences: any

DetectionResults = list[DetectionResult]


class DetectionProcessor(PredictionProcessorWithClasses[DetectionModelOutput, DetectionResults]):
    __slots__: tuple

    NMS_threshold: float

    def __init__(self, *args, NMS_threshold: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.NMS_threshold = NMS_threshold
    
    def process(self, output: DetectionModelOutput) -> DetectionResults:
        return DetectionProcessorInstance(self, output).process()

class DetectionProcessorInstance:

    processor: DetectionProcessor
    output: DetectionModelOutput

    def __init__(self, processor: DetectionProcessor, output: DetectionModelOutput):
        self.processor = processor
        self.output = output

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

        classes = self.processor.cs.get_filtered_classes(scores)

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

        return cv2.dnn.NMSBoxes(self.bounding_boxes, [confidence[0] for confidence in self.confidences], self.processor.cs.min_confidence, self.processor.NMS_threshold)
    
    def get_results(self, filtered) -> DetectionResults:
        return [DetectionResult(BoundingBox(*self.bounding_boxes[i]), self.processor.class_names.get_names(self.classes[i]), list(self.confidences[i])) for i in filtered]
    
    def process(self) -> DetectionResults:
        self.bounding_boxes = []
        self.confidences = []
        self.classes = []

        for obj in self.output:
            self.process_object(obj)

        filtered = self.NMSBoxes()

        return self.get_results(filtered)

@dataclass
class DetectionProcessorFactory(PredictionProcessorWithClassesFactory[DetectionModelOutput, DetectionResults]):

    NMS_threshold = DEFAULT_NMS_THRESHOLD

    def __init__(self):
        super().__init__(DetectionProcessor)

    def get_prediction_processor(self, *args, NMS_threshold: Optional[float]=None, **kwargs) -> DetectionProcessor:
        if not NMS_threshold:
            NMS_threshold = self.NMS_threshold

        return super().get_prediction_processor(*args, NMS_threshold=NMS_threshold, **kwargs)

class ObjectDetector(PredictorWithClasses[PredictionInputT_cls, Image, DetectionModelOutput, DetectionResults]):
    __slots__: tuple

    model_cls = DetectionModel

    prediction_processor = DetectionProcessor

@dataclass
class DetectorConfig(PredictorConfig[DetectionModelConfig]):
    pass

object_detector_factory = PredictorWithClassesFactory(
    predictor=ObjectDetector,
    predictor_config=DetectorConfig,
    model_factory=model_factory,
    prediction_processor_factory=DetectionProcessorFactory()
)

get_object_detector = object_detector_factory.get_predictor

"""
get_object_detector = get_predictor_factory(
    name="get_object_detector",
    predictor=ObjectDetector,
    predictor_config_cls=DetectorConfig,
    get_model=get_detection_model
)
"""

def detect_objects(
        images: Union[list[str], list[np.ndarray], str, np.ndarray],
        *args,
        detector: Optional[Union[type[ObjectDetector],ObjectDetector]]=None,
        **kwargs
    ):

    if type(images) is not list:
        images = [images]
    
    if not detector:
        detector = ObjectDetector
    
    
    if isinstance(detector, type):
        detector = get_object_detector(*args, predictor=detector, **kwargs)
    
    return [detector.predict(image) for image in images]