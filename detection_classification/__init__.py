from dataclasses import dataclass

from prediction.classes import Scores
from image_utils import BoundingBox, Image
from detection.detector import ObjectDetector, DetectionResult, DetectionResults
from classification.classifier import ImageClassifier, ClassificationResult
from extraction.detection_extraction import extract_detection

@dataclass
class DetectionClassificationResult:
    detection_result: DetectionResult
    classification_result: ClassificationResult

    @property
    def bounding_box(self) -> BoundingBox:
        return self.detection_result.bounding_box
    
    @property
    def confidences(self) -> list[float]:
        return self.classification_result.confidences
    
    @property
    def class_names(self) -> list[str]:
        return self.classification_result.class_names

DetectionClassificationResults = list[DetectionClassificationResult]

class DetectionClassifier:

    detector: ObjectDetector
    classifier: ImageClassifier

    def __init__(self, detector: ObjectDetector, classifier: ImageClassifier):
        self.detector = detector
        self.classifier = classifier
    
    def predict(self, input: Image) -> list[DetectionClassificationResult]:
        detection_results = self.detector.predict(input)

        results: DetectionClassificationResults = []
        for detection_result in detection_results:
            extracted_image = extract_detection(input, detection_result)
            classification_result = self.classifier.predict(extracted_image)

            results.append(DetectionClassificationResult(
                detection_result=detection_result,
                classification_result=classification_result
            ))
        
        return results