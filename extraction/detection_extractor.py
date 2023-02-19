from abc import ABC, abstractmethod
from dataclasses import dataclass

from prediction.models import Image
from detection.detector import ObjectDetector, Result, BoundingBox

class Extraction:
    bounding_box: BoundingBox
    image: Image

class DetectionExtractor:
    detector: ObjectDetector

    def __init__(self, detector: ObjectDetector):
        self.detector = detector
    
    def process_result(self, image: Image, result: Result) -> Extraction:
        bounding_box: BoundingBox = result.bounding_box
        extracted_img: Image = image[bounding_box.range()]

        return Extraction(bounding_box, extracted_img)

    def extract(self, image: Image) -> list[Image]:
        results: list[Result] = self.detector.predict(image)

        extractions: list[Extraction]
        for result in results:
            extractions.append(self.process_result(image, result))