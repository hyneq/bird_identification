from typing import Optional

from image_utils import Image
from streams.stream_processor import IFrameProcessor
from prediction.predictor import APredictor, IPredictionResultWithClassesAndBoundingBoxes
from annotation import IAnnotation, MultiAnnotation, RectangleWithTextAnnotation
from annotation.stream_annotator import StreamAnnotator

class ImagePredictionFrameProcessor(IFrameProcessor[Image, Image]):

    predictor: APredictor[Image, Image, list[IPredictionResultWithClassesAndBoundingBoxes]]
    annotator: StreamAnnotator
    img: Optional[Image]

    def __init__(self, predictor: APredictor[Image, Image, list[IPredictionResultWithClassesAndBoundingBoxes]]):
        self.predictor = predictor
        self.annotator = StreamAnnotator()
        self.img = None
    
    def predict(self):
        if self.img:
            self.set_annotations(self.predictor.predict(self.img))
    
    def set_annotations(self, results: list[IPredictionResultWithClassesAndBoundingBoxes]):
        annotations: list[IAnnotation] = []
        for result in results:
            annotations.append(RectangleWithTextAnnotation(
                result.bounding_box,
                "{} {}%".format(result.class_name, result.confidence*100) if result.class_name is not None else "not recognized",
                (255, 255, 0)
            ))
        
        self.annotator.set_annotation(MultiAnnotation(annotations))

    def process(self, img: Image) -> Image:
        self.img = img.copy()

        return self.annotator.process(img)