from typing import Optional

from streams.stream_processor import IFrameProcessor
from image_utils import Image
from . import IAnnotation

class StreamAnnotator(IFrameProcessor[Image, Image]):

    annotation: Optional[IAnnotation]

    def __init__(self):
        self.annotation = None

    def set_annotation(self, annotation: IAnnotation):
        self.annotation = annotation

    def process(self, img: Image) -> Image:
        if self.annotation:
            self.annotation.annotate(img)

        return img