from streams.stream_processor import IFrameProcessor
from image_utils import Image
from . import IAnnotation

class StreamAnnotator(IFrameProcessor[Image, Image]):

    annotation: IAnnotation

    def set_annotation(self, annotation: IAnnotation):
        self.annotation = annotation

    def process(self, img: Image) -> Image:
        self.annotation.annotate(img)

        return img