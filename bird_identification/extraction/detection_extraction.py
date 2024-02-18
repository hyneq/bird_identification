from ..image_utils import Image
from ..detection.detector import DetectionResult


def extract_detection(image: Image, result: DetectionResult) -> Image:
    return image[result.bounding_box.slices()]
