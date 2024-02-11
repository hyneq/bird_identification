#!/usr/bin/python3

import os, sys, cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from bird_identification.image_utils import Image, BoundingBox
from bird_identification import annotation

BASE_DIR: str = os.path.join(os.path.dirname(__file__), "images", "annotation_example")

source_img: Image = cv2.imread(os.path.join(BASE_DIR, "source.jpg"))

def annotate(annotation: annotation.IAnnotation, name: str):
    img = source_img.copy()
    annotation.annotate(img)
    cv2.imwrite(os.path.join(BASE_DIR, name), img)

# Rectangle annotation
rect_annotation = annotation.RectangleAnnotation(BoundingBox(50, 50, 200, 200), (0, 255, 0))
annotate(rect_annotation, "rect.jpg")

# Filled rectangle annotation
filled_rect_annotation = annotation.RectangleAnnotation(BoundingBox(50, 50, 200, 200), (0, 255, 0), filled=True)
annotate(filled_rect_annotation, "filled_rect.jpg")

# Text annotation
text_annotation = annotation.TextAnnotation("example text", (50, 50), (0, 0, 0))
annotate(text_annotation, "text.jpg")

# Text with background annotation
text_with_background_annotation = annotation.TextWithBackgroundAnnotation("example text with background", (50, 50), (0, 255, 0))
annotate(text_with_background_annotation, "text_with_background.jpg")

# Rectangle with text annotation
rect_with_text_annotation = annotation.RectangleWithTextAnnotation(BoundingBox(50, 50, 200, 200), "example label", (0, 255, 0))
annotate(rect_with_text_annotation, "rect_with_text.jpg")