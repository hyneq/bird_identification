from typing import Optional
from abc import ABC, abstractmethod

import cv2

from image_utils import Image, Point, Color, BoundingBox

DEFAULT_FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_TEXT_THICKNESS = 2

class IAnnotation(ABC):
    @abstractmethod
    def annotate(self, img: Image):
        pass

class MultiAnnotation(IAnnotation):

    annotations: list[IAnnotation]

    def __init__(self, annotations: list[IAnnotation]):
        self.annotations = annotations
    
    def annotate(self, img: Image):
        for annotation in self.annotations:
            annotation.annotate(img)

class RectangleAnnotation(IAnnotation):

    start_point: Point
    end_point: Point
    color: Color
    thickness: int

    def __init__(self, bounding_box: BoundingBox, color: Color, thickness: int=10, filled: bool=False):
        self.start_point, self.end_point = bounding_box.points()
        self.color = color
        if filled:
            self.thickness = -1
        else:
            self.thickness = thickness

    def annotate(self, img: Image):
        cv2.rectangle(img, self.start_point, self.end_point, self.color, self.thickness)

class TextAnnotation(IAnnotation):

    text: str
    origin: Point
    color: Color

    def __init__(self, text: str, origin: Point, color: Color):
        self.text = text
        self.origin = origin
        self.color = color
    
    def annotate(self, img: Image):
        cv2.putText(img, self.text, self.origin, DEFAULT_FONT, 1, self.color)

class TextWithBackgroundAnnotation(IAnnotation):

    text: TextAnnotation
    background: RectangleAnnotation

    def __init__(self, text: str, origin: Point, background_color: Color, color: Color=(0,0,0)):

        self.text = TextAnnotation(text, origin, color)

        (rect_width, rect_height), _ = cv2.getTextSize(text, DEFAULT_FONT, 1, DEFAULT_TEXT_THICKNESS)

        rect_origin = (origin[0], origin[1]-rect_height)

        self.background = RectangleAnnotation(
            BoundingBox(
                rect_origin[0],
                rect_origin[1],
                rect_width,
                rect_height
            ),
            background_color,
            -1
        )
    
    def annotate(self, img: Image):
        self.background.annotate(img)
        self.text.annotate(img)

class RectangleWithTextAnnotation(IAnnotation):

    rect: RectangleAnnotation
    text: TextWithBackgroundAnnotation

    def __init__(self, bounding_box: BoundingBox, text: str, color: Color, thickness: int=10):
        self.rect = RectangleAnnotation(bounding_box, color, thickness)
        self.text = TextWithBackgroundAnnotation(text, bounding_box.points()[0], color)
    
    def annotate(self, img: Image):
        self.rect.annotate(img)
        self.text.annotate(img)
