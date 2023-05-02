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

    def __init__(self, text: str, origin: Point, background_color: Color, color: Color=(255,255,255), padding: int = 10, up: bool=False, left: bool=False):
        self.text = TextAnnotation(text, origin, color)

        self.text_size = text_size = cv2.getTextSize(text, DEFAULT_FONT, 1, DEFAULT_TEXT_THICKNESS)
        origin = (origin[0]-padding, origin[1]-padding)
        text_size = (text_size[0]+padding*2, text_size[1]+padding*2)

        if up:
            origin = (origin[0], origin[1]-text_size[1])
        
        if left:
            origin = (origin[0]-text_size[1], origin[1])

        self.background = RectangleAnnotation(
            BoundingBox(
                origin[0],
                origin[1],
                text_size[0],
                text_size[1]
            ), background_color, -1)
    
    def annotate(self, img: Image):
        self.background.annotate(img)
        self.text.annotate(img)

class RectangleWithTextAnnotation(IAnnotation):

    rect: RectangleAnnotation
    text: TextWithBackgroundAnnotation

    def __init__(self, bounding_box: BoundingBox, text: str, color: Color, thickness: int=10):
        self.rect = RectangleAnnotation(bounding_box, color, thickness)
        self.text = TextWithBackgroundAnnotation(text, bounding_box.points()[0], color, up=True)
    
    def annotate(self, img: Image):
        self.rect.annotate(img)
        self.text.annotate(img)
