from typing_extensions import Self
from dataclasses import dataclass
import numpy as np
import cv2

Image = np.ndarray
Point = tuple[int, int]
Size = tuple[int, int]
Color = tuple[int, int, int]


def load_img(img: str) -> Image:
    return cv2.imread(img)


@dataclass()
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def slices(self) -> tuple[slice, slice]:
        return (slice(self.y, self.y + self.height), slice(self.x, self.x + self.width))

    def points(self) -> tuple[Point, Point]:
        return ((self.x, self.y), (self.x + self.width, self.y + self.height))
