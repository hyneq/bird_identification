from dataclasses import dataclass
import numpy as np
import cv2

Image = np.ndarray

def load_img(img: str) -> Image:
    return cv2.imread(img)

@dataclass()
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    def range(self) -> tuple[np.arange, np.arange]:
        return (
            np.arange(self.y, self.y + self.height),
            np.arange(self.x, self.x + self.width)
        )