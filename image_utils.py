import numpy as np
import cv2

Image = np.ndarray

def load_img(img: str) -> Image:
    return cv2.imread(img)