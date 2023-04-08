import numpy as np
import cv2

Image = np.ndarray

def img_from_file_strategy(input: str) -> Image:
    return cv2.imread(input)