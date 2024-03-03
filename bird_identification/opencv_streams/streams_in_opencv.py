import cv2

from . import OpenCVVideoInStream

def get_file_video_in_stream(path: str, *_, **__) -> OpenCVVideoInStream:
    return OpenCVVideoInStream(cv2.VideoCapture(path))
