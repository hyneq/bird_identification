from typing import Optional
import time

import cv2

from .image_utils import Image, Size
from .time_utils import sleep_interval

from .streams import IVideoInStream, IVideoOutStream, StreamError


class OpenCVVideoInStream(IVideoInStream):
    __slots__: tuple

    reader: cv2.VideoCapture

    def __init__(self, reader: cv2.VideoCapture):
        self.reader = reader

        if not reader.isOpened():
            raise StreamError("Video capture is not open")

    def read(self) -> Optional[Image]:
        (status, frame) = self.reader.read()
        if not status:
            return None

        return frame

    def is_open(self):
        return self.reader.isOpened()

    def close(self):
        self.reader.release()

    @property
    def size(self) -> Size:
        return (
            int(self.reader.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.reader.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    @property
    def fps(self) -> float:
        return self.reader.get(cv2.CAP_PROP_FPS)


class OpenCVVideoOutStream(IVideoOutStream):
    __slots__: tuple

    writer: cv2.VideoWriter

    size: Size
    fps: float

    def __init__(self, writer: cv2.VideoWriter, size: Size, fps: float):
        self.writer = writer

        self.size = size
        self.fps = fps

        if not writer.isOpened():
            raise StreamError("Video writer is not open")

    def write(self, frame: Image):
        self.writer.write(frame)

    def close(self):
        self.writer.release()


class RealTimeOpenCVVideoOutStream(OpenCVVideoOutStream):
    __slots__: tuple

    last_write_time: float = 0.0

    def write(self, frame: Image):
        sleep_interval(self.last_write_time, (1 / self.fps))
        super().write(frame)
        self.last_write_time = time.time()


def get_file_video_in_stream(path: str, *_, **__) -> OpenCVVideoInStream:
    return OpenCVVideoInStream(cv2.VideoCapture(path))


# inspired by https://www.geeksforgeeks.org/saving-a-video-using-opencv/
def get_file_video_out_stream_h264(
    path: str, fps: float, size: tuple[int, int], *_, **__
) -> OpenCVVideoOutStream:
    return OpenCVVideoOutStream(
        cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"H264"), fps, (size[0], size[1])),
        fps=fps,
        size=size,
    )


def get_file_video_out_stream_h264_gstreamer(
    path: str, fps: float, size: Size
) -> OpenCVVideoOutStream:
    pipeline = f"appsrc ! videoconvert ! v4l2h264enc min-force-key-unit-interval=10000000000 ! video/x-h264,profile=(string)main,level=(string)4 ! h264parse ! filesink location={path}"

    return OpenCVVideoOutStream(
        cv2.VideoWriter(
            pipeline, cv2.CAP_GSTREAMER, cv2.VideoWriter_fourcc(*"H264"), fps, size
        ),
        fps=fps,
        size=size,
    )

def get_file_video_out_stream_mjpg(
    path: str, fps: float, size: tuple[int, int], *_, **__
) -> OpenCVVideoOutStream:
    return OpenCVVideoOutStream(
        cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (size[0], size[1])),
        fps=fps,
        size=size,
    )
