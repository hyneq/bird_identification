from typing import Optional

import time
import picamera2

from .image_utils import Image, Size
from .time_utils import sleep_interval

from .streams import IVideoInStream

class Picamera2VideoInStream(IVideoInStream):
    __slots__: tuple

    picam2: picamera2.Picamera2
    last_time: float
    
    def __init__(self, picam2: picamera2.Picamera2):
        self.picam2 = picam2

        picam2.start()
        self.last_time = time.time()
    
    @property
    def fps(self) -> float:
        return 1e6/self.picam2.camera_config["controls"].FrameDurationLimits[0]
    
    @property
    def size(self) -> Size:
        return self.picam2.camera_config["main"]["size"]
    
    def read(self) -> Optional[Image]:
        #self._wait()

        frame = self.picam2.capture_array()

        return frame

    def _wait(self):
        sleep_interval(self.last_time, 1/self.fps)
        self.last_time = time.time()

    def is_open(self):
        return self.picam2.started
    
    def close(self):
        self.picam2.close()


def get_camera_video_in_stream(*_, fps: Optional[float]=None, size: Optional[Size]=None, **__) -> Picamera2VideoInStream:
    picam2 = picamera2.Picamera2()

    if fps:
        frame_duration = round(1e6/fps) # to microseconds
        picam2.video_configuration.controls.FrameDurationLimits = (frame_duration, frame_duration)

    if size:
        picam2.video_configuration.main.size = size
    
    picam2.video_configuration.main.format = "RGB888"

    picam2.configure("video")

    return Picamera2VideoInStream(picam2)

factory = get_camera_video_in_stream
