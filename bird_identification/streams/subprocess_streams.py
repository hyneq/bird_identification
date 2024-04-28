import subprocess
import numpy as np

from ..image_utils import Image, Size

from . import IVideoStream, IVideoOutStream

class SubprocessVideoOutStream(IVideoOutStream):
    __slots__: tuple

    process: subprocess.Popen
    fps: float
    size: Size

    def __init__(self, process: subprocess.Popen, fps: float, size: Size):
        self.process = process
        self.fps = fps
        self.size = size

    def write(self, frame: Image):
        self.process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        self.process.stdin.close()
        self.process.wait()
