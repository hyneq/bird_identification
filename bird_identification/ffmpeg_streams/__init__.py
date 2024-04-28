from typing import Sequence

import subprocess

from ..image_utils import Size
from ..streams import IPathVideoOutStreamFactory
from ..streams.subprocess_streams import SubprocessVideoOutStream

class FFmpegVideoOutStreamFactory(IPathVideoOutStreamFactory):

    def __call__(self, 
        path: str, fps: float, size: Size, *_, **__
    ):
        args = self.get_args(path, fps, size)
        
        return SubprocessVideoOutStream(
            subprocess.Popen(args, stdin=subprocess.PIPE),
            fps=fps,
            size=size
        )
    

    def get_args(self,
        path: str, fps: float, size: Size
    ):
        return get_base_args(fps, size)


def get_input_args(fps: float, size: Size) -> list[str]:
    return [
        "-r", f"{fps}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{size[0]}x{size[1]}",
        "-i", "pipe:0"
    ]


def get_base_args(fps: float, size: Size) -> list[str]:
    return ["ffmpeg"] + get_input_args(fps, size)


def get_h264_encode_args() -> list[str]:
    return ["-vcodec", "h264_v4l2m2m", "-b:v", "4M"]
