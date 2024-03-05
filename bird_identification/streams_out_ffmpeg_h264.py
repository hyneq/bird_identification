from typing import Optional
import subprocess
import numpy as np

from .image_utils import Image, Size

from .streams import IVideoOutStream

# Based on https://stackoverflow.com/a/69383391

class FfmpegVideoOutStream(IVideoOutStream):
    __slots__: tuple

    ffmpeg_process: subprocess.Popen
    fps: float
    size: Size

    def __init__(self, ffmpeg_process: subprocess.Popen, fps: float, size: Size):
        self.ffmpeg_process = ffmpeg_process
        self.fps = fps
        self.size = size

    def write(self, frame: Image):
        self.ffmpeg_process.stdin.write(frame.astype(np.uint8).tobytes())

    def close(self):
        self.ffmpeg_process.stdin.close()
        self.ffmpeg_process.wait()

def get_file_video_out_stream_h264(
    path: str, fps: float, size: Size, *_, **__
) -> FfmpegVideoOutStream:
    args = [
        "ffmpeg",
        "-re", "-stream_loop", "-1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{size[0]}x{size[1]}", "-i", "pipe:0",
        "-vcodec", "h264_v4l2m2m", "-b:v", "4M", "-f", "h264", "-y", path
    ]

    return FfmpegVideoOutStream(
        subprocess.Popen(args, stdin=subprocess.PIPE),
        fps=fps,
        size=size
    )

factory = get_file_video_out_stream_h264
