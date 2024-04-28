from typing import Optional
import subprocess
import numpy as np

from .image_utils import Image, Size

from .streams.subprocess_streams import SubprocessVideoOutStream

# Based on https://stackoverflow.com/a/69383391

def get_file_video_out_stream_h264(
    path: str, fps: float, size: Size, *_, **__
) -> SubprocessVideoOutStream:
    args = [
        "ffmpeg",
        "-re", "-stream_loop", "-1",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{size[0]}x{size[1]}", "-i", "pipe:0",
        "-vcodec", "h264_v4l2m2m", "-b:v", "4M", "-f", "h264", "-y", path
    ]

    return SubprocessVideoOutStream(
        subprocess.Popen(args, stdin=subprocess.PIPE),
        fps=fps,
        size=size
    )

factory = get_file_video_out_stream_h264
