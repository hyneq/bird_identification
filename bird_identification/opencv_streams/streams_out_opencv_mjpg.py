import cv2

from ..image_utils import Size

from . import OpenCVVideoOutStream

def get_file_video_out_stream_mjpg(
    path: str, fps: float, size: Size, *_, **__
) -> OpenCVVideoOutStream:
    return OpenCVVideoOutStream(
        cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (size[0], size[1])),
        fps=fps,
        size=size,
    )

factory = get_file_video_out_stream_mjpg
