import cv2

from ..image_utils import Size

from . import OpenCVVideoOutStream

# inspired by https://www.geeksforgeeks.org/saving-a-video-using-opencv/
def get_file_video_out_stream_h264(
    path: str, fps: float, size: Size, *_, **__
) -> OpenCVVideoOutStream:
    return OpenCVVideoOutStream(
        cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"H264"), fps, (size[0], size[1])),
        fps=fps,
        size=size,
    )

factory = get_file_video_out_stream_h264
