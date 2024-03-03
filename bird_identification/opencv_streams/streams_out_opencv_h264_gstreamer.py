import cv2

from ..image_utils import Size

from . import OpenCVVideoOutStream

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


factory = get_file_video_out_stream_h264_gstreamer
