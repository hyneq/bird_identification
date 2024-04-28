from ..image_utils import Size

from . import FFmpegVideoOutStreamFactory, get_h264_encode_args

def get_rtsp_out_args(path: str):
    return [
        "-f", "rtsp", path
    ]

class FFmpegRTSPVideoOutStreamFactory(FFmpegVideoOutStreamFactory):

    name = "ffmpeg_rtsp"

    def get_args(self,
        path: str, fps: float, size: Size
    ):
        return (
            super().get_args(path, fps, size) + 
            get_h264_encode_args() +
            get_rtsp_out_args(path)
        )


get_rtsp_video_out_stream = FFmpegRTSPVideoOutStreamFactory()

factory = get_rtsp_video_out_stream
