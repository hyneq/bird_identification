from ..image_utils import Size

from . import FFmpegVideoOutStreamFactory, get_h264_encode_args

def get_h264_file_out_args(path: str):
    return [
        "-f", "h264", "-y", path
    ]

class FFmpegH264VideoOutStreamFactory(FFmpegVideoOutStreamFactory):

    name = "ffmpeg_h264"

    def get_args(self,
        path: str, fps: float, size: Size
    ):
        return (
            super().get_args(path, fps, size) + 
            get_h264_encode_args() +
            get_h264_file_out_args(path)
        )


get_file_video_out_stream_h264 = FFmpegH264VideoOutStreamFactory()

factory = get_file_video_out_stream_h264