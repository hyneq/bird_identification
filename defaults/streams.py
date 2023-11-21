from typing import Callable, Mapping

from streams import IPathVideoInStreamFactory, IPathVideoOutStreamFactory
import opencv_streams

DEFAULT_PATH_IN_STREAM_FACTORY: str
DEFAULT_PATH_OUT_STREAM_FACTORY: str

DEFAULT_PATH_IN_STREAM_PATH: str
DEFAULT_PATH_OUT_STREAM_PATH: str

PATH_IN_STREAM_FACTORIES: Mapping[str, IPathVideoInStreamFactory]
PATH_OUT_STREAM_FACTORIES: Mapping[str, IPathVideoOutStreamFactory]

DEFAULT_PATH_IN_STREAM_FACTORY = "opencv"
DEFAULT_PATH_OUT_STREAM_FACTORY = "opencv"

DEFAULT_PATH_IN_STREAM_PATH = "-"
DEFAULT_PATH_OUT_STREAM_PATH = "-"

PATH_IN_STREAM_FACTORIES = {
    "opencv": opencv_streams.get_file_video_in_stream
}

PATH_OUT_STREAM_FACTORIES = {
    "opencv": opencv_streams.get_file_video_out_stream
}