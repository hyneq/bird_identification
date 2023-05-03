from typing import Generic, TypeVar, Optional, Callable
from typing_extensions import Self 
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from image_utils import Image, Size

StreamInputT = TypeVar("StreamInputT")
StreamOutputT = TypeVar("StreamOutputT")

class StreamError(Exception):
    pass

class IStream(ABC):
    __slots__: tuple

    @abstractmethod
    def close(self):
        pass

class IInStream(IStream, Generic[StreamOutputT], ABC):
    __slots__: tuple

    def __iter__(self) -> Self:
        return self
    
    def __next__(self) -> StreamOutputT:
        frame = self.read()
        if frame is None:
            raise StopIteration
        
        return frame

    @abstractmethod
    def read(self) -> Optional[StreamOutputT]:
        pass

class IOutStream(IStream, Generic[StreamInputT], ABC):
    __slots__: tuple
    
    @abstractmethod
    def write(self, frame: StreamInputT):
        pass

class IFileStreamPairFactory(Generic[StreamInputT, StreamOutputT], ABC):

    @abstractmethod
    def __call__(self, in_path: str, out_path: str) -> tuple[IInStream[StreamInputT], IOutStream[StreamOutputT]]:
        pass

class IVideoStream(IStream):
    __slots__: tuple

    size: Size
    fps: float

class IVideoInStream(IVideoStream, IInStream[Image]):
    __slots__: tuple

class IVideoOutStream(IVideoStream, IOutStream[Image]):
    __slots__: tuple

FileVideoInStreamFactory = Callable[[str], IVideoInStream]
FileVideoOutStreamFactory = Callable[[str, float, Size], IVideoOutStream]

from defaults.streams import DEFAULT_FILE_IN_STREAM_FACTORY, DEFAULT_FILE_OUT_STREAM_FACTORY, DEFAULT_FILE_IN_STREAM_PATH, DEFAULT_FILE_OUT_STREAM_PATH

@dataclass
class FileVideoStreamPairFactory(IFileStreamPairFactory[Image, Image]):
    stream_in_factory: FileVideoInStreamFactory = DEFAULT_FILE_IN_STREAM_FACTORY
    stream_out_factory: FileVideoOutStreamFactory = DEFAULT_FILE_OUT_STREAM_FACTORY
    default_in_path: str = DEFAULT_FILE_IN_STREAM_PATH
    default_out_path: str = DEFAULT_FILE_OUT_STREAM_PATH
    
    def __call__(self, in_path: str, out_path: str) -> tuple[IVideoInStream, IVideoOutStream]:
        in_stream = self.stream_in_factory(in_path)
        out_stream = self.stream_out_factory(out_path, in_stream.fps, in_stream.size)

        return in_stream, out_stream

get_file_video_stream_pair = FileVideoStreamPairFactory()