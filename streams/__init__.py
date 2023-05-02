from typing import Generic, TypeVar, Optional, Callable
from typing_extensions import Self 
from abc import ABC, abstractmethod
from dataclasses import dataclass

from image_utils import Image, Size
from . import IInStream, IOutStream

StreamInputT = TypeVar("StreamInputT")
StreamOutputT = TypeVar("StreamOutputT")

class IInStream(Generic[StreamOutputT], ABC):
    def __iter__(self) -> Self:
        return self
    
    def __next__(self) -> StreamOutputT:
        return self.read()

    @abstractmethod
    def read(self) -> StreamOutputT:
        pass

class IOutStream(Generic[StreamInputT], ABC):
    
    @abstractmethod
    def write(self, frame: StreamInputT):
        pass

class IFileStreamPairFactory(Generic[StreamInputT, StreamOutputT], ABC):

    @abstractmethod
    def __call__(self, in_path: str, out_path: str) -> tuple[IInStream[StreamInputT], IOutStream[StreamOutputT]]:
        pass


class IVideoInStream(IInStream[Image]):

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        pass

class IVideoOutStream(IOutStream[Image]):

    @property
    @abstractmethod
    def size(self) -> tuple[int, int]:
        pass

FileVideoInStreamFactory = Callable[[str], IVideoInStream]
FileVideoOutStreamFactory = Callable[[str, Size], IVideoOutStream]

from defaults.streams import DEFAULT_FILE_IN_STREAM_FACTORY, DEFAULT_FILE_OUT_STREAM_FACTORY, DEFAULT_FILE_IN_STREAM_PATH, DEFAULT_FILE_OUT_STREAM_PATH

@dataclass
class FileVideoStreamPairFactory(IFileStreamPairFactory[Image, Image]):
    stream_in_factory: FileVideoInStreamFactory = DEFAULT_FILE_IN_STREAM_FACTORY
    stream_out_factory: FileVideoOutStreamFactory = DEFAULT_FILE_OUT_STREAM_FACTORY
    default_in_path: str = DEFAULT_FILE_IN_STREAM_PATH
    default_out_path: str = DEFAULT_FILE_OUT_STREAM_PATH
    
    def __call__(self, in_path: str, out_path: str) -> tuple[IVideoInStream, IVideoOutStream]:
        in_stream = self.stream_in_factory(in_path)
        out_stream = self.stream_out_factory(out_path, in_stream.size)

        return in_stream, out_stream

get_file_video_stream_pair = FileVideoStreamPairFactory()