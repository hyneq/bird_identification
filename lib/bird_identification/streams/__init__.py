from typing import Generic, TypeVar, Optional, Callable, Protocol, Any, overload
from typing_extensions import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time

from image_utils import Image, Size
from factories import MultiFactory

StreamInputT = TypeVar("StreamInputT")
StreamOutputT = TypeVar("StreamOutputT")

class StreamError(Exception):
    pass

class IStream(Protocol):
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

StreamT = TypeVar("StreamT", bound=IStream, covariant=True)
InStreamT = TypeVar("InStreamT", bound=IInStream, covariant=True)
OutStreamT = TypeVar("OutStreamT", bound=IOutStream, covariant=True)

class IStreamFactory(Protocol[StreamT]):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> StreamT:
        pass

class IInStreamFactory(IStreamFactory[InStreamT]):
    pass

class IOutStreamFactory(IStreamFactory[OutStreamT]):
    pass

class IStreamPairFactory(Protocol[InStreamT, OutStreamT]):
    @abstractmethod
    def __call__(self, *args,
            in_args: tuple=tuple(), in_kwargs: dict[str, Any]={},
            out_args: tuple=tuple(), out_kwargs: dict[str, Any]={},
            **kwargs
    ) -> tuple[InStreamT, OutStreamT]:
        pass

class IPathStreamFactory(IStreamFactory[StreamT]):
    @abstractmethod
    def __call__(self, path, *args, **kwargs) -> StreamT:
        pass

class IPathInStreamFactory(IPathStreamFactory[InStreamT], IInStreamFactory[InStreamT]):
    pass

class IPathOutStreamFactory(IPathStreamFactory[OutStreamT], IOutStreamFactory[OutStreamT]):
    pass

class IPathStreamPairFactory(IStreamPairFactory[InStreamT, OutStreamT]):
    @abstractmethod
    def __call__(self, *args,
            in_path: Optional[str]=None, out_path: Optional[str]=None,
            in_args: tuple=tuple(), in_kwargs: dict[str, Any]={},
            out_args: tuple=tuple(), out_kwargs: dict[str, Any]={},
            **kwargs
    ) -> tuple[InStreamT, OutStreamT]:
        pass

class IVideoStream(IStream):
    __slots__: tuple

    size: Size
    fps: float

class IVideoInStream(IVideoStream, IInStream[Optional[Image]]):
    pass

class IVideoOutStream(IVideoStream, IOutStream[Image]):
    pass

VideoStreamT = TypeVar("VideoStreamT", bound=IVideoStream, covariant=True)
VideoInStreamT = TypeVar("VideoInStreamT", bound=IVideoInStream, covariant=True)
VideoOutStreamT = TypeVar("VideoOutStreamT", bound=IVideoOutStream, covariant=True)

class IVideoStreamFactory(IStreamFactory[VideoStreamT]):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> VideoStreamT:
        pass

class IVideoInStreamFactory(IVideoStreamFactory, IInStreamFactory[VideoInStreamT]):
    @abstractmethod
    def __call__(self, fps: Optional[float]=None, size: Optional[Size]=None, *args, **kwargs) -> VideoInStreamT:
        pass

class IVideoOutStreamFactory(IVideoStreamFactory, IOutStreamFactory[VideoOutStreamT]):
    @abstractmethod
    def __call__(self, fps: float, size: Size, *args, **kwargs) -> VideoOutStreamT:
        pass

class IVideoStreamPairFactory(IStreamPairFactory[VideoInStreamT, VideoOutStreamT]):
    @abstractmethod
    def __call__(self, *args,
            fps: Optional[float]=None, size: Optional[Size]=None,
            in_args: tuple=tuple(), in_kwargs: dict[str, Any]={},
            out_args: tuple=tuple(), out_kwargs: dict[str, Any]={},
            **kwargs
    ) -> tuple[VideoInStreamT, VideoOutStreamT]:
        pass

class IPathVideoStreamFactory(IVideoStreamFactory[VideoStreamT]):
    @abstractmethod
    def __call__(self, *args, path: str, fps: Optional[float]=None, size: Optional[Size]=None, **kwargs) -> VideoStreamT:
        pass

class IPathVideoInStreamFactory(IPathVideoStreamFactory[VideoInStreamT], IVideoInStreamFactory[VideoInStreamT]):
    pass

class IPathVideoOutStreamFactory(IPathVideoStreamFactory[VideoOutStreamT], IVideoOutStreamFactory[VideoOutStreamT]):
    @abstractmethod
    def __call__(self, path: str, fps: float, size: Size, *args, **kwargs) -> VideoOutStreamT:
        pass

from defaults import streams as stream_defaults

@dataclass(frozen=True)
class AStreamPairFactory(IStreamPairFactory[InStreamT, OutStreamT], ABC):
    stream_in_factory: IInStreamFactory[InStreamT]
    stream_out_factory: IOutStreamFactory[OutStreamT]

@dataclass(frozen=True)
class AVideoStreamPairFactory(AStreamPairFactory[VideoInStreamT, VideoOutStreamT], IVideoStreamPairFactory[VideoInStreamT, VideoOutStreamT]):
    stream_in_factory: IVideoInStreamFactory[VideoInStreamT]
    stream_out_factory: IVideoOutStreamFactory[VideoOutStreamT]

    @abstractmethod
    def __call__(self, *_,
            fps: Optional[float]=None, size: Optional[Size]=None,
            in_args: tuple=tuple(), in_kwargs: dict[str, Any]={},
            out_args: tuple=tuple(), out_kwargs: dict[str, Any]={},
            **__
    ) -> tuple[VideoInStreamT, VideoOutStreamT]:
        in_stream = self.stream_in_factory(*in_args, fps=fps, size=size, **in_kwargs)
        out_stream = self.stream_out_factory(*out_args, fps=in_stream.fps, size=in_stream.size, **out_kwargs)

        return in_stream, out_stream


class MultiPathVideoStreamFactory(
        MultiFactory[VideoStreamT],
        IPathVideoStreamFactory[VideoStreamT]
    ):
    pass

class MultiPathVideoInStreamFactory(
        MultiPathVideoStreamFactory[IVideoInStream],
        IVideoInStreamFactory[IVideoInStream],
    ):
    pass

class MultiPathVideoOutStreamFactory(
        MultiPathVideoStreamFactory[IVideoOutStream],
        IVideoOutStreamFactory[IVideoOutStream],
    ):
    pass

STREAM_IN_FACTORY = MultiPathVideoInStreamFactory(
    stream_defaults.PATH_IN_STREAM_FACTORIES,
    stream_defaults.DEFAULT_PATH_IN_STREAM_FACTORY
)

STREAM_OUT_FACTORY = MultiPathVideoOutStreamFactory(
    stream_defaults.PATH_OUT_STREAM_FACTORIES,
    stream_defaults.DEFAULT_PATH_OUT_STREAM_FACTORY
)

@dataclass(frozen=True)
class MultiPathVideoStreamPairFactory(AVideoStreamPairFactory[IVideoInStream, IVideoOutStream], IPathStreamPairFactory[IVideoInStream, IVideoOutStream]):
    stream_in_factory: MultiPathVideoInStreamFactory = STREAM_IN_FACTORY
    stream_out_factory: MultiPathVideoOutStreamFactory = STREAM_OUT_FACTORY
    
    def __call__(self, *_,
            in_type: Optional[str]=None, out_type: Optional[str]=None,
            in_path: Optional[str]=None, out_path: Optional[str]=None,
            fps: Optional[float]=None, size: Optional[Size]=None,
            in_args: tuple=tuple(), in_kwargs: dict[str, Any]={},
            out_args: tuple=tuple(), out_kwargs: dict[str, Any]={},
            **__
    ) -> tuple[IVideoInStream, IVideoOutStream]:
        return super().__call__(
            fps=fps, size=size,
            in_args=in_args, in_kwargs={
                "factory": in_type,
                "path": in_path,
                **in_kwargs
            },
            out_args=out_args, out_kwargs={
                "factory": out_type,
                "path": out_path,
                **out_kwargs
            },
        )

get_path_video_stream_pair = MultiPathVideoStreamPairFactory()