from typing import Generic, TypeVar, Optional
from typing_extensions import Self 
from abc import ABC, abstractmethod
from dataclasses import dataclass

from image_utils import Image
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