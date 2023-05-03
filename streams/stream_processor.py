from typing import Generic, TypeVar, Optional, Callable
from abc import ABC, abstractmethod
import copy
import time

from . import IInStream, IOutStream

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
FrameT = TypeVar("FrameT")

class IFrameProcessor(Generic[InputT, OutputT], ABC):
    __slots__: tuple

    @abstractmethod
    def process(self, input: InputT) -> OutputT:
        pass

ISameTypeFrameProcessor = IFrameProcessor[FrameT, FrameT]

CopyStrategy = Callable[[FrameT], FrameT]
class FrameCache(ISameTypeFrameProcessor[FrameT]):
    __slots__: tuple

    cached: Optional[FrameT]
    copy_strategy: CopyStrategy[FrameT]

    def __init__(self, copy_strategy: CopyStrategy[FrameT] = copy.copy):
        self.cached = None
        self.copy_strategy = copy_strategy
    
    def process(self, input: FrameT) -> FrameT:
        self.cached = self.copy_strategy(input)

        return input

class MultiFrameProcessor(ISameTypeFrameProcessor[FrameT]):
    __slots__: tuple

    frame_processors: list[ISameTypeFrameProcessor[FrameT]]

    def __init__(self, frame_processors: list[ISameTypeFrameProcessor[FrameT]]):
        self.frame_processors = frame_processors
    
    def process(self, input: FrameT) -> FrameT:
        frame = input
        for frame_processor in self.frame_processors:
            frame = frame_processor.process(frame)
        
        return frame

class StreamProcessor(Generic[InputT, OutputT]):
    __slots__: tuple

    input_stream: IInStream[InputT]
    output_stream: IOutStream[OutputT]

    frame_processor: IFrameProcessor[InputT, OutputT]

    def __init__(self, input_stream: IInStream[InputT], output_stream: IOutStream[OutputT], frame_processor: IFrameProcessor[InputT, OutputT]):
        self.input_stream = input_stream
        self.output_stream = output_stream

        self.frame_processor = frame_processor
    
    def run(self):
        try:
            self.loop()
        finally:
            self.input_stream.close()
            self.output_stream.close()
    
    def loop(self):
        for input in self.input_stream:
            output = self.process(input)
            self.output_stream.write(output)

    def process(self, input: InputT) -> OutputT:
        return self.frame_processor.process(input)

SameTypeStreamProcessor = StreamProcessor[FrameT, FrameT]