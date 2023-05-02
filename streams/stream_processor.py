from typing import Generic, TypeVar
from abc import ABC, abstractmethod

from . import IInStream, IOutStream

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
FrameT = TypeVar("FrameT")

class IFrameProcessor(Generic[InputT, OutputT], ABC):

    @abstractmethod
    def process(self, input: InputT) -> OutputT:
        pass

ISameTypeFrameProcessor = IFrameProcessor[FrameT, FrameT]

class MultiFrameProcessor(ISameTypeFrameProcessor[FrameT]):

    frame_processors: list[ISameTypeFrameProcessor[FrameT]]

    def __init__(self, frame_processors: list[ISameTypeFrameProcessor[FrameT]]):
        self.frame_processors = frame_processors
    
    def process(self, input: FrameT) -> FrameT:
        frame = input
        for frame_processor in self.frame_processors:
            frame = frame_processor.process(frame)
        
        return frame

class StreamProcessor(Generic[InputT, OutputT]):

    input_stream: IInStream[InputT]
    output_stream: IOutStream[OutputT]

    frame_processor: IFrameProcessor[InputT, OutputT]

    def __init__(self, input_stream: IInStream[InputT], output_stream: IOutStream[OutputT], frame_processor: IFrameProcessor[InputT, OutputT]):
        self.input_stream = input_stream
        self.output_stream = output_stream

        self.frame_processor = frame_processor
    
    def loop(self):
        for input in self.input_stream:
            output = self.process(input)
            self.output_stream.write(output)

    def process(self, input: InputT) -> OutputT:
        return self.frame_processor.process(input)

ISameTypeStreamProcessor = StreamProcessor[FrameT, FrameT]