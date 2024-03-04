from abc import ABC, abstractmethod
from typing import Generic, Optional

from . import (
    ArgumentParser,
    ICLIPart,
    CLIPart,
    args_required,
)

from ..streams import (
    InStreamT,
    OutStreamT,
    IVideoInStream,
    IVideoOutStream,
    MultiPathVideoStreamPairFactory,
    get_path_video_stream_pair,
)

from ..defaults.streams import DEFAULT_PATH_IN_STREAM_PATH, DEFAULT_PATH_OUT_STREAM_FACTORY


class IStreamPairCLIPart(ICLIPart, ABC, Generic[InStreamT, OutStreamT]):
    @args_required
    @abstractmethod
    def get_stream_pair(self) -> tuple[InStreamT, OutStreamT]:
        pass


class MultiPathVideoStreamPairCLIPart(CLIPart, IStreamPairCLIPart):
    stream_pair_factory: MultiPathVideoStreamPairFactory

    def __init__(
        self,
        stream_pair_factory: MultiPathVideoStreamPairFactory = get_path_video_stream_pair,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.stream_pair_factory = stream_pair_factory

    def add_opts(self, parser: ArgumentParser):
        parser.add_argument("-i", "--in", "--in-path", dest="in_path", default=DEFAULT_PATH_IN_STREAM_PATH)
        parser.add_argument("-o", "--out", "--out-path", dest="out_path", default=DEFAULT_PATH_OUT_STREAM_FACTORY)
        parser.add_argument(
            "--in-type",
            dest="in_type",
            default=self.stream_pair_factory.stream_in_factory.default_factory,
        )
        parser.add_argument(
            "--out-type",
            dest="out_type",
            default=self.stream_pair_factory.stream_out_factory.default_factory,
        )
        parser.add_argument(
            "--fps", "--framerate",
            dest="fps",
            type=int,
        )
        parser.add_argument(
            "--width", "--frame-width",
            dest="width",
        )
        parser.add_argument(
            "--height", "--frame-height",
            dest="height",
        )

    @args_required
    def get_stream_pair(self) -> tuple[IVideoInStream, IVideoOutStream]:
        size = None
        if self.args.width:
            if not self.args.height:
                raise ValueError("Width and height must both be specified or not.")
            
            size = (int(self.args.width), int(self.args.height))

        return self.stream_pair_factory(
            fps=self.args.fps,
            size=size,
            in_path=self.args.in_path,
            out_path=self.args.out_path,
            in_type=self.args.in_type,
            out_type=self.args.out_type,
        )
