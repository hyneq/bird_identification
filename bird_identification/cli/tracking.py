from abc import ABC, abstractmethod
from typing import Generic, Optional

from . import (
    ArgumentParser,
    ICLIPart,
    CLIPart,
    args_required,
)

from ..tracking.tracker import (
    ITracker,
    MultiLoggingTrackerFactory,
    PredictionResultT,
    TrackingLogicInputT,
    LoggedObjectT,
    tracker_factory
)


class ITrackerCLIPart(ICLIPart, ABC, Generic[PredictionResultT]):
    @args_required
    @abstractmethod
    def get_tracker(self) -> ITracker[PredictionResultT]:
        pass


class MultiLoggingTrackingLogicCLIPart(CLIPart, ITrackerCLIPart[PredictionResultT], Generic[PredictionResultT, TrackingLogicInputT, LoggedObjectT]):
    factory: MultiLoggingTrackerFactory

    def __init__(self,
        factory: MultiLoggingTrackerFactory = tracker_factory,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.factory = factory
    

    def add_opts(self, parser: ArgumentParser):
        parser.add_argument(
            "--logic-type",
            dest="logic_type",
            default=self.factory.logic_factory.default_factory,
            choices=self.factory.logic_factory.factory_names
        )
        parser.add_argument(
            "--idle-interval",
            dest="idle_interval",
            required=False
        )
        parser.add_argument(
            "--logger-type",
            dest="logger_type",
            default=self.factory.logger_factory.default_factory,
            choices=self.factory.logger_factory.factory_names
        )
        parser.add_argument(
            "--log-path",
            dest="log_path",
            required=False
        )


    @args_required
    def get_tracker(self) -> ITracker[PredictionResultT]:
        logic_kwargs = {
            "idle_interval": self.args.idle_interval
        }

        logger_kwargs = {
            "path": self.args.log_path
        }

        return self.factory(
            logic_type=self.args.logic_type,
            logger_type=self.args.logger_type,
            logic_kwargs=logic_kwargs,
            logger_kwargs=logger_kwargs
        )
