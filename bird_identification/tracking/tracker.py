from typing import Generic, TypeVar, Any, Protocol, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..prediction.predictor import PredictionResultT
from ..factories import IFactory

from .tracking_logic import (
    ITrackingLogic,
    TrackingLogicInputT,
    ITrackingLogicFactory,
    ILoggingTrackingLogic,
    ILoggingTrackingLogicFactory
)
from .logger import IObjectLogger, LoggedObjectT, IObjectLoggerFactory

class ITracker(ABC, Generic[PredictionResultT]):
    __slots__: tuple

    @abstractmethod
    def update(self,
        result: PredictionResultT 
    ):
        pass


class ITrackerFactory(IFactory[ITracker[PredictionResultT]]):
    __slots__: tuple


class IPredictionParser(Protocol, Generic[PredictionResultT, TrackingLogicInputT]):

    @abstractmethod
    def __call__(self, result: PredictionResultT) -> TrackingLogicInputT:
        pass


DEFAULT_PREDICTION_PARSER: IPredictionParser = lambda result: result


class Tracker(ITracker[PredictionResultT], Generic[PredictionResultT, TrackingLogicInputT]):
    __slots__: tuple

    logic: ITrackingLogic[TrackingLogicInputT]
    prediction_parser: IPredictionParser[PredictionResultT, TrackingLogicInputT]

    def __init__(self,
        logic: ITrackingLogic[TrackingLogicInputT], 
        prediction_parser: IPredictionParser[PredictionResultT, TrackingLogicInputT] = DEFAULT_PREDICTION_PARSER
    ):
        self.logic = logic
        self.prediction_parser = prediction_parser


    def _parse(self, result: PredictionResultT) -> TrackingLogicInputT:
        return self.prediction_parser(result)


    def update(self, 
        result: PredictionResultT
    ):
        self.logic.update(self._parse(result))


@dataclass(frozen=True)
class TrackerFactory(IFactory[Tracker[PredictionResultT, TrackingLogicInputT]]):

    logic_factory: ITrackingLogicFactory[TrackingLogicInputT]
    default_prediction_parser: IPredictionParser[PredictionResultT, TrackingLogicInputT]

    def __call__(self,
        *args,
        logic: Optional[ITrackingLogic[TrackingLogicInputT]] = None,
        prediction_parser: Optional[IPredictionParser[PredictionResultT, TrackingLogicInputT]] = None,
        **kwargs
    ) -> Tracker[PredictionResultT, TrackingLogicInputT]:
        if not logic:
            logic = self.logic_factory(*args, **kwargs)
        
        if not prediction_parser:
            prediction_parser = self.default_prediction_parser
        
        return Tracker(logic, prediction_parser)


class LoggingTracker(Tracker[PredictionResultT, TrackingLogicInputT], Generic[PredictionResultT, TrackingLogicInputT, LoggedObjectT]):
    __slots__: tuple

    logic: ILoggingTrackingLogic[TrackingLogicInputT, LoggedObjectT]
    logger: IObjectLogger[LoggedObjectT]

    def __init__(self,
        logic: ILoggingTrackingLogic[TrackingLogicInputT, LoggedObjectT],
        logger: IObjectLogger[LoggedObjectT],
        prediction_parser: IPredictionParser[PredictionResultT, TrackingLogicInputT] = DEFAULT_PREDICTION_PARSER,
    ):
        super().__init__(logic, prediction_parser)

        self.logger = logger
        logic.logger = logger


    def log(self):
        self.logger.log()


@dataclass(frozen=True)
class LoggingTrackerFactory(TrackerFactory[PredictionResultT, TrackingLogicInputT], Generic[PredictionResultT, TrackingLogicInputT, LoggedObjectT]):

    logic_factory: ILoggingTrackingLogicFactory[TrackingLogicInputT, LoggedObjectT]
    logger_factory: IObjectLoggerFactory[LoggedObjectT]

    def __call__(self,
        *args,
        logic: Optional[ILoggingTrackingLogic[TrackingLogicInputT, LoggedObjectT]] = None,
        logger: Optional[IObjectLogger[LoggedObjectT]] = None,
        prediction_parser: Optional[IPredictionParser[PredictionResultT, TrackingLogicInputT]] = None,
        **kwargs
    ) -> LoggingTracker[PredictionResultT, TrackingLogicInputT, LoggedObjectT]:
        if not logic:
            logic = self.logic_factory(*args, **kwargs)
        
        if not logger:
            logger = self.logger_factory()

        if not prediction_parser:
            prediction_parser = self.default_prediction_parser
        
        return LoggingTracker(logic, logger, prediction_parser)
