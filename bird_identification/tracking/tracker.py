from typing import Generic, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..prediction.predictor import PredictionResultT
from ..factories import IFactory

from .logger import IObjectLogger, LoggedObjectT, IObjectLoggerFactory

class ITracker(ABC, Generic[PredictionResultT]):
    __slots__: tuple

    @abstractmethod
    def update(self,
        result: PredictionResultT 
    ):
        pass


class ITrackerFactory(IFactory[ITracker[PredictionResultT]]):
    pass


class ALoggingTrackingLogic(ABC, Generic[PredictionResultT, LoggedObjectT]):
    __slots__: tuple

    tracker: "LoggingTracker[PredictionResultT, LoggedObjectT]"

    @abstractmethod
    def update(self, results: PredictionResultT):
        pass


    def log(self, objects: list[LoggedObjectT]):
        self.tracker.log(objects)


class ALoggingTrackingLogicFactory(IFactory[ALoggingTrackingLogic[PredictionResultT, LoggedObjectT]]):
    pass


class LoggingTracker(ITracker[PredictionResultT], Generic[PredictionResultT, LoggedObjectT]):
    __slots__: tuple

    logic: ALoggingTrackingLogic[PredictionResultT, LoggedObjectT]
    logger: IObjectLogger[LoggedObjectT]

    def __init__(self,
        logic: ALoggingTrackingLogic[PredictionResultT, LoggedObjectT],
        logger: IObjectLogger[LoggedObjectT]
    ):
        self.logic = logic
        self.logger = logger
        
        logic.tracker = self
    

    def update(self, result: PredictionResultT):
        self.logic.update(result)


    def log(self, objects: list[LoggedObjectT]):
        self.logger.log(objects)


@dataclass(frozen=True)
class LoggingTrackerFactory(ITrackerFactory[PredictionResultT], Generic[PredictionResultT, LoggedObjectT]):

    logic_factory: ALoggingTrackingLogicFactory[PredictionResultT, LoggedObjectT]
    logger_factory: IObjectLoggerFactory[LoggedObjectT]

    def __call__(self,
        logic: Optional[ALoggingTrackingLogic[PredictionResultT, LoggedObjectT]] = None,
        logger: Optional[IObjectLogger[LoggedObjectT]] = None
    ):
        logic = logic or self.logic_factory()
        logger = logger or self.logger_factory()

        return LoggingTracker(logic, logger)
