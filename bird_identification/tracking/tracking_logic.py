from typing import TypeVar, Generic
from abc import ABC, abstractmethod

from ..factories import IFactory, MultiFactory

from .logger import IObjectLogger, LoggedObjectT


TrackingLogicInputT = TypeVar("TrackingLogicInputT")

class ITrackingLogic(ABC, Generic[TrackingLogicInputT]):
    __slots__: tuple

    @abstractmethod
    def update(self,
        result: TrackingLogicInputT
    ):
        pass


class ITrackingLogicFactory(IFactory[ITrackingLogic[TrackingLogicInputT]]):
    __slots__: tuple


class ILoggingTrackingLogic(ITrackingLogic[TrackingLogicInputT], Generic[TrackingLogicInputT, LoggedObjectT]):
    __slots__: tuple

    logger: IObjectLogger


class ILoggingTrackingLogicFactory(IFactory[ILoggingTrackingLogic[TrackingLogicInputT, LoggedObjectT]]):
    pass


class MultiLoggingTrackingLogicFactory(
    MultiFactory[
        ILoggingTrackingLogic[TrackingLogicInputT, LoggedObjectT]
    ],
    ILoggingTrackingLogicFactory[TrackingLogicInputT, LoggedObjectT]
):
    pass
