from typing import Optional
from typing_extensions import Self
from dataclasses import dataclass
import time
from collections import Counter
from datetime import datetime

from ..prediction.predictor import IPredictionResultWithClasses

from .logger import ClassLoggedObject
from .tracking_logic import ILoggingTrackingLogic, ILoggingTrackingLogicFactory

DEFAULT_IDLE_INTERVAL: float = 15.0

@dataclass()
class TrackedObject:
    class_name: Optional[str]
    start_time: float
    last_seen: float
    end_time: Optional[float] = None
    level: int = 1
    child: Optional[Self] = None
    parent: Optional[Self] = None


class IntervalTrackingLogic(ILoggingTrackingLogic[list[IPredictionResultWithClasses], ClassLoggedObject]):
    __slots__: tuple

    idle_interval: float

    tracked_objects: dict[Optional[str], TrackedObject]
    logged_objects: list[ClassLoggedObject]
    now: float

    def __init__(self, idle_interval: float):
        self.idle_interval = idle_interval

        self.tracked_objects = {}
        self.now = time.time()
    

    def update(self, result: list[IPredictionResultWithClasses]):
        self.now = time.time()
        self.update_classes(result)
        self.check_interval()
    

    def update_classes(self, result: list[IPredictionResultWithClasses]):    
        class_counts = self._get_class_counts(result)
        for class_name, count in class_counts.items():
            self._update_class(class_name, count)
    

    def _get_class_counts(self, result: list[IPredictionResultWithClasses]):
        return Counter(o.class_name for o in result)


    def _update_class(self, class_name: Optional[str], count: int):
        obj = self._update_obj_root(class_name)
        for _ in range(2, count+1):
            obj = self._update_obj_level(obj)


    def _update_obj_root(self, class_name: Optional[str]) -> TrackedObject:
        obj = self.tracked_objects[class_name] = self._update_obj(
            self.tracked_objects.get(class_name),
            class_name
        )

        return obj
    

    def _update_obj_level(self, parent: TrackedObject):
        obj = parent.child = self._update_obj(
            parent.child,
            parent.class_name,
            parent.level+1,
            parent
        )

        return obj
    

    def _update_obj(self,
            obj: Optional[TrackedObject],
            class_name: Optional[str],
            level: int = 1,
            parent: Optional[TrackedObject] = None ) -> TrackedObject:
        if obj:
            obj.last_seen = self.now
        else:
            obj = TrackedObject(
                class_name,
                self.now,
                self.now,
                level=level,
                parent=parent
            )

        return obj


    def check_interval(self):
        for class_name in list(self.tracked_objects.keys()):
            self._check_interval_class(class_name)
    

    def _check_interval_class(self, class_name: Optional[str]=None):
        obj = self.tracked_objects.get(class_name)
        while obj:
            self._check_interval_obj(obj)
            obj = obj.child


    def _check_interval_obj(self, obj: TrackedObject):
        if self._obj_is_idle(obj):
            self._log_and_remove(obj)


    def _obj_is_idle(self, obj: TrackedObject):
        return self.now - obj.last_seen > self.idle_interval * obj.level


    def _log_and_remove(self, obj: Optional[TrackedObject]):
        while obj:
            obj.end_time = self.now
            self._log_object(obj)
            self._remove_object(obj)
            obj = obj.child


    def _log_object(self, obj: TrackedObject):
        logged_obj = ClassLoggedObject(
            obj.class_name,
            datetime.fromtimestamp(obj.start_time),
            datetime.fromtimestamp(obj.end_time)
        )
        self.logger.add(logged_obj)

    
    def _remove_object(self, obj: TrackedObject):
        if obj.parent:
            obj.parent.child = None
        else:
            del self.tracked_objects[obj.class_name]


@dataclass(frozen=True)
class IntervalTrackingLogicFactory(ILoggingTrackingLogicFactory[list[IPredictionResultWithClasses], ClassLoggedObject]):

    name = "interval"
    default_idle_interval: float = DEFAULT_IDLE_INTERVAL

    def __call__(self,
        idle_interval: Optional[float] = None,
        *_, **__
    ) -> ILoggingTrackingLogic[list[IPredictionResultWithClasses], ClassLoggedObject]:
        idle_interval = idle_interval or self.default_idle_interval

        return IntervalTrackingLogic(idle_interval=idle_interval)


factory = IntervalTrackingLogicFactory()

get_interval_tracking_logic = factory
