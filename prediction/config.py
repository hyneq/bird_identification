from dataclasses import dataclass
from typing import Union, Generic

from .models import TPredictionModelConfig
from . import classes

@dataclass
class PredictorConfig(Generic[TPredictionModelConfig]):
    model_config: TPredictionModelConfig
    min_confidence: float = None
    classification_mode: classes.ClassificationMode = None
    classes: Union[list[int],list[str]] = None