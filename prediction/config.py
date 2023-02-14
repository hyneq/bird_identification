from dataclasses import dataclass
from typing import Optional, Generic

from .models import TPredictionModelConfig
from . import classes

@dataclass
class PredictorConfig(Generic[TPredictionModelConfig]):
    model_config: TPredictionModelConfig
    min_confidence: float = classes.DEFAULT_MIN_CONFIDENCE
    classification_mode: classes.ClassificationMode