import os

from classification.models import ClassificationModelConfig, ClassificationModel

from keras.prediction import KerasModelConfig
from keras.classification import KerasClassificationModel

DEFAULT_MODEL_CLS: type[ClassificationModel] = KerasClassificationModel

DEFAULT_MODEL_CONFIG_CLS = ClassificationModelConfig

DEFAULT_MODEL_PATH: str = os.path.join(os.path.dirname(__file__), "models", "czbirds")

DEFAULT_MODEL_CONFIG: DEFAULT_MODEL_CONFIG_CLS = KerasModelConfig.from_dir(DEFAULT_MODEL_PATH)