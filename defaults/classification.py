import os

from classification.models import ClassificationModelConfig, ClassificationModel

from keras_models.classification import KerasClassificationModel, KerasClassificationModelConfig, KERAS_CLASSIFICATION_MODEL_TYPE

DEFAULT_MODEL_CLS: type[ClassificationModel]

DEFAULT_MODEL_CONFIG: ClassificationModelConfig


DEFAULT_MODEL_CLS = KerasClassificationModel

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "models", "czbirds")

DEFAULT_MODEL_CONFIG = KerasClassificationModelConfig.from_dir(DEFAULT_MODEL_PATH)

DEFAULT_MODEL_TYPE = KERAS_CLASSIFICATION_MODEL_TYPE

CLASSIFICATION_MODEL_TYPES = [
    KERAS_CLASSIFICATION_MODEL_TYPE
]