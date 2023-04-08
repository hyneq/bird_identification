import os

from classification.models import ClassificationModelConfig, ClassificationModel

from keras_models.classification import KERAS_CLASSIFICATION_MODEL_FACTORY

DEFAULT_MODEL_CLS: type[ClassificationModel]

DEFAULT_MODEL_CONFIG: ClassificationModelConfig

MODEL_FACTORIES = [
    KERAS_CLASSIFICATION_MODEL_FACTORY
]

DEFAULT_MODEL_FACTORY = KERAS_CLASSIFICATION_MODEL_FACTORY.name

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), os.path.pardir, "models", "czbirds")