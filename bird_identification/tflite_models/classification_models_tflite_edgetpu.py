from dataclasses import dataclass

from ..classification.models import ClassificationModelFactory

from .edgetpu import EdgeTPUTFLiteModelWithClassesConfig
from .classification_models_tflite import TFLiteClassificationModel, TFLiteClassificationModelConfig

@dataclass
class EdgeTPUTFLiteClassificationModelConfig(
    EdgeTPUTFLiteModelWithClassesConfig,
    TFLiteClassificationModelConfig,
):
    pass


factory = ClassificationModelFactory[
    str, EdgeTPUTFLiteClassificationModelConfig
](
    name="tflite_edgetpu",
    model_cls=TFLiteClassificationModel,
    model_config_cls=EdgeTPUTFLiteClassificationModelConfig,
    model_config_loader=EdgeTPUTFLiteClassificationModelConfig.from_path,
    default_model_config_input="models/classification",
)
