
from dataclasses import dataclass, field
import tensorflow as tf

from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig

logger = configure_logger(__name__)

@dataclass
class PrepareModel:
    """
    Componente para preparar o modelo de classificação CNN.
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    image_config: ImageConfig = field(default_factory=ImageConfig)

    def build_model(self) -> tf.keras.Model:
        """Constrói e retorna um modelo Keras simples para classificação de imagens."""
        modelo = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.image_config.size_tuple),
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.model_config.num_classes, activation=tf.nn.softmax)
        ])
        return modelo