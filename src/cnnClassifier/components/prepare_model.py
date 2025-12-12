from typing import Optional
import tensorflow as tf

from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig

logger = configure_logger(__name__)


class PrepareModel:
    """
    Componente para preparar o modelo de classificação CNN.
    """

    def __init__(
        self, model_config: ModelConfig = None, image_config: ImageConfig = None
    ):
        """
        Inicializa o componente de preparação de modelo.

        Args:
            model_config: Configuração do modelo
            image_config: Configuração da imagem
        """

        if model_config is None:
            raise ValueError(
                " ModelConfig é obrigatória! Use ModelConfig.from_yaml() "
                "ou forneça configuração explícita."
            )

        if image_config is None:
            raise ValueError(
                "ImageConfig é obrigatória! Forneça configuração explícita."
            )

        self.model_config = model_config
        self.image_config = image_config

        if not hasattr(self.model_config, "augmentation_enabled"):
            raise ValueError("'augmentation_enabled' não encontrado no YAML!")
        self.data_augmentation = self._data_augmentation()

    def build_model(self) -> tf.keras.Model:
        """Constrói modelo - custom ou pré-treinado baseado na configuração"""
        logger.info(f"🏗️ Construindo modelo {self.model_config.model_name}")

        # Verificar se deve usar modelo pré-treinado
        use_pretrained = getattr(self.model_config, "use_pretrained", False)

        if use_pretrained:
            return self._build_pretrained_model()
        else:
            return self._build_custom_model()

    def _build_pretrained_model(self) -> tf.keras.Model:
        """
        Constrói e retorna um modelo Keras pré-treinado para classificação de imagens.

        Returns:
            tf.keras.Model: Modelo compilado pronto para treinamento
        """
        modelo_base = self._pretrained_model(
            include_top=False, weights=self.model_config.weights
        )

        modelo_base.trainable = False
        logger.info(f"🔒 Modelo base congelado: {len(modelo_base.layers)} camadas")

        inputs = tf.keras.layers.Input(shape=self.image_config.size_tuple)
        x = self.data_augmentation(inputs)

        # x = tf.keras.layers.Rescaling(1./255)(x) # Descomentar se necessário em algum modelo (MobileNet já faz isso internamente)

        x = modelo_base(x, training=False)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation="relu")(x)
        x = tf.keras.layers.Dropout(self.model_config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(
            self.model_config.num_classes, activation="softmax"
        )(x)

        modelo = tf.keras.Model(inputs=inputs, outputs=outputs)
        logger.info(f"Dropout rate aplicado: {self.model_config.dropout_rate}")
        logger.info(
            f"✅ Utilizando modelo pré-treinado {self.model_config.model_name} construído"
        )
        return modelo

    def _build_custom_model(
        self, activation_last_layer: str = "softmax"
    ) -> tf.keras.Model:
        """
        Constrói e retorna um modelo Keras para classificação de imagens.

        Returns:
            tf.keras.Model: Modelo compilado pronto para treinamento
        """
        logger.info("🔨 Construindo modelo customizado CNN")

        modelo = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=self.image_config.size_tuple),
                self.data_augmentation,
                tf.keras.layers.Rescaling(1.0 / 255),
                # Primeira camada convolucional
                tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # Segunda camada convolucional
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # Terceira camada convolucional
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(2, 2),
                # Flatten e Dense layers
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(self.model_config.dropout_rate),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dropout(self.model_config.dropout_rate),
                tf.keras.layers.Dense(
                    self.model_config.num_classes, activation=activation_last_layer
                ),
            ]
        )

        logger.info(f"✅ Modelo {self.model_config.model_name} construído com sucesso")
        return modelo

    def _data_augmentation(self):
        """
        Cria pipeline de data augmentation baseado na configuração.

        Returns:
            tf.keras.Sequential: Pipeline de augmentação
        """
        layers = []

        if (
            hasattr(self.model_config, "horizontal_flip")
            and self.model_config.horizontal_flip
        ):
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
            logger.info("🔄Data Augmentation: RandomFlip horizontal adicionado")

        if hasattr(self.model_config, "zoom_factor"):
            layers.append(tf.keras.layers.RandomZoom(self.model_config.zoom_factor))
            logger.info("🔄Data Augmentation: RandomZoom adicionado")

        if hasattr(self.model_config, "brightness_range"):
            brightness = self.model_config.brightness_range
            layers.append(
                tf.keras.layers.RandomBrightness(
                    factor=(brightness[0] - 1.0, brightness[1] - 1.0)
                )
            )
            logger.info("🔄Data Augmentation: RandomBrightness adicionado")

        if hasattr(self.model_config, "zoom_range"):
            layers.append(tf.keras.layers.RandomZoom(self.model_config.zoom_range))
            logger.info("🔄Data Augmentation: RandomZoom adicionado")

        if hasattr(self.model_config, "rotation_range"):
            rotation = self.model_config.rotation_range / 360.0  # Converter para fração
            layers.append(tf.keras.layers.RandomRotation(rotation))
            logger.info("🔄Data Augmentation: RandomRotation adicionado")

        data_augmentation = tf.keras.Sequential(layers)
        logger.info(f"🔄Data augmentation configurado")

        return data_augmentation

    def _pretrained_model(self, include_top: bool = False, weights: str = "imagenet"):
        """
        Cria modelo pré-treinado baseado no modelo.

        Args:
            include_top:
            weights:

        Returns:
            tf.keras.Model: Modelo base pré-treinado
        """
        model_name = self.model_config.model_name.lower()

        if "mobilenet" in model_name:
            modelo_base = tf.keras.applications.MobileNetV3Large(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights,
            )
        elif "inception" in model_name:
            modelo_base = tf.keras.applications.InceptionV3(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights,
            )
        elif "vgg" in model_name:
            modelo_base = tf.keras.applications.VGG19(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights,
            )
        else:
            # Default para MobileNet
            modelo_base = tf.keras.applications.MobileNetV3Large(
                input_shape=self.image_config.size_tuple,
                include_top=include_top,
                weights=weights,
            )

        logger.info(f"🏗️ Modelo base {model_name} carregado")
        return modelo_base
