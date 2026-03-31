import tensorflow as tf

from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ModelConfig

logger = configure_logger(__name__)

class AugmentationPipeline:

    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def data_augmentation(self) -> tf.keras.Sequential:
        """
        Cria pipeline de data augmentation baseado na configuração.

        Returns:
            tf.keras.Sequential: Pipeline de augmentação
        """
        seed = self.model_config.random_seed
        layers = []

        # Respeita flag de habilitação
        if (
            hasattr(self.model_config, "augmentation_enabled")
            and not self.model_config.augmentation_enabled
        ):
            logger.info("🔒 Data Augmentation desabilitado via config")
            # layers.append(tf.keras.layers.Lambda(lambda x: x))
            # return tf.keras.layers.Activation("linear")
            return tf.keras.Sequential(layers)

        # if (
        #     hasattr(self.model_config, "gaussian_noise")
        #     and self.model_config.gaussian_noise
        # ):
        #     noise_std = self.model_config.gaussian_noise
        # else:
        #     noise_std = 0.05
        # logger.info(f"Adicionando Gaussian Noise: {noise_std}")
        # layers.append(tf.keras.layers.GaussianNoise(noise_std))
        # x = tf.keras.layers.GaussianNoise(noise_std)(x)

        if (
            hasattr(self.model_config, "horizontal_flip")
            and self.model_config.horizontal_flip
        ):
            layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
            logger.info("Data Augmentation: RandomFlip horizontal adicionado")

        if hasattr(self.model_config, "zoom_factor"):
            layers.append(
                tf.keras.layers.RandomZoom(self.model_config.zoom_factor, seed=seed)
            )
            logger.info("Data Augmentation: RandomZoom adicionado")

        # NOTE: brightness_range está piorando os resultados para este dataset (soybean leaf pest damage)
        # if hasattr(self.model_config, "brightness_range"):
        #     brightness = self.model_config.brightness_range
        #     layers.append(tf.keras.layers.RandomBrightness(factor=brightness,seed=seed))
        #     logger.info("Data Augmentation: RandomBrightness adicionado")

        if (
            hasattr(self.model_config, "contrast_range")
            and self.model_config.contrast_range
        ):
            contrast_range = self.model_config.contrast_range
            logger.info("Data Augmentation: RandomContrast (config) adicionado")
        else:
            contrast_range = [0.1, 1.1]
            logger.info("Data Augmentation: RandomContrast default adicionado")
        layers.append(tf.keras.layers.RandomContrast(factor=contrast_range, seed=seed))

        rotation = None
        if (
            hasattr(self.model_config, "rotation_range")
            and self.model_config.rotation_range
        ):
            rotation = self.model_config.rotation_range / 360.0
        elif (
            hasattr(self.model_config, "rotation_factor")
            and self.model_config.rotation_factor
        ):
            rotation = self.model_config.rotation_factor

        if rotation is not None:
            layers.append(tf.keras.layers.RandomRotation(rotation, seed=seed))
            logger.info("Data Augmentation: RandomRotation adicionado")

        data_augmentation = tf.keras.Sequential(layers, name="augmentation")
        logger.info("🔒 Data augmentation configurado")

        return data_augmentation