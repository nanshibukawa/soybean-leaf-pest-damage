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

        if (
            hasattr(self.model_config, "gaussian_noise")
            and self.model_config.gaussian_noise
        ):
            noise_std = self.model_config.gaussian_noise
        else:
            noise_std = 0.01
        logger.info(f"Adicionando Gaussian Noise: {noise_std}")
        layers.append(tf.keras.layers.GaussianNoise(noise_std))
        # x = tf.keras.layers.GaussianNoise(noise_std)(x)

        if (
            hasattr(self.model_config, "horizontal_flip")
            and self.model_config.horizontal_flip
        ):
            layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
            logger.info("Data Augmentation: RandomFlip horizontal adicionado")

        # 2. FLIP VERTICAL (Altamente recomendado para o app no campo)
        # O produtor pode virar o celular de cabeça para baixo para fotografar a folha por baixo.
        layers.append(tf.keras.layers.RandomFlip("vertical", seed=seed))
        logger.info("Data Augmentation: RandomFlip vertical adicionado")

        # 3. RANDOM TRANSLATION (Crítico para mitigar o erro da Diabrotica na borda)
        # Desloca a imagem em até 15% para os lados e para cima/baixo
        layers.append(
            tf.keras.layers.RandomTranslation(
                height_factor=0.15,
                width_factor=0.15,
                fill_mode="reflect",
                seed=seed,
            )
        )
        logger.info(
            "Data Augmentation: RandomTranslation [0.15] adicionado para invariância de borda"
        )

        if hasattr(self.model_config, "zoom_factor"):
            layers.append(
                tf.keras.layers.RandomZoom(self.model_config.zoom_factor, seed=seed)
            )
            logger.info(
                f"Data Augmentation: RandomZoom adicionado{self.model_config.zoom_factor}"
            )

        # NOTE: brightness_range está piorando os resultados para este dataset (soybean leaf pest damage)
        # if hasattr(self.model_config, "brightness_range"):
        #     brightness = self.model_config.brightness_range
        #     layers.append(
        #         tf.keras.layers.RandomBrightness(factor=brightness, seed=seed)
        #     )
        #     logger.info("Data Augmentation: RandomBrightness adicionado")

        # TODO: implement RandomTranslation
        # """
        # Por que isso resolve o seu problema da Diabrotica? Na sua imagem real, o modelo deu "Healthy" porque o furo da Diabrotica estava colado na margem de cima. O RandomTranslation vai empurrar as imagens artificialmente para os lados, para cima e para baixo durante o treino. Isso força os filtros convolucionais a rastrearem furos periféricos.
        # O que testar no código: Ativar o RandomTranslation preenchendo os espaços vazios com o modo "reflect" (para simular a continuação da folha/fundo).
        # """

        if (
            hasattr(self.model_config, "contrast_range")
            and self.model_config.contrast_range
        ):
            contrast_range = self.model_config.contrast_range
            logger.info("Data Augmentation: RandomContrast (config) adicionado")
        else:
            # contrast_range = [0.1, 1.1]
            contrast_range = [0.8, 1.2]
            logger.info(
                f"Data Augmentation: RandomContrast default adicionado: {contrast_range}"
            )
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

        # # TODO add RandomCop and RandomTranlation to the complete pipeline

        # factor_erasing = getattr(self.model_config, "random_erasing_factor", 0.3)
        # max_scale = getattr(self.model_config, "random_erasing_max_scale", 0.12)

        # layers.append(
        #     tf.keras.layers.RandomErasing(
        #         factor=factor_erasing,  # 30% de chance de aplicar em cada imagem do lote
        #         scale=(
        #             0.02,
        #             max_scale,
        #         ),  # O tamanho do retângulo ocupará entre 2% e 12% da área da folha
        #         fill_value=0.0,  # Preenchimento preto constante para simular sombra perfeitamente
        #         value_range=(
        #             0,
        #             255,
        #         ),  # Executado no início do pipeline (antes do Rescaling)
        #         # seed=42
        #         seed=seed,
        #     )
        # )
        # logger.info(
        #     f"Data Augmentation: RandomErasing configurado com sucesso ({factor_erasing} prob)"
        # )
        data_augmentation = tf.keras.Sequential(layers, name="augmentation")
        logger.info("🔒 Data augmentation configurado")

        return data_augmentation
