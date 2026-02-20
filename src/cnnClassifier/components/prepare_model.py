from typing import Optional
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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

        # x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
        # x = tf.keras.layers.Rescaling(scale=2.0 / 255, offset=-1.0)(inputs)

        x = inputs

        # use_data_augmentation = getattr(
        #     self.model_config, "use_data_augmentation", False
        # )
        # logger.info(f"🔒 Using Data Augmentation: {use_data_augmentation}")

        # if use_data_augmentation:
        # Gaussian Noise
        # if (
        #     hasattr(self.model_config, "gaussian_noise")
        #     and self.model_config.gaussian_noise
        # ):
        #     noise_std = self.model_config.gaussian_noise
        # else:
        #     noise_std = 0.05
        # logger.info(f"Adicionando Gaussian Noise: {noise_std}")

        # x = tf.keras.layers.GaussianNoise(noise_std)(x)

        x = self.data_augmentation(x)

        # Normalização condicional baseada no modelo
        model_name = self.model_config.model_name.lower()
        
        # Modelos que esperam entrada [0, 1]: VGG, EfficientNet
        if "vgg" in model_name or "efficientnet" in model_name:
            x = tf.keras.layers.Rescaling(1.0 / 255)(x)
            logger.info(
                f"✅ Normalização [0,1]: Rescaling(1/255) para {self.model_config.model_name}"
            )
        # Modelos que esperam entrada [-1, 1]: MobileNet, Inception, NASNet
        elif any(keyword in model_name for keyword in ["mobilenet", "inception", "nasnet"]):
            x = tf.keras.layers.Rescaling(scale=2.0 / 255, offset=-1.0)(x)
            logger.info(
                f"✅ Normalização [-1,1]: Rescaling(2/255, -1) para {self.model_config.model_name}"
            )
        else:
            logger.warning(
                f"⚠️ Modelo '{self.model_config.model_name}' sem normalização específica - pulando"
            )

        # Modelo pré-treinado
        x = modelo_base(x, training=False)

        use_compression_blocks = getattr(
            self.model_config, "use_compression_blocks", False
        )
        use_se_block = getattr(self.model_config, "use_se_block", False)

        logger.info(f"🔒 Using Compression Blocks: {use_compression_blocks}")
        logger.info(f"🔒 Using Squeeze-and-Excitation (SE) Block: {use_se_block}")

        # Blocos compressions e SE após o backbone (condicionais)
        if use_compression_blocks:
            x = self._compression_block(32)(x)
            if use_se_block:
                x = self.se_block(x)
            x = self._compression_block(64)(x)
        elif use_se_block:
            x = self.se_block(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.model_config.dropout_rate)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_constraint=tf.keras.constraints.MaxNorm(3),
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.model_config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(
            self.model_config.num_classes,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.L2(0.01),
        )(x)

        modelo = tf.keras.Model(inputs=inputs, outputs=outputs)
        logger.info(f"Dropout rate aplicado: {self.model_config.dropout_rate}")
        logger.info(
            f"Utilizando modelo pré-treinado {self.model_config.model_name} + blocos compressions"
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
        logger.info("Construindo modelo customizado CNN")

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
                # DENSE COM L2 REGULARIZATION
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.005),
                ),
                tf.keras.layers.Dropout(self.model_config.dropout_rate),
                # SAÍDA COM L2 REGULARIZATION
                tf.keras.layers.Dense(
                    self.model_config.num_classes,
                    activation=activation_last_layer,
                    kernel_regularizer=tf.keras.regularizers.L2(0.001),
                ),
            ]
        )

        logger.info(f"Modelo {self.model_config.model_name} construído com sucesso")
        return modelo

    def _data_augmentation(self):
        """
        Cria pipeline de data augmentation baseado na configuração.

        Returns:
            tf.keras.Sequential: Pipeline de augmentação
        """
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
            layers.append(tf.keras.layers.RandomFlip("horizontal"))
            logger.info("Data Augmentation: RandomFlip horizontal adicionado")

        if hasattr(self.model_config, "zoom_factor"):
            layers.append(tf.keras.layers.RandomZoom(self.model_config.zoom_factor))
            logger.info("Data Augmentation: RandomZoom adicionado")

        # NOTE: brightness_range está piorando os resultados para este dataset (soybean leaf pest damage)
        # if hasattr(self.model_config, "brightness_range"):
        #     brightness = self.model_config.brightness_range
        #     layers.append(tf.keras.layers.RandomBrightness(factor=brightness))
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
        layers.append(tf.keras.layers.RandomContrast(factor=contrast_range))

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
            layers.append(tf.keras.layers.RandomRotation(rotation))
            logger.info("Data Augmentation: RandomRotation adicionado")

        data_augmentation = tf.keras.Sequential(layers)
        logger.info("🔒 Data augmentation configurado")

        return data_augmentation

    def _pretrained_model(self, include_top: bool = False, weights: str = "imagenet"):
        """
        Cria modelo pré-treinado baseado no modelo.

        Args:
            include_top: Se true, inclui a camada fully-connected no topo do modelo.
            weights: Pesos a serem carregados (ex: "imagenet")

        Returns:
            tf.keras.Model: Modelo base pré-treinado
        """
        model_name = self.model_config.model_name.lower()

        models = {
            "mobilenetv3large": tf.keras.applications.MobileNetV3Large,
            "mobilenetv3small": tf.keras.applications.MobileNetV3Small,
            "inceptionv3": tf.keras.applications.InceptionV3,
            "vgg19": tf.keras.applications.VGG19,
            "nasnetlarge": tf.keras.applications.NASNetLarge,
            "nasnetmobile": tf.keras.applications.NASNetMobile,
            "efficientnetb0": tf.keras.applications.EfficientNetB0,
            "efficientnetb7": tf.keras.applications.EfficientNetB7,
        }
        model_class = models.get(model_name)

        # Fallback: busca por substring (ex: "mobilenet" -> "mobilenetv3large")
        if model_class is None:
            for key, cls in models.items():
                if key in model_name or model_name in key:
                    model_class = cls
                    logger.info(f"🔍 Modelo encontrado por substring: {key}")
                    break

        # Default: MobileNetV3Large
        if model_class is None:
            model_class = tf.keras.applications.MobileNetV3Large
            logger.warning(
                f"⚠️ Modelo '{model_name}' não reconhecido. Usando MobileNetV3Large."
            )

        # Parâmetros comuns
        comuns_params = {
            "input_shape": self.image_config.size_tuple,
            "include_top": include_top,
            "weights": weights,
        }

        # Parâmetros extras para modelos específicos
        if model_name in [
            "nasnetlarge",
            "nasnetmobile",
            "efficientnetb0",
            "efficientnetb7",
        ]:
            comuns_params["input_tensor"] = None
            comuns_params["pooling"] = None

        # Instancia APENAS o modelo necessário
        modelo_base = model_class(**comuns_params)
        logger.info(f"✅ Modelo base {model_class.__name__} carregado")

        return modelo_base
        # if "mobilenet" in model_name:
        #     modelo_base = tf.keras.applications.MobileNetV3Large(
        #         # modelo_base = tf.keras.applications.MobileNetV3Small(
        #         input_shape=self.image_config.size_tuple,
        #         include_top=include_top,
        #         weights=weights,
        #         # include_preprocessing=False,
        #     )
        #     logger.info("🧠 Using model: MobileNetV3Large")
        # elif "inception" in model_name:
        #     modelo_base = tf.keras.applications.InceptionV3(
        #         input_shape=self.image_config.size_tuple,
        #         include_top=include_top,
        #         weights=weights,
        #     )
        # elif "vgg" in model_name:
        #     modelo_base = tf.keras.applications.VGG19(
        #         input_shape=self.image_config.size_tuple,
        #         include_top=include_top,
        #         weights=weights,
        #     )
        # else:
        #     # Default para MobileNet
        #     modelo_base = tf.keras.applications.MobileNetV3Large(
        #         input_shape=self.image_config.size_tuple,
        #         include_top=include_top,
        #         weights=weights,
        #         # include_preprocessing=False,
        #     )

        # logger.info(f"Modelo base {model_name} carregado")
        # return modelo_base

    def _compression_block(self, filters, kernel_size=3, strides=1):
        return tf.keras.Sequential(
            [
                tf.keras.layers.DepthwiseConv2D(
                    kernel_size, strides=strides, padding="same"
                ),
                tf.keras.layers.Conv2D(
                    filters,
                    1,
                    padding="same",
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.L2(0.01),
                ),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
        )

    def se_block(self, input_tensor, ratio=8):
        """
        Implementa bloco Squeeze-and-Excitation (SE).

        Este bloco realiza recalibração de canais adaptativa através de dois passos:
        1. Squeeze: Comprime informações espaciais usando GlobalAveragePooling2D
        2. Excitation: Modela relacionamentos inter-canais com duas camadas Dense

        O resultado é multiplicado element-wise com o tensor de entrada original.

        Args:
            input_tensor (tf.Tensor): Tensor de entrada com shape (batch, height, width, channels)
            ratio (int): Fator de redução para o gargalo (bottleneck) no bloco SE.
                        Default: 8. O número de neurônios na primeira Dense será
                        filters // ratio. Deve ser >= 1.

        Returns:
            tf.Tensor: Tensor recalibrado com mesmo shape que input_tensor

        Raises:
            ValueError: Se ratio <= 0 ou se filters // ratio < 1

        Nota:
            A validação garante que filters // ratio >= 1 para evitar dimensões inválidas.
        """
        filters = input_tensor.shape[-1]

        # Validação: garantir que ratio é válido
        if ratio <= 0:
            raise ValueError(f"ratio deve ser positivo, recebido: {ratio}")

        reduced_filters = filters // ratio
        if reduced_filters < 1:
            raise ValueError(
                f"Número de filtros ({filters}) muito pequeno para ratio ({ratio}). "
                f"filters // ratio deve ser >= 1, obteve {reduced_filters}. "
                f"Considere usar ratio <= {filters}."
            )

        # Squeeze: comprime informações espaciais
        se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)

        # Excitation: modela dependências inter-canais
        se = tf.keras.layers.Dense(reduced_filters, activation="relu")(se)
        se = tf.keras.layers.Dense(filters, activation="sigmoid")(se)

        # Reshape para compatibilidade com multiplicação
        se = tf.keras.layers.Reshape((1, 1, filters))(se)

        # Recalibração: multiplicação element-wise
        return tf.keras.layers.multiply([input_tensor, se])

    # TODO: Implementar, se necessário para implantação em smartphones:
    # - Pruning do modelo utilizando TensorFlow Model Optimization (TFMOT) para
    #   reduzir o tamanho e o custo de inferência.
    # - Função de distillation loss baseada em um modelo professor (teacher) para
    #   treinar um modelo aluno (student) mais compacto.
    # - Geração de mapas de calor Grad-CAM para explicar visualmente as decisões
    #   do modelo em imagens de entrada.
    #
    # As implementações devem ser adicionadas aqui como métodos da classe
    # `PrepareModel` quando os requisitos de implantação exigirem esses recursos.
