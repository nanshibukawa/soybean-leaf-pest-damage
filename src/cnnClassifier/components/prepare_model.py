from typing import Optional
import tensorflow as tf


# from cnnClassifier.models.mobilevit import create_mobilevit
# from cnnClassifier.models.vit_small_ds import create_vit_classifier
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig


physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
        """Constrói modelo - custom, pré-treinado ou transformer baseado na configuração"""
        logger.info(f"🏗️ Construindo modelo {self.model_config.model_name}")

        model_name = self.model_config.model_name.lower()
        # Lógica para selecionar arquitetura
        if any(t in model_name for t in ["vit", "mobilevit"]):
            return self._build_transformer_model(model_name)
        use_pretrained = getattr(self.model_config, "use_pretrained", False)
        if use_pretrained:
            return self._build_pretrained_model()
        else:
            return self._build_custom_model()

    def _build_transformer_model(self) -> tf.keras.Model:
        # TODO: Implementar construção de modelos transformer (ViT, MobileViT, ViTImageClassifier)
        """
        Constrói e retorna um modelo Vision Transformer (ViT), MobileViT ou ViTImageClassifier.
        Adapta o pipeline para transformers, incluindo data augmentation, pré-processamento e pooling adequados.
        """

        model_name = self.model_config.model_name.lower()
        logger.info(f"🔬 Construindo modelo Transformer: {model_name}")

        # Dicionário de modelos: cada função retorna um modelo Keras funcional
        vit_models = {
            # "mobilevit": create_mobilevit,
            # "vit_small_ds": create_vit_classifier,
        }
        if model_name not in vit_models:
            supported_models = ", ".join(sorted(vit_models.keys()))
            raise ValueError(
                f"❌ Modelo '{model_name}' não é suportado para transformer! "
                f"Modelos disponíveis: {supported_models}"
            )

        # Input e data augmentation
        inputs = tf.keras.layers.Input(shape=self.image_config.size_tuple)
        x = self.data_augmentation(inputs)

        # Se o modelo NÃO faz rescaling internamente, aplica aqui
        # Exemplo: MobileViT precisa de Rescaling externo, ViT já faz internamente
        if model_name == "mobilevit":
            x = tf.keras.layers.Rescaling(1.0 / 255)(x)
            logger.info("Rescaling externo aplicado para MobileViT")
        else:
            logger.info(
                f"Modelo {model_name} faz preprocessing interno ou não requer rescaling externo"
            )

        # Chama a função do modelo, conectando o pipeline
        if model_name == "mobilevit":
            model_body = vit_models[model_name](
                num_classes=self.model_config.num_classes
            )
        elif model_name == "vit_small_ds":
            model_body = vit_models[model_name](vanilla=False)
        else:
            raise ValueError(f"Modelo {model_name} não suportado!")

        outputs = model_body(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        logger.info(f"✅ Modelo transformer {model_name} construído com sucesso!")
        return model

    def _build_pretrained_model(self) -> tf.keras.Model:
        """
        Constrói e retorna um modelo Keras pré-treinado para classificação de imagens.

        Returns:
            tf.keras.Model: Modelo compilado pronto para treinamento
        """

        l2_val = getattr(self.model_config, "l2_regularization", 0.01)
        logger.info(f"L2 regularization configurada: {l2_val}")

        inputs = tf.keras.layers.Input(shape=self.image_config.size_tuple)

        x = inputs

        x = self.data_augmentation(x)

        # Normalização condicional baseada no modelo
        model_name = self.model_config.model_name.lower()

        # VGG: Preprocessing específico (RGB→BGR + zero-centering, SEM rescaling)
        # Ref: https://keras.io/api/applications/vgg/vgg_preprocessing/

        preprocess_input = {
            "mobilenetv3large": None,
            "mobilenetv3small": None,
            "inceptionv3": tf.keras.applications.inception_v3.preprocess_input(x),
            "vgg19": tf.keras.applications.vgg19.preprocess_input(x),
            "nasnetlarge": tf.keras.applications.nasnet.preprocess_input(x),
            "nasnetmobile": tf.keras.applications.nasnet.preprocess_input(x),
            "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input(x),
            "efficientnetb1": tf.keras.applications.efficientnet.preprocess_input(x),
            "efficientnetb2": tf.keras.applications.efficientnet.preprocess_input(x),
            "efficientnetb3": tf.keras.applications.efficientnet.preprocess_input(x),
            "efficientnetb7": tf.keras.applications.efficientnet.preprocess_input(x),
            "efficientnetv2b0": None,
            "efficientnetv2b1": None,
            "efficientnetv2b2": None,
            "efficientnetv2b3": None,
            "convnexttiny": tf.keras.applications.convnext.preprocess_input(x),
            "convnextsmall": tf.keras.applications.convnext.preprocess_input(x),
        }

        if model_name not in preprocess_input.keys():
            raise ValueError(
                f"❌ Modelo '{self.model_config.model_name}' não reconhecido para normalização! "
                f"Verifique a configuração ou adicione lógica de normalização para este modelo."
            )

        preprocess_model = preprocess_input.get(model_name)

        if preprocess_model is not None:
            x = preprocess_model
            logger.info(
                f"✅ Preprocessing específico aplicado para {self.model_config.model_name}"
            )

        else:
            logger.info(
                f"✅ Modelo {self.model_config.model_name} tem preprocessing interno ativo - pulando normalização externa"
            )

        # IMPORTANTE: Chamar modelo base com training=False para manter BatchNormalization em modo de inferência
        # Isto é crítico durante fine-tuning: impede que as estatísticas armazenadas (mean/variance)
        # sejam sobrescritas pelas estatísticas do lote atual, destruindo o conhecimento pré-treinado
        # Referência: https://keras.io/guides/transfer_learning/

        modelo_base = self._pretrained_model(
            include_top=False, weights=self.model_config.weights
        )

        modelo_base.trainable = False
        logger.info(f"🔒 Modelo base congelado: {len(modelo_base.layers)} camadas")

        x = modelo_base(x, training=False)

        use_compression_blocks = getattr(
            self.model_config, "use_compression_blocks", False
        )
        use_se_block = getattr(self.model_config, "use_se_block", False)

        logger.info(f"🔒 Using Compression Blocks: {use_compression_blocks}")
        logger.info(f"🔒 Using Squeeze-and-Excitation (SE) Block: {use_se_block}")

        # Blocos compressions e SE após o backbone (condicionais)
        if use_compression_blocks:
            x = self._compression_block(32, l2_config=l2_val)(x)
            if use_se_block:
                x = self.se_block(x)
            x = self._compression_block(64, l2_config=l2_val)(x)
        elif use_se_block:
            x = self.se_block(x)

        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(self.model_config.dropout_rate)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(
            128,
            activation="relu",
            kernel_constraint=tf.keras.constraints.MaxNorm(3),
            # kernel_regularizer=tf.keras.regularizers.L2(0.01),
            kernel_regularizer=tf.keras.regularizers.L2(l2_val),
        )(x)

        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(self.model_config.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(
            self.model_config.num_classes,
            activation="softmax",
            # kernel_regularizer=tf.keras.regularizers.L2(0.01),
            kernel_regularizer=tf.keras.regularizers.L2(l2_val),
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
        l2_val = getattr(self.model_config, "l2_regularization", 0.005)

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
                    kernel_regularizer=tf.keras.regularizers.L2(l2_val),
                ),
                tf.keras.layers.Dropout(self.model_config.dropout_rate),
                # SAÍDA COM L2 REGULARIZATION
                tf.keras.layers.Dense(
                    self.model_config.num_classes,
                    activation=activation_last_layer,
                    kernel_regularizer=tf.keras.regularizers.L2(l2_val),
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

        # Gaussian Noise
        if (
            hasattr(self.model_config, "gaussian_noise")
            and self.model_config.gaussian_noise
        ):
            # std de ruído configurável, default 0.05
            noise_std = self.model_config.gaussian_noise
            layers.append(tf.keras.layers.GaussianNoise(noise_std))
            logger.info(
                f"Data Augmentation: Gaussian Noise adicionado - noise: {noise_std}"
            )

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
            "efficientnetb1": tf.keras.applications.EfficientNetB1,
            "efficientnetb2": tf.keras.applications.EfficientNetB2,
            "efficientnetb3": tf.keras.applications.EfficientNetB3,
            "efficientnetb7": tf.keras.applications.EfficientNetB7,
            "efficientnetv2b0": tf.keras.applications.EfficientNetV2B0,
            "efficientnetv2b1": tf.keras.applications.EfficientNetV2B1,
            "efficientnetv2b2": tf.keras.applications.EfficientNetV2B2,
            "efficientnetv2b3": tf.keras.applications.EfficientNetV2B3,
            "convnexttiny": tf.keras.applications.ConvNeXtTiny,
            "convnextsmall": tf.keras.applications.ConvNeXtSmall,
        }
        model_class = models.get(model_name)

        # Erro se modelo não reconhecido
        if model_class is None:
            supported_models = ", ".join(sorted(models.keys()))
            raise ValueError(
                f"❌ Modelo '{model_name}' não é suportado!\n"
                f"   Modelos disponíveis: {supported_models}"
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
            "efficientnetb1",
            "efficientnetb2",
            "efficientnetb3",
            "efficientnetb7",
            "efficientnetv2b0",
            "efficientnetv2b1",
            "efficientnetv2b2",
            "efficientnetv2b3",
            "convnexttiny",
            "convnextsmall",
        ]:
            comuns_params["input_tensor"] = None
            comuns_params["pooling"] = None

        # EfficientNetV2: manter preprocessing interno explícito
        if model_name.startswith("efficientnetv2") or model_name.startswith("convnext"):
            comuns_params["include_preprocessing"] = True

        # Instancia APENAS o modelo necessário
        modelo_base = model_class(**comuns_params)
        logger.info(f"✅ Modelo base {model_class.__name__} carregado")

        return modelo_base

    def _compression_block(self, filters, kernel_size=3, strides=1, l2_config=None):
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
                    kernel_regularizer=tf.keras.regularizers.L2(l2_config),
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
