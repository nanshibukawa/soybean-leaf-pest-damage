from typing import Optional
import tensorflow as tf

# from cnnClassifier.models.mobilevit import create_mobilevit
# from cnnClassifier.models.vit_small_ds import create_vit_classifier
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ModelConfig, ImageConfig
from cnnClassifier.models.custom_blocks import compression_block, se_block
from cnnClassifier.models.factory import ModelFactory
from cnnClassifier.components.augmentation_pipeline import AugmentationPipeline

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
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

        data_augmentation_pipeline = AugmentationPipeline(self.model_config)
        self.data_augmentation = data_augmentation_pipeline.data_augmentation()

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
        modelo_base, preprocess_layer = ModelFactory.get_pretrained_model(
            model_name=self.model_config.model_name,
            input_shape=self.image_config.size_tuple,
            include_top=False,
            weights=self.model_config.weights,
        )

        l2_val = getattr(self.model_config, "l2_regularization", 0.01)
        logger.info(f"L2 regularization configurada: {l2_val}")

        inputs = tf.keras.layers.Input(shape=self.image_config.size_tuple)

        x = inputs

        x = self.data_augmentation(x)

        # Aplicar a camada de pré-processamento nativa do Keras devolvida pela Factory
        if preprocess_layer is not None:
            x = preprocess_layer(x)

        # IMPORTANTE: Chamar modelo base com training=False para manter BatchNormalization em modo de inferência
        # Isto é crítico durante fine-tuning: impede que as estatísticas armazenadas (mean/variance)
        # sejam sobrescritas pelas estatísticas do lote atual, destruindo o conhecimento pré-treinado
        # Referência: https://keras.io/guides/transfer_learning/

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
            x = compression_block(32, l2_reg=l2_val)(x)
            if use_se_block:
                x = se_block(x)
            x = compression_block(64, l2_reg=l2_val)(x)
        elif use_se_block:
            x = se_block(x)
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
            name="classification_head",
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
