import keras_hub
from cnnClassifier.models.mobilevit import create_mobilevit
from cnnClassifier.models.mobilevit_v2 import create_mobilevit_custom

import tensorflow as tf
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)


class ModelFactory:
    """
    Fábrica responsável por fornecer o backbone instanciado e as lógicas
    de pré-processamento exclusivas de cada arquitetura.
    """

    @staticmethod
    def get_pretrained_model(
        model_name: str,
        input_shape: tuple,
        include_top: bool = False,
        weights: str = "imagenet",
    ):
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

        model_name_lower = model_name.lower()
        model_class = models.get(model_name_lower)

        if model_class is None:
            supported_models = ", ".join(sorted(models.keys()))
            raise ValueError(
                f"❌ Modelo '{model_name}' não é suportado!"
                f"   Modelos disponíveis: {supported_models}"
            )
        comuns_params = {
            "input_shape": input_shape,
            "include_top": include_top,
            "weights": weights,
        }

        if "nasnet" in model_name_lower or "efficientnetb" in model_name_lower:
            comuns_params["input_tensor"] = None
            comuns_params["pooling"] = None

        modelo_base_temp = model_class(**comuns_params)
        logger.info(
            f"✅ Modelo base {model_class.__name__} carregado com sucesso pela Factory."
        )

        try:
            # Ao reconstruir um container Model, forçamos que a camada no grafo final
            # receba obrigatoriamente e garantidamente a string name="core_backbone"
            modelo_base = tf.keras.Model(
                inputs=modelo_base_temp.input,
                outputs=modelo_base_temp.output,
                name="core_backbone",
            )
        except Exception as e:
            logger.warning(
                f"⚠️ Renomeação estrutural falhou, usando fallback _name. Erro: {e}"
            )
            modelo_base = modelo_base_temp
            modelo_base._name = "core_backbone"

        # Tabela de funções de preprocessamento originais Keras
        preprocess_input_dict = {
            "mobilenetv3large": None,
            "mobilenetv3small": None,
            "inceptionv3": tf.keras.applications.inception_v3.preprocess_input,
            "vgg19": tf.keras.applications.vgg19.preprocess_input,
            "nasnetlarge": tf.keras.applications.nasnet.preprocess_input,
            "nasnetmobile": tf.keras.applications.nasnet.preprocess_input,
            "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetb1": tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetb2": tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetb3": tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetb7": tf.keras.applications.efficientnet.preprocess_input,
            "efficientnetv2b0": None,
            "efficientnetv2b1": None,
            "efficientnetv2b2": None,
            "efficientnetv2b3": None,
            "convnexttiny": tf.keras.applications.convnext.preprocess_input,
            "convnextsmall": tf.keras.applications.convnext.preprocess_input,
        }

        preprocess_func = preprocess_input_dict.get(model_name_lower, None)

        if preprocess_func is not None:
            logger.info(
                f"✅ Preprocessing nativo importado na Factory para {model_name_lower}"
            )
        else:
            logger.info(
                f"✅ Modelo {model_name_lower} possui built-in preprocessing ou não requer."
            )

        return modelo_base, preprocess_func

    @staticmethod
    def get_vit_keras_hub(model_name: str, input_shape: tuple):

        # O Keras Hub bate na porta da Hugging Face e baixa a arquitetura + pesos
        backbone = keras_hub.models.ViTBackbone.from_preset(model_name)

        inputs = tf.keras.Input(shape=input_shape)
        outputs = backbone(inputs)

        modelo_base = tf.keras.Model(
            inputs=inputs, outputs=outputs, name="core_backbone"
        )

        # Retornamos (modelo, None) para manter a compatibilidade com a
        # descompactação (unpacking) que ocorre lá no PrepareModel
        return modelo_base, None

    @staticmethod
    def get_mobilevit(model_name: str, input_shape: tuple):
        if "custom" in model_name.lower() or "v2" in model_name.lower() or "se" in model_name.lower():
            modelo_base = create_mobilevit_custom(input_shape=input_shape)
            logger.info(f"✅ MobileViT Custom ({model_name}) criado com sucesso pela Factory.")
        else:
            modelo_base = create_mobilevit(input_shape=input_shape)
            logger.info("✅ MobileViT (Padrão) criado com sucesso pela Factory.")
        return modelo_base, None
