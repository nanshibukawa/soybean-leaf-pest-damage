from typing import Any, Dict
from pathlib import Path
import numpy as np
import tensorflow as tf


from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = configure_logger(__name__)


class ModelTraining:
    """
    Componente para treinamento de modelos CNN.
    """

    def __init__(self, model: tf.keras.Model, model_config: ModelConfig):
        """
        Inicializa o componente de treinamento do modelo.

        Args:
            model: Modelo Keras pré-construído para treinar
            model_config: Configuração contendo parâmetros de treinamento (épocas, taxa de aprendizado, etc.)
        """
        self.model_config = model_config
        self.model = model
        self.history = None

    def train_model(
        self, train_data: tf.data.Dataset, validation_data: tf.data.Dataset
    ) -> tf.keras.callbacks.History:
        try:
            logger.info("Iniciando preparação para Fine-Tuning...")

            try:
                backbone = self.model.get_layer("core_backbone")
                logger.info(
                    "🧠 Backbone 'core_backbone' encontrado com sucesso para treinamento."
                )
            except ValueError:
                raise Exception(
                    "Backbone não encontrado no modelo! Certifique-se de usar a ModelFactory."
                )

            # 2. Configurar o congelamento seletivo
            backbone.trainable = True
            total_layers = len(backbone.layers)

            is_from_scratch = (
                not self.model_config.weights
                or str(self.model_config.weights).lower() == "none"
            )

            if is_from_scratch:
                logger.info(
                    f"🔓 Treinamento from scratch detectado (sem pesos pré-treinados). "
                    f"Mantendo todas as {total_layers} camadas aprendendo (trainable=True)."
                )
            else:
                layers_to_unfreeze = min(
                    self.model_config.unfreeze_last_n_layers, total_layers
                )
                # Congela as iniciais, libera as últimas
                for layer in backbone.layers[: total_layers - layers_to_unfreeze]:
                    layer.trainable = False
                for layer in backbone.layers[total_layers - layers_to_unfreeze :]:
                    if not isinstance(layer, tf.keras.layers.BatchNormalization):
                        layer.trainable = True
                    else:
                        layer.trainable = False
                logger.info(
                    f"🔓 Fine-tuning: {layers_to_unfreeze} camadas liberadas de {total_layers}."
                )

            self._compile_model()

            logger.info("Iniciando fit do modelo...")
            # class_weight = class_weights or self.model_config.class_weights
            # logger.info(f"🔍 class_weights injetados no fit: {class_weight}")

            self.history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=self.model_config.epochs,
                callbacks=self._callbacks(),
                # class_weight=class_weight,
                verbose=1,
            )
            return self.history

        except Exception as e:
            logger.error(f"❌ Erro durante treinamento: {e}")
            raise

    def _callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
                verbose=1,
                mode="min",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1,
            ),
        ]
        return callbacks

    def get_training_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do treinamento"""
        if self.history is None:
            return {"trained": False}

        history_dict = self.history.history

        return {
            "trained": True,
            "epochs_completed": len(history_dict.get("loss", [])),
            "final_train_loss": (
                history_dict.get("loss", [None])[-1] if "loss" in history_dict else None
            ),
            "final_val_loss": (
                history_dict.get("val_loss", [None])[-1]
                if "val_loss" in history_dict
                else None
            ),
            "final_train_accuracy": (
                history_dict.get("accuracy", [None])[-1]
                if "accuracy" in history_dict
                else None
            ),
            "final_val_accuracy": (
                history_dict.get("val_accuracy", [None])[-1]
                if "val_accuracy" in history_dict
                else None
            ),
        }

    def _compile_model(self):
        # Calcular os pesos alpha balanceados (Effective Number of Samples - Cui et al.)
        train_dir = Path("artifacts/data/final/DatasetPests-split/train")
        alpha_weights = 1.0  # Fallback padrão

        if train_dir.exists():
            class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            train_counts = {
                d.name: len([f for f in d.iterdir() if f.is_file()])
                for d in train_dir.iterdir() if d.is_dir()
            }
            
            if self.model_config.class_weights:
                raw_weights = [self.model_config.class_weights.get(i, 1.0) for i in range(len(class_names))]
                raw_weights = np.array(raw_weights, dtype=np.float32)
                alpha_weights = (raw_weights / np.mean(raw_weights)).tolist()
                logger.info(f"⚖️ Usando pesos alpha manuais do YAML (normalizados): {dict(zip(class_names, alpha_weights))}")
            else:
                beta = 0.999
                weights = []
                for class_name in class_names:
                    n = train_counts.get(class_name, 1)
                    w = (1.0 - beta) / (1.0 - np.power(beta, n)) if n > 0 else 1.0
                    weights.append(w)
                
                # Normalizar para que a média dos pesos seja 1.0 (mantém a escala original da loss)
                weights = np.array(weights, dtype=np.float32)
                alpha_weights = (weights / np.mean(weights)).tolist()
                logger.info(f"⚖️ Pesos alpha calculados (Focal Loss): {dict(zip(class_names, alpha_weights))}")
        else:
            logger.warning(f"⚠️ Diretório de treino {train_dir} não encontrado. Usando alpha=1.0")

        loss_function = tf.keras.losses.CategoricalFocalCrossentropy(
            gamma=1.5, alpha=alpha_weights
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.SGD(
                learning_rate=self.model_config.learning_rate,
            ),
            # loss=self.model_config.loss_function,
            loss=loss_function,
            metrics=self.model_config.metrics,
        )
        logger.info(f"📉 Loss function: {loss_function}")

        logger.info("Modelo SGD recompilado com configurações de treinamento")
