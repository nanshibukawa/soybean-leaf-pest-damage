from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np
import tensorflow as tf


from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.utils.model_utils import configure_backbone_trainability, create_sgd_optimizer


physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

logger = configure_logger(__name__)


class ModelTraining:
    """
    Componente para treinamento de modelos CNN.
    """

    def __init__(self, model: tf.keras.Model, model_config: ModelConfig, train_dir: Optional[Path] = None):
        """
        Inicializa o componente de treinamento do modelo.

        Args:
            model: Modelo Keras pré-construído para treinar
            model_config: Configuração contendo parâmetros de treinamento (épocas, taxa de aprendizado, etc.)
            train_dir: Diretório de treino opcional para cálculo de pesos da focal loss.
        """
        self.model_config = model_config
        self.model = model
        self.history = None
        self.train_dir = train_dir

    def train_model(
        self, train_data: tf.data.Dataset, validation_data: tf.data.Dataset
    ) -> tf.keras.callbacks.History:
        try:
            logger.info("Iniciando preparação para Fine-Tuning...")

            is_from_scratch = (
                not self.model_config.weights
                or str(self.model_config.weights).lower() == "none"
            )

            if is_from_scratch:
                try:
                    backbone = self.model.get_layer("core_backbone")
                    backbone.trainable = True
                    logger.info(
                        f"🔓 Treinamento from scratch detectado (sem pesos pré-treinados). "
                        f"Mantendo todas as {len(backbone.layers)} camadas aprendendo (trainable=True)."
                    )
                except ValueError:
                    logger.warning("⚠️ Backbone 'core_backbone' não encontrado no modelo customizado.")
            else:
                configure_backbone_trainability(
                    self.model,
                    unfreeze_last_n=self.model_config.unfreeze_last_n_layers,
                    backbone_name="core_backbone"
                )

            self._compile_model(train_data)

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
        ]
        
        # Só adiciona ReduceLROnPlateau se o learning rate não for um schedule (ex: CosineDecay)
        is_schedule = False
        try:
            # 1. Verificar se a configuração do otimizador indica um schedule
            opt_config = self.model.optimizer.get_config()
            if isinstance(opt_config.get("learning_rate"), dict):
                is_schedule = True
        except Exception:
            pass

        if not is_schedule:
            # 2. Verificar se o atributo _learning_rate existe e é callable (como CosineDecay)
            lr_obj = getattr(self.model.optimizer, "_learning_rate", None)
            if lr_obj is not None:
                if callable(lr_obj) or "LearningRateSchedule" in type(lr_obj).__name__ or isinstance(lr_obj, tf.keras.optimizers.schedules.LearningRateSchedule):
                    is_schedule = True

        if not is_schedule:
            # 3. Fallback para verificar o learning_rate direto
            lr_obj = getattr(self.model.optimizer, "learning_rate", None)
            if lr_obj is not None:
                if callable(lr_obj) or "LearningRateSchedule" in type(lr_obj).__name__ or isinstance(lr_obj, tf.keras.optimizers.schedules.LearningRateSchedule):
                    is_schedule = True

        if not is_schedule:
            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1,
                )
            )
        else:
            logger.info("ℹ️ Learning rate schedule detectado no otimizador. ReduceLROnPlateau desabilitado para evitar conflitos de escrita.")
            
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

    def _compile_model(self, train_data: tf.data.Dataset):
        # Calcular passos dinâmicos por época
        steps_per_epoch = tf.data.experimental.cardinality(train_data).numpy()
        if steps_per_epoch < 0:
            steps_per_epoch = 1  # Fallback seguro caso indefinido

        # Configurar Otimizador Dinâmico
        opt_name = self.model_config.optimizer_name.lower()
        opt_params = self.model_config.optimizer_params or {}

        if opt_name == "sgd":
            initial_lr = opt_params.get("initial_lr", 1e-5)
            peak_lr = self.model_config.learning_rate
            momentum = opt_params.get("momentum", 0.9)
            nesterov = opt_params.get("nesterov", True)
            alpha = opt_params.get("alpha", 0.001)

            optimizer = create_sgd_optimizer(
                initial_lr=initial_lr,
                peak_lr=peak_lr,
                epochs=self.model_config.epochs,
                steps_per_epoch=steps_per_epoch,
                momentum=momentum,
                nesterov=nesterov,
                alpha=alpha,
            )
            logger.info(f"🏎️ Otimizador SGD com CosineDecay configurado (peak_lr={peak_lr})")
        elif opt_name == "adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_config.learning_rate,
                beta_1=opt_params.get("beta_1", 0.9),
                beta_2=opt_params.get("beta_2", 0.999),
                epsilon=opt_params.get("epsilon", 1e-07),
            )
            logger.info(f"🏎️ Otimizador Adam configurado (lr={self.model_config.learning_rate})")
        else:
            optimizer = tf.keras.optimizers.get(opt_name)
            if hasattr(optimizer, "learning_rate"):
                optimizer.learning_rate = self.model_config.learning_rate
            logger.info(f"🏎️ Otimizador genérico '{opt_name}' configurado.")

        # Calcular os pesos alpha balanceados (Effective Number of Samples - Cui et al.)
        train_dir = self.train_dir or Path("artifacts/data/final/DatasetPests-split/train")
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
            optimizer=optimizer,
            # loss=self.model_config.loss_function,
            loss=loss_function,
            metrics=self.model_config.metrics,
        )
        logger.info(f"📉 Loss function: {loss_function}")

        logger.info("Modelo compilado com configurações de treinamento.")
