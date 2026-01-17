import keras_tuner as kt
import tensorflow as tf
import random
from pathlib import Path
from loguru import logger
import json
import numpy as np

from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.components.data_splitter import DataSplitter
from cnnClassifier.components.prepare_model import PrepareModel


class KerasTunerSearch:
    """Otimização de hiperparâmetros com Keras Tuner + Bayesian Optimization."""

    def __init__(self, model_config: ModelConfig, data_splitter: DataSplitter):
        self.model_config = model_config
        self.data_splitter = data_splitter
        self.best_hp = None
        self.best_model = None
        self.tuner = None

        # Seeds globais para reprodutibilidade
        # TODO VERIFY
        seed = getattr(model_config, "random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def build_model(self, hp):
        """
        Constrói modelo com hiperparâmetros sugeridos pelo tuner.

        Args:
            hp: Objeto HyperParameters do Keras Tuner

        Returns:
            Modelo Keras compilado
        """
        # Hiperparâmetros a otimizar
        hp_learning_rate = hp.Float(
            "learning_rate",
            min_value=1e-6,
            max_value=2e-4,
            sampling="log",
            default=1e-4,
        )
        dropout_rate = hp.Float(
            "dropout_rate", min_value=0.2, max_value=0.4, step=0.1, default=0.3
        )
        unfreeze_last_n = hp.Int(
            "unfreeze_last_n_layers",
            min_value=8,
            max_value=32,
            step=4,
            default=16,
        )

        # L2 Regularization para reduzir overfitting
        l2_reg = hp.Float(
            "l2_regularization",
            min_value=1e-6,
            max_value=1e-3,
            sampling="log",
            default=1e-5,
        )

        logger.info(
            f"Building model: lr={hp_learning_rate:.6f}, dropout={dropout_rate:.2f}, unfreeze={unfreeze_last_n}, l2={l2_reg:.6f}"
        )

        self.model_config.dropout_rate = float(dropout_rate)

        image_config = self.data_splitter.image_config
        prepare_model = PrepareModel(self.model_config, image_config)
        model = prepare_model.build_model()

        # Descongelar últimas camadas do BACKBONE
        if unfreeze_last_n > 0:
            backbone = None
            # Procurar pela camada MobileNetV3 dentro do modelo
            for layer in model.layers:
                if (
                    isinstance(layer, tf.keras.Model)
                    and "mobilenet" in layer.name.lower()
                ):
                    backbone = layer
                    logger.info(
                        f"Backbone encontrado: {layer.name} ({len(layer.layers)} camadas)"
                    )
                    break

            if backbone is not None:
                total_layers = len(backbone.layers)
                # congela iniciais, libera últimas unfreeze_last_n
                layers_to_unfreeze = min(unfreeze_last_n, total_layers)
                for layer in backbone.layers[: total_layers - layers_to_unfreeze]:
                    layer.trainable = False
                for layer in backbone.layers[total_layers - layers_to_unfreeze :]:
                    layer.trainable = True
                    if hasattr(layer, "kernel_regularizer"):
                        layer.kernel_regularizer = tf.keras.regularizers.l2(l2_reg)
                logger.info(
                    f"Backbone fine-tune: liberadas {layers_to_unfreeze} de {total_layers} camadas com L2={l2_reg:.6f}"
                )
            else:
                logger.warning("Backbone MobileNetV3 não identificado")
                # Fallback: unfreeze últimas camadas treináveis do modelo
                trainable_layers = [l for l in model.layers if l.trainable]
                for layer in trainable_layers[-unfreeze_last_n:]:
                    layer.trainable = True

        # Compilar usando mesmas definições do pipeline principal
        # TODO: implementar função com outros otimizadores, e chamar a função
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.loss_function,
            metrics=self.model_config.metrics,
        )

        return model

    def search(self, max_trials: int = 30, epochs_per_trial: int = 30):
        """
        Executa busca de hiperparâmetros com Bayesian Optimization.

        Args:
            max_trials: Número máximo de configurações a testar
            epochs_per_trial: Épocas de treino por candidato
        """
        logger.info("Iniciando busca com Bayesian Optimization...")
        logger.info(f"   Max trials: {max_trials}")
        logger.info(f"   Epochs per trial: {epochs_per_trial}")

        # Criar tuner
        self.tuner = kt.BayesianOptimization(
            hypermodel=self.build_model,
            objective="val_accuracy",
            max_trials=max_trials,
            num_initial_points=5,
            directory="artifacts/tuning/keras_tuner",
            project_name="mobilenetv3_bayesian",
            seed=42,
            overwrite=True,
        )

        # Carregar dados
        train_ds = self.data_splitter.load_train_data()
        val_ds = self.data_splitter.load_validation_data()

        logger.info(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

        # Rodar busca
        self.tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_per_trial,
            callbacks=self._callbacks(),
            verbose=1,
        )

        # Melhores hiperparâmetros
        self.best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]

        logger.info("Busca completa!")
        logger.info("Melhores Hiperparâmetros:")
        logger.info(f"   Learning Rate:       {self.best_hp.get('learning_rate'):.6f}")
        logger.info(f"   Dropout Rate:        {self.best_hp.get('dropout_rate'):.2f}")
        logger.info(
            f"   Unfreeze Last N:     {self.best_hp.get('unfreeze_last_n_layers')}"
        )
        if "l2_regularization" in self.best_hp.values:
            logger.info(
                f"   L2 Regularization:   {self.best_hp.get('l2_regularization'):.6f}"
            )
        logger.info("✅ Melhor modelo recuperado - pronto para retreino")

    def _callbacks(self):
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_accuracy",
                factor=0.2,
                patience=5,
                min_lr=1e-7,
                verbose=1,
                mode="max",
            ),
        ]
        return callbacks

    def retrain_best_model(self, epochs: int = 100):
        """
        Reconstrói e treina o modelo final com os melhores hiperparâmetros.

        Args:
            epochs: Número de épocas para treinar
        """
        if self.best_hp is None:
            raise ValueError("Execute .search() antes de .retrain_best_model()")

        logger.info(
            f"Retreinando modelo com melhores hiperparâmetros por {epochs} épocas..."
        )

        # Reconstruir modelo com melhores hiperparâmetros
        model = self.build_model(self.best_hp)

        # Descongelar últimas camadas do BACKBONE
        backbone = None
        for layer in model.layers:
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
                backbone = layer
                logger.info(
                    f"Backbone encontrado no retreino: {layer.name} ({len(layer.layers)} camadas)"
                )
                break

        if backbone is not None:
            total_layers = len(backbone.layers)
            layers_to_unfreeze = min(
                self.best_hp.values.get("unfreeze_last_n_layers", 20),
                total_layers,
            )
            for layer in backbone.layers[: total_layers - layers_to_unfreeze]:
                layer.trainable = False
            for layer in backbone.layers[total_layers - layers_to_unfreeze :]:
                layer.trainable = True
            logger.info(
                f"Fine-tuning (retreino): {layers_to_unfreeze} camadas liberadas de {total_layers}."
            )
        else:
            logger.warning("Backbone MobileNetV3 não encontrado no retreino")

        # Recompilar com LR fixo (pipeline)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.best_hp.get("learning_rate"),
                epsilon=1e-8,
                beta_1=0.9,
                beta_2=0.999,
                amsgrad=False,
            ),
            loss=self.model_config.loss_function,
            metrics=self.model_config.metrics,
        )

        # Carregar dados
        train_ds = self.data_splitter.load_train_data()
        val_ds = self.data_splitter.load_validation_data()

        # Callbacks para estabilizar treinamento
        callbacks = [
            # Early stopping monitorando val_loss (mais sensível ao overfitting)
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001,
            ),
            # ReduceLROnPlateau para suavizar oscilações
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1,
                mode="min",
            ),
        ]

        # Treinar
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.model_config.class_weights,
            verbose=1,
        )

        # Salvar modelo
        output_dir = Path("artifacts/models")
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = (
            output_dir / f"{self.model_config.model_name}_keras_tuner_best.keras"
        )
        model.save(model_path)

        logger.info(f"✅ Modelo final salvo em: {model_path}")

        self.best_model = model
        return model, history

    def save_best_hyperparameters(self, save_dir: str = "artifacts/tuning"):
        """Salva melhores hiperparâmetros em JSON."""
        if self.best_hp is None:
            raise ValueError("Execute .search() antes")

        save_path = Path(save_dir) / "best_hyperparameters.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best_params = {
            "learning_rate": float(self.best_hp.get("learning_rate")),
            "dropout_rate": float(self.best_hp.get("dropout_rate")),
            "unfreeze_last_n_layers": int(self.best_hp.get("unfreeze_last_n_layers")),
        }

        # Adicionar L2 se existir (compatibilidade com buscas antigas)
        if "l2_regularization" in self.best_hp.values:
            best_params["l2_regularization"] = float(
                self.best_hp.get("l2_regularization")
            )

        with open(save_path, "w") as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"✅ Hiperparâmetros salvos em: {save_path}")
