import json
import random
from pathlib import Path

import keras_tuner as kt
import numpy as np


import tensorflow as tf
from loguru import logger

from cnnClassifier.components.data_splitter import DataSplitter
from cnnClassifier.components.prepare_model import PrepareModel
from cnnClassifier.entity.config_entity import ImageConfig, ModelConfig
from cnnClassifier.tuning.mlflow_tuning import MLflowHyperModel
from tensorflow.keras.utils import plot_model


class KerasTunerSearch:
    """Otimização de hiperparâmetros com Keras Tuner + Bayesian Optimization."""

    def __init__(
        self,
        model_config: ModelConfig,
        data_splitter: DataSplitter,
        image_config: ImageConfig,
    ):
        self.model_config = model_config
        self.data_splitter = data_splitter
        self.image_config = image_config
        self.best_hp = None
        self.best_model = None
        self.tuner = None

        # Seeds globais para reprodutibilidade
        # TODO VERIFY
        seed = getattr(model_config, "random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _configure_backbone_trainability(
        self, model: tf.keras.Model, unfreeze_last_n: int
    ):
        """Helper unificado que configura o congelamento/descongelamento das camadas do backbone."""
        try:
            backbone = model.get_layer("core_backbone")
            logger.info(
                f"🧠 Backbone 'core_backbone' encontrado ({len(backbone.layers)} camadas)"
            )
        except ValueError:
            backbone = None
            logger.warning("⚠️  Backbone 'core_backbone' não encontrado")

        if backbone is not None:
            total_layers = len(backbone.layers)
            is_from_scratch = (
                not self.model_config.weights
                or str(self.model_config.weights).lower() == "none"
            )

            if is_from_scratch:
                logger.info(
                    f"🔓 Modelo from scratch detectado. Mantendo todas as {total_layers} camadas aprendendo (trainable=True)."
                )
                backbone.trainable = True
            else:
                backbone.trainable = True
                layers_to_unfreeze = min(unfreeze_last_n, total_layers)

                for layer in backbone.layers[: total_layers - layers_to_unfreeze]:
                    layer.trainable = False

                for layer in backbone.layers[total_layers - layers_to_unfreeze :]:
                    # ⚠️ Manter BatchNormalization congelado (best practice for transfer learning)
                    layer.trainable = not isinstance(
                        layer, tf.keras.layers.BatchNormalization
                    )

                logger.info(
                    f"✅ Fine-tune: liberadas {layers_to_unfreeze} de {total_layers} camadas (BNs mantidos congelados)."
                )
        else:
            logger.warning(
                "⚠️  Backbone pré-treinado não identificado - usando fallback"
            )
            trainable_layers = [l for l in model.layers if l.trainable]
            if unfreeze_last_n > 0:
                for layer in trainable_layers[-unfreeze_last_n:]:
                    layer.trainable = True
                logger.info(
                    f"🔄 Fallback: {len(trainable_layers[-unfreeze_last_n:])} camadas liberadas."
                )

    def build_model(self, hp):
        """
        Constrói modelo com hiperparâmetros sugeridos pelo tuner.

        Args:
            hp: Objeto HyperParameters do Keras Tuner

        Returns:
            Modelo Keras compilado
        """
        # Carregar ranges de tuning do YAML baseado no modelo
        search_space = self.model_config.get_tuning_search_space()

        # Hiperparâmetros a otimizar (com fallback para defaults)
        lr_range = search_space.get("learning_rate")

        dropout_range = search_space.get("dropout_rate")

        # Torna unfreeze_last_n_layers opcional
        unfreeze_range = search_space.get("unfreeze_last_n_layers", None)
        l2_range = search_space.get("l2_regularization")

        hp_learning_rate = hp.Float("learning_rate", **lr_range)
        dropout_rate = hp.Float("dropout_rate", **dropout_range)
        if unfreeze_range is not None:
            unfreeze_last_n = hp.Int("unfreeze_last_n_layers", **unfreeze_range)
        else:
            unfreeze_last_n = hp.values.get("unfreeze_last_n_layers", 0)
        l2_reg = hp.Float("l2_regularization", **l2_range)

        logger.info(
            f"Building model: lr={hp_learning_rate:.6f}, dropout={dropout_rate:.2f}, unfreeze={unfreeze_last_n}, l2={l2_reg:.6f}"
        )
        self.model_config.l2_regularization = float(l2_reg)
        self.model_config.dropout_rate = float(dropout_rate)

        image_config = self.data_splitter.image_config
        prepare_model = PrepareModel(self.model_config, image_config)
        model = prepare_model.build_model()

        # Configurar congelamento/descongelamento do backbone unificadamente
        self._configure_backbone_trainability(model, unfreeze_last_n)

        # Compilar usando mesmas definições do pipeline principal
        # TODO: implementar função com outros otimizadores, e chamar a função
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
        model.compile(
            optimizer=optimizer,
            loss=self.model_config.loss_function,
            metrics=self.model_config.metrics,
        )

        return model

    def search(
        self,
        max_trials: int,
        epochs_per_trial: int,
        log_trials_to_mlflow: bool = True,
    ):
        """
        Executa busca de hiperparâmetros com Bayesian Optimization.

        Args:
            max_trials: Número máximo de configurações a testar
            epochs_per_trial: Épocas de treino por candidato
            log_trials_to_mlflow: Se True, loga cada trial no MLflow como child run
        """
        logger.info("Iniciando busca com Bayesian Optimization...")
        logger.info(f"   Max trials: {max_trials}")
        logger.info(f"   Epochs per trial: {epochs_per_trial}")
        logger.info(f"   Log trials to MLflow: {log_trials_to_mlflow}")

        # Escolher hypermodel baseado no flag de MLflow
        if log_trials_to_mlflow:
            hypermodel = MLflowHyperModel(
                build_fn=self.build_model, experiment_name=self.model_config.model_name
            )
            logger.info("🔗 MLflow trial logging ATIVADO")
        else:
            hypermodel = self.build_model
            logger.info("🔗 MLflow trial logging DESATIVADO")

        # Criar tuner
        self.tuner = kt.BayesianOptimization(
            hypermodel=hypermodel,
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

        unfreeze_val = self.best_hp.values.get("unfreeze_last_n_layers")
        logger.info(
            f"   Unfreeze Last N:     {unfreeze_val if unfreeze_val is not None else 'N/A (from scratch)'}"
        )

        if "l2_regularization" in self.best_hp.values:
            logger.info(
                f"   L2 Regularization:   {self.best_hp.get('l2_regularization'):.6f}"
            )
        logger.info("✅ Melhor modelo recuperado - pronto para retreino")

    def _callbacks(self):
        # Suprimir warnings de HDF5 durante tuning (checkpoints internos do EarlyStopping)
        import warnings

        warnings.filterwarnings("ignore", message=".*HDF5 file.*")

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

        # Suprimir warnings de HDF5 do EarlyStopping
        import warnings

        warnings.filterwarnings("ignore", message=".*HDF5.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="keras")

        logger.info(
            f"Retreinando modelo com melhores hiperparâmetros por {epochs} épocas..."
        )

        # Reconstruir modelo com melhores hiperparâmetros
        model = self.build_model(self.best_hp)

        # Recompilar com LR fixo (pipeline)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.best_hp.get("learning_rate"),
                # epsilon=1e-8,
                # beta_1=0.9,
                # beta_2=0.999,
                # amsgrad=False,
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

        # Salvar modelo em múltiplos formatos
        output_dir = Path("artifacts/models/mobile")
        output_dir.mkdir(parents=True, exist_ok=True)

        # 📊 Salvar visualização da arquitetura do modelo final
        logger.info("\n" + "=" * 70)
        logger.info("📊 SALVANDO VISUALIZAÇÃO DA ARQUITETURA")
        logger.info("=" * 70)

        try:

            # 1. PNG da arquitetura
            architecture_path = (
                output_dir / f"{self.model_config.model_name}_architecture.png"
            )
            plot_model(
                model,
                to_file=str(architecture_path),
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",  # Top to Bottom
                expand_nested=True,
                dpi=150,
                show_layer_activations=True,
            )
            arch_size = architecture_path.stat().st_size / 1024
            logger.info(
                f"   ✅ Arquitetura PNG: {architecture_path.name} ({arch_size:.1f} KB)"
            )

            # 2. Summary em texto
            summary_path = output_dir / f"{self.model_config.model_name}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
            summary_size = summary_path.stat().st_size / 1024
            logger.info(
                f"   ✅ Model Summary: {summary_path.name} ({summary_size:.1f} KB)"
            )

            # 3. Informações detalhadas
            total_params = model.count_params()
            trainable_params = sum(
                [tf.size(w).numpy() for w in model.trainable_weights]
            )
            non_trainable_params = sum(
                [tf.size(w).numpy() for w in model.non_trainable_weights]
            )

            logger.info(f"\n   📈 Parâmetros do Modelo:")
            logger.info(f"      Total:          {total_params:,}")
            logger.info(f"      Treináveis:     {trainable_params:,}")
            logger.info(f"      Não-treináveis: {non_trainable_params:,}")
            logger.info("=" * 70 + "\n")

        except Exception as e:
            logger.warning(
                f"⚠️ Não foi possível salvar visualização da arquitetura: {e}"
            )

        self.save_multiple_formats(
            model, output_dir, self.model_config.model_name, train_ds
        )

        self.best_model = model
        return model, history

    def save_best_hyperparameters(
        self, model_name: str, save_dir: str = "artifacts/tuning"
    ):
        """Salva melhores hiperparâmetros em JSON."""
        if self.best_hp is None:
            raise ValueError("Execute .search() antes")

        save_path = Path(save_dir) / f"best_hyperparameters_{model_name}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best_params = {
            "learning_rate": float(self.best_hp.get("learning_rate")),
            "dropout_rate": float(self.best_hp.get("dropout_rate")),
        }

        if "unfreeze_last_n_layers" in self.best_hp.values:
            best_params["unfreeze_last_n_layers"] = int(
                self.best_hp.get("unfreeze_last_n_layers")
            )

        # Adicionar L2 se existir (compatibilidade com buscas antigas)
        if "l2_regularization" in self.best_hp.values:
            best_params["l2_regularization"] = float(
                self.best_hp.get("l2_regularization")
            )

        with open(save_path, "w") as f:
            json.dump(best_params, f, indent=2)

        logger.info(f"✅ Hiperparâmetros salvos em: {save_path}")

    def save_multiple_formats(
        self, model, output_dir, model_name: str, train_dataset=None
    ):
        """
        Salva modelo em múltiplos formatos para diferentes use cases.

        Args:
            model: Modelo Keras treinado
            output_dir: Diretório para salvar os modelos
            model_name: Nome base do modelo
            train_dataset: Dataset de treinamento para calibração do TFLite (opcional)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "=" * 70)
        logger.info("💾 SALVANDO MODELO EM MÚLTIPLOS FORMATOS")
        logger.info("=" * 70)

        # 1. H5 (Keras legado) - Para retreino
        # 1. .keras (Nativo TF 2.8+) - Recomendado
        logger.info("\n1️⃣ Salvando .keras (formato nativo TF 2.8+)...")
        keras_path = output_dir / f"{model_name}_keras_tuner_best.keras"
        model.save(str(keras_path))
        keras_size = keras_path.stat().st_size / (1024 * 1024)
        logger.info(f"   ✅ {keras_path.name} ({keras_size:.2f} MB)")

        # Definir h5_size para compatibilidade com o resumo
        h5_size = keras_size

        # 4. TFLite (mobile/edge) - Com quantização FP16
        logger.info("\n4️⃣ Salvando TFLite (mobile/edge)...")
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

            tflite_model = converter.convert()

            tflite_path = output_dir / f"{model_name}_keras_tuner_best.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)

            tflite_size = tflite_path.stat().st_size / (1024 * 1024)
            reduction = (1 - tflite_size / keras_size) * 100
            logger.info(
                f"   ✅ {tflite_path.name} ({tflite_size:.2f} MB, {reduction:.1f}% redução)"
            )
            logger.info("   📝 Quantização FP16 (16-bit floating point)")
        except Exception as e:
            logger.warning(f"   ⚠️ TFLite falhou: {e}")

        # 5. ONNX (opcional) - Portabilidade
        # TODO: implementar quantização ONNX (QAT ou PTQ) para reduzir tamanho e acelerar inferência em edge devices
        logger.info("\n5️⃣ Tentando salvar ONNX (portabilidade)...")
        try:
            import onnx
            import tf2onnx

            # spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input"),)
            spec = (
                tf.TensorSpec(self.image_config.size_tuple, tf.float32, name="input"),
            )

            output_path = tf2onnx.convert.from_keras(model, input_signature=spec)
            onnx_model = onnx.load(output_path)

            onnx_path = output_dir / f"{model_name}_keras_tuner_best.onnx"
            onnx.save(onnx_model, str(onnx_path))
            onnx_size = onnx_path.stat().st_size / (1024 * 1024)
            logger.info(f"   ✅ {onnx_path.name} ({onnx_size:.2f} MB)")
        except ImportError:
            logger.info(
                "   ⚠️ tf2onnx não instalado (opcional). "
                "Instale com: pip install tf2onnx onnx"
            )
        except Exception as e:
            logger.warning(f"   ⚠️ ONNX falhou: {e}")

        # Resumo
        logger.info("\n" + "=" * 70)
        logger.info("📦 RESUMO DOS FORMATOS SALVOS:")
        logger.info("=" * 70)
        logger.info(f"\n📁 Diretório: {output_dir}")
        logger.info("\n┌─ DESENVOLVIMENTO:")
        logger.info(
            f"│  .keras   {keras_size:6.2f} MB   - Formato nativo TF 2.8+ (✅ recomendado)"
        )
        logger.info("│")
        logger.info("├─ SERVIDORES:")
        logger.info("│  SavedModel .pb  - Para TensorFlow Serving / APIs REST")
        logger.info("│")
        logger.info("├─ MOBILE/EDGE:")
        if "tflite_size" in locals():
            logger.info(
                f"│  TFLite    {tflite_size:6.2f} MB   - ⚡ Rápido, compacto (Android/iOS)"
            )
        else:
            logger.info("│  TFLite              - (falhou na conversão)")
        logger.info("│")
        logger.info("└─ PORTABILIDADE:")
        logger.info("   ONNX              - Multiplataforma (Windows/Linux/Web)")
        logger.info("\n" + "=" * 70)
