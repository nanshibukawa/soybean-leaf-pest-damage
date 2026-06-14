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

        seed = getattr(model_config, "random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def _configure_backbone_trainability(
        self, model: tf.keras.Model, unfreeze_last_n: int
    ):
        """Configura o congelamento/descongelamento das camadas do backbone."""
        try:
            backbone = model.get_layer("core_backbone")
            logger.info(
                f"🧠 Backbone 'core_backbone' encontrado ({len(backbone.layers)} camadas)"
            )
        except ValueError:
            backbone = None
            logger.warning("⚠️ Backbone 'core_backbone' não encontrado")

        if backbone is not None:
            total_layers = len(backbone.layers)
            is_from_scratch = (
                not self.model_config.weights
                or str(self.model_config.weights).lower() == "none"
            )

            if is_from_scratch:
                logger.info(
                    f"🔓 Modelo from scratch detectado. Todas as {total_layers} camadas ativas."
                )
                backbone.trainable = True
            else:
                backbone.trainable = True
                layers_to_unfreeze = min(unfreeze_last_n, total_layers)

                for layer in backbone.layers[: total_layers - layers_to_unfreeze]:
                    layer.trainable = False

                for layer in backbone.layers[total_layers - layers_to_unfreeze :]:
                    layer.trainable = not isinstance(
                        layer, tf.keras.layers.BatchNormalization
                    )

                logger.info(
                    f"✅ Fine-tune: liberadas {layers_to_unfreeze} camadas (BNs congelados)."
                )
        else:
            logger.warning("⚠️ Backbone não identificado - usando fallback")
            trainable_layers = [l for l in model.layers if l.trainable]
            if unfreeze_last_n > 0:
                for layer in trainable_layers[-unfreeze_last_n:]:
                    layer.trainable = True
                logger.info(
                    f"🔄 Fallback: {len(trainable_layers[-unfreeze_last_n:])} camadas liberadas."
                )

    def _calculate_steps(self, epochs: int) -> tuple:
        """Calcula dinamicamente os steps de treino para o CosineDecay."""
        try:
            train_ds = self.data_splitter.load_train_data()
            steps_per_epoch = len(train_ds)
        except Exception:
            steps_per_epoch = 32  # Fallback seguro caso o gerador não exponha o __len__

        total_steps = steps_per_epoch * epochs
        warmup_steps = int(total_steps * 0.1)  # 10% do treino total será warmup
        return total_steps, warmup_steps

    def build_model(self, hp, epochs_context: int = 50):
        """Constrói e compila o modelo com suporte a busca dinâmica de hiperparâmetros."""
        search_space = self.model_config.get_tuning_search_space()

        lr_range = search_space.get("learning_rate")
        dropout_range = search_space.get("dropout_rate")
        unfreeze_range = search_space.get("unfreeze_last_n_layers", None)
        l2_range = search_space.get("l2_regularization")

        # Registro de hiperparâmetros no espaço de busca
        hp_learning_rate = hp.Float("learning_rate", **lr_range)
        dropout_rate = hp.Float("dropout_rate", **dropout_range)

        if unfreeze_range is not None:
            unfreeze_last_n = hp.Int("unfreeze_last_n_layers", **unfreeze_range)
        else:
            unfreeze_last_n = hp.values.get("unfreeze_last_n_layers", 0)

        l2_reg = hp.Float("l2_regularization", **l2_range)

        # 🔥 ALTERAÇÃO 1: Adicionar a taxa de esparsidade do Top-K no Tuner!
        # Isso permite encontrar o balanço exato entre manchas de praga e tecido saudável.
        top_k_percent = hp.Float(
            "top_k_percent", min_value=0.05, max_value=0.25, step=0.05, default=0.15
        )

        logger.info(
            f"Building model: lr={hp_learning_rate:.6f}, dropout={dropout_rate:.2f}, "
            f"unfreeze={unfreeze_last_n}, l2={l2_reg:.6f}, top_k={top_k_percent:.2f}"
        )

        # Injetar os valores amostrados no objeto de configuração antes do build
        self.model_config.l2_regularization = float(l2_reg)
        self.model_config.dropout_rate = float(dropout_rate)

        # Sobrescrever temporariamente o valor estático do YAML com o valor dinâmico escolhido pelo Tuner
        image_config = self.data_splitter.image_config
        prepare_model = PrepareModel(self.model_config, image_config)

        # Modificar a instância interna antes da montagem final do grafo
        prepare_model.model_config.top_k_percent = float(top_k_percent)
        model = prepare_model.build_model()

        # Configurar malha de treinamento do backbone
        self._configure_backbone_trainability(model, unfreeze_last_n)

        # 🔥 ALTERAÇÃO 2: Cálculo dos steps dinâmicos baseados no contexto da fase de execução
        total_decay_steps, warmup_steps = self._calculate_steps(epochs_context)

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-5,
            decay_steps=total_decay_steps,
            alpha=0.001,  # Não zerar totalmente o LR evita estagnação extrema no fim do treino
            warmup_target=hp_learning_rate,
            warmup_steps=warmup_steps,
        )

        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=0.9,
            nesterov=True,
        )

        loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=self.model_config.metrics,
        )
        return model

    def search(
        self, max_trials: int, epochs_per_trial: int, log_trials_to_mlflow: bool = True
    ):
        """Executa busca de hiperparâmetros com Bayesian Optimization."""
        logger.info("Iniciando busca com Bayesian Optimization...")

        # Função lambda para repassar o contexto das épocas de trial ao build_model
        build_with_context = lambda hp: self.build_model(
            hp, epochs_context=epochs_per_trial
        )

        if log_trials_to_mlflow:
            hypermodel = MLflowHyperModel(
                build_fn=build_with_context,
                experiment_name=self.model_config.model_name,
            )
            logger.info("🔗 MLflow trial logging ATIVADO")
        else:
            hypermodel = build_with_context
            logger.info("🔗 MLflow trial logging DESATIVADO")

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

        train_ds = self.data_splitter.load_train_data()
        val_ds = self.data_splitter.load_validation_data()

        self.tuner.search(
            train_ds,
            validation_data=val_ds,
            epochs=epochs_per_trial,
            callbacks=self._callbacks(),
            class_weight=self.model_config.class_weights,
            verbose=1,
        )

        self.best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        logger.info("✅ Melhor modelo mapeado. Melhores parâmetros extraídos.")

    def _callbacks(self):
        import warnings

        warnings.filterwarnings("ignore", message=".*HDF5 file.*")
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                verbose=1,
                mode="max",
            ),
        ]

    def retrain_best_model(self, epochs: int = 100):
        """Reconstrói e treina o modelo final adaptando o agendador de LR para o espaço estendido."""
        if self.best_hp is None:
            raise ValueError("Execute .search() antes de .retrain_best_model()")

        import warnings

        warnings.filterwarnings("ignore", message=".*HDF5.*")
        warnings.filterwarnings("ignore", category=UserWarning, module="keras")

        logger.info(
            f"Retreinando modelo final ajustando scheduler para o ciclo de {epochs} épocas..."
        )

        # 🔥 ALTERAÇÃO 3: O modelo final é reconstruído passando o escopo correto de épocas estendidas
        model = self.build_model(self.best_hp, epochs_context=epochs)

        train_ds = self.data_splitter.load_train_data()
        val_ds = self.data_splitter.load_validation_data()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=15,
                restore_best_weights=True,
                verbose=1,
                min_delta=0.001,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=callbacks,
            class_weight=self.model_config.class_weights,
            verbose=1,
        )

        output_dir = Path("artifacts/models/mobile")
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_architecture_metadata(model, output_dir)
        self.save_multiple_formats(
            model, output_dir, self.model_config.model_name, train_ds
        )

        self.best_model = model
        return model, history

    def _save_architecture_metadata(self, model, output_dir):
        """Gera os metadados textuais e gráficos de visualização técnica do grafo."""
        try:
            architecture_path = (
                output_dir / f"{self.model_config.model_name}_architecture.png"
            )
            plot_model(
                model,
                to_file=str(architecture_path),
                show_shapes=True,
                show_layer_names=True,
                rankdir="TB",
                expand_nested=True,
                dpi=150,
                show_layer_activations=True,
            )

            summary_path = output_dir / f"{self.model_config.model_name}_summary.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                model.summary(print_fn=lambda x: f.write(x + "\n"))
        except Exception as e:
            logger.warning(f"⚠️ Falha ao salvar artefatos gráficos de arquitetura: {e}")

    def save_best_hyperparameters(
        self, model_name: str, save_dir: str = "artifacts/tuning"
    ):
        """Salva os melhores hiperparâmetros consolidados em formato JSON estruturado."""
        if self.best_hp is None:
            raise ValueError("Execute .search() antes")

        save_path = Path(save_dir) / f"best_hyperparameters_{model_name}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        best_params = {
            "learning_rate": float(self.best_hp.get("learning_rate")),
            "dropout_rate": float(self.best_hp.get("dropout_rate")),
            "top_k_percent": float(
                self.best_hp.get("top_k_percent")
            ),  # Registra o Top-K vencedor
        }

        if "unfreeze_last_n_layers" in self.best_hp.values:
            best_params["unfreeze_last_n_layers"] = int(
                self.best_hp.get("unfreeze_last_n_layers")
            )
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
            # --- PATCH PARA PYTHON 3.12 + KERAS 3 INCOMPATIBILIDADE TFLITE ---
            import inspect
            try:
                orig_check_instance = inspect._check_instance
                def patched_check_instance(obj, attr):
                    try:
                        return orig_check_instance(obj, attr)
                    except TypeError:
                        return inspect._sentinel
                inspect._check_instance = patched_check_instance
            except Exception:
                pass

            try:
                from tensorflow.python.util import keras_deps
                class DummyContextManager:
                    def __enter__(self): return self
                    def __exit__(self, exc_type, exc_val, exc_tb): pass
                class DummyCallContext:
                    def enter(self, model, inputs, build_graph=False, training=False):
                        return DummyContextManager()
                keras_deps.register_call_context_function(lambda: DummyCallContext())
            except Exception:
                pass
            # -----------------------------------------------------------------

            # Reconstruir modelo limpo (sem augmentation/preprocessing_lambda que falham no TFLite)
            try:
                input_shape = model.input_shape
                if isinstance(input_shape, list) and len(input_shape) > 0:
                    shape_tuple = input_shape[0][1:]
                else:
                    shape_tuple = input_shape[1:]
                    
                inputs_clean = tf.keras.layers.Input(shape=shape_tuple)
                x_clean = inputs_clean
                
                # Encontrar a camada core_backbone
                backbone_layer = model.get_layer("core_backbone")
                x_clean = backbone_layer(x_clean, training=False)
                
                # Aplicar camadas subsequentes
                layer_names = [l.name for l in model.layers]
                backbone_idx = layer_names.index("core_backbone")
                for layer in model.layers[backbone_idx + 1:]:
                    x_clean = layer(x_clean)
                    
                clean_model = tf.keras.Model(inputs=inputs_clean, outputs=x_clean)
                converter = tf.lite.TFLiteConverter.from_keras_model(clean_model)
            except Exception as e_rebuild:
                logger.warning(f"   ⚠️ Reconstrução limpa falhou, usando modelo direto: {e_rebuild}")
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
