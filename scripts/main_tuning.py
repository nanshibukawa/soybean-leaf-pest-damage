import logging
import os
from pathlib import Path
import warnings
import json
import keras_tuner as kt
import typer
import random
import numpy as np
import tensorflow as tf

from cnnClassifier.config.constants import DATA_SOURCE_DIR
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_data_splitting import DataSplittingPipeline
from cnnClassifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import (
    ImageConfig,
    ModelConfig,
    DataSplitterConfig,
    DataSubsetType,
)
from cnnClassifier.components.data_splitter import DataSplitter
from cnnClassifier.tuning.keras_tuner import KerasTunerSearch

import mlflow

from loguru import logger
from dotenv import load_dotenv

import dagshub

load_dotenv()

mlflow_uri = os.environ["MLFLOW_TRACKING_URI"]
mlflow_user = os.environ["MLFLOW_TRACKING_USERNAME"]
mlflow_pass = os.environ["MLFLOW_TRACKING_PASSWORD"]

# 🔇 CONFIGURAÇÕES TF NO INÍCIO
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Só erros críticos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", ".*Invalid SOS parameters.*")

# 📊 Habilitar MLflow System Metrics Logging
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

logging.getLogger("tensorflow").setLevel(logging.ERROR)

logger = configure_logger(__name__)


def main(
    mode: str = typer.Option(
        "tune",
        help="Modo de execução: 'tune' (busca + retreino) ou 'retrain' (somente retreino com HPs salvos)",
    ),
    experiment: str = typer.Option(
        "mobilenetv3large",
        help="Nome do experimento no YAML (ex: mobilenetv3large, efficientnetb0, vgg19)",
    ),
    best_hp_path: Path = typer.Option(
        None,  # Será construído dinamicamente se None
        help="Caminho para o JSON com os melhores hiperparâmetros (auto: artifacts/tuning/best_hyperparameters_<model>.json)",
    ),
    max_trials: int = typer.Option(30, help="Número máximo de trials na busca"),
    epochs_per_trial: int = typer.Option(50, help="Épocas por trial na busca"),
    final_epochs: int = typer.Option(100, help="Épocas no retreino final"),
    log_trials: bool = typer.Option(
        True, help="Logar cada trial no MLflow (recomendado)"
    ),
):
    logger.info("=" * 80)
    logger.info("🚀 PIPELINE DE TUNING - Keras Tuner com Bayesian Optimization")
    logger.info("=" * 80)

    try:

        # ===== STAGE 0: Load Configs =====
        logger.info("\n📋 === Stage 0: Carregando Configurações ===")
        model_config = ModelConfig.from_yaml("model_params.yaml", experiment=experiment)
        logger.info(
            f"✅ Configuração carregada: Experimento '{experiment}' - Modelo {model_config.model_name}"
        )

        seed = model_config.random_seed
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"🎲 Seed global definido: {seed}")

        # if use_mlflow:

        mlflow.set_experiment(f"tuning-{experiment}")

        dagshub.init(
            repo_owner="nanshibukawa", repo_name="soybean-leaf-pest-damage", mlflow=True
        )
        # mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_tracking_uri(
            "https://dagshub.com/nanshibukawa/soybean-leaf-pest-damage.mlflow"
        )

        # 📊 Configurar coleta de system metrics
        mlflow.set_system_metrics_sampling_interval(10)  # Coletar a cada 10 segundos
        mlflow.set_system_metrics_samples_before_logging(1)  # Logar após cada coleta

        with mlflow.start_run(
            run_name=f"{model_config.model_name}-{mode}", log_system_metrics=True
        ) as parent_run:
            logger.info(f"🔗 MLflow parent run: {parent_run.info.run_id}")
            logger.info(
                f"📊 Trial logging: {'ATIVADO' if log_trials else 'DESATIVADO'}"
            )

            mlflow.log_param("random_seed", seed)
            mlflow.log_param("model_name", model_config.model_name)
            mlflow.log_param("image_size", str(model_config.image_size))
            mlflow.log_param("batch_size", model_config.batch_size)
            mlflow.log_param("max_trials", max_trials)
            mlflow.log_param("epochs_per_trial", epochs_per_trial)
            mlflow.log_param("log_trials_to_mlflow", log_trials)
            logger.info(f"🎯 Batch Size da Config: {model_config.batch_size}")

            # Logar class_weights se existir
            class_weights = getattr(model_config, "class_weights", None)
            if class_weights:
                class_names = ["caterpillar", "diabrotica_speciosa", "healthy"]
                for class_id, weight in class_weights.items():
                    class_name = (
                        class_names[class_id]
                        if class_id < len(class_names)
                        else f"class_{class_id}"
                    )
                    mlflow.log_param(f"class_weight_{class_name}", weight)
            else:
                mlflow.log_param("class_weight", "None")

            # Logar augmentation params
            mlflow.log_param("augmentation_enabled", model_config.augmentation_enabled)
            if model_config.augmentation_enabled:
                mlflow.log_param("aug_rotation_range", model_config.rotation_range)
                mlflow.log_param(
                    "aug_brightness_range", str(model_config.brightness_range)
                )
                mlflow.log_param("aug_zoom_range", str(model_config.zoom_range))
                if hasattr(model_config, "contrast_range"):
                    mlflow.log_param(
                        "aug_contrast_range", str(model_config.contrast_range)
                    )
                if hasattr(model_config, "horizontal_flip"):
                    mlflow.log_param(
                        "aug_horizontal_flip", model_config.horizontal_flip
                    )

            # Construir path dos HPs se não fornecido
            if best_hp_path is None:
                best_hp_path = Path(
                    f"artifacts/tuning/best_hyperparameters_{model_config.model_name}.json"
                )
                logger.info(f"📝 Path de HPs (auto): {best_hp_path}")

            image_config = ImageConfig(
                altura=model_config.image_size[0],
                largura=model_config.image_size[1],
                canais=model_config.image_size[2],
                data_dir=Path(DATA_SOURCE_DIR),
            )

            # ===== STAGE 1: DATA INGESTION =====
            logger.info("\n🔄 === Stage 1: Data Ingestion ===")
            data_ingestion = DataIngestionPipeline()
            stage1_result = data_ingestion.main()

            if isinstance(stage1_result, Path):
                logger.info(f"✅ Stage 1 completo: {stage1_result}")
            else:
                logger.info(f"✅ Stage 1 completo: {stage1_result}")

            # ===== STAGE 2: DATA SPLITTING =====
            logger.info("\n🔄 === Stage 2: Data Splitting ===")
            data_splitting = DataSplittingPipeline(
                config=model_config, image_config=image_config
            )
            stage2_result = data_splitting.main()

            if stage2_result["success"]:
                logger.info("✅ Stage 2 completo!")
            else:
                logger.error(f"❌ Stage 2 falhou: {stage2_result['error']}")
                return stage2_result

            # ===== STAGE 3: SKIP PrepareModel =====
            # Keras Tuner cria modelos internamente com hiperparâmetros variados
            logger.info("=== Stage 3: PULADO (tuner cria modelos internamente) ===")

            # ===== STAGE 4: KERAS TUNER (Search + Retrain) =====
            logger.info("=== Stage 4: Keras Tuner - Busca e Retreino ===")

            # Preparar data splitter para o tuner
            data_splitter_config = DataSplitterConfig(
                batch_size=model_config.batch_size or 32,
                random_seed=model_config.random_seed,
                train_ratio=model_config.train_ratio,
                val_ratio=model_config.val_ratio,
                test_ratio=model_config.test_ratio,
            )

            data_splitter = DataSplitter(
                data_split_config=data_splitter_config,
                image_config=image_config,
                subset=DataSubsetType,
            )

            class_distribution = data_splitter.get_class_distribution()
            mlflow.log_dict(class_distribution, "data_split_class_distribution.json")

            # Inicializar tuner
            logger.info("🔧 Inicializando Keras Tuner...")
            tuner = KerasTunerSearch(model_config, data_splitter)

            if mode == "tune":
                # 4a) Busca de hiperparâmetros
                logger.info("🔍 Fase 1/3: Busca de hiperparâmetros...")
                tuner.search(
                    max_trials=max_trials,
                    epochs_per_trial=epochs_per_trial,
                    log_trials_to_mlflow=log_trials,
                )

                # 4b) Salvar melhores hiperparâmetros
                logger.info("💾 Salvando melhores hiperparâmetros...")
                tuner.save_best_hyperparameters(model_name=model_config.model_name)
                if best_hp_path.exists():
                    mlflow.log_artifact(best_hp_path, artifact_path="tuning")

                # 4c) Retreinar com melhores hiperparâmetros
                logger.info("📈 Fase 2/3: Retreinando modelo final...")
                best_model, history = tuner.retrain_best_model(epochs=final_epochs)
            else:
                # Modo retrain-only: carregar HPs e retreinar direto
                logger.info("🔁 Modo 'retrain': usando hiperparâmetros salvos")
                if not best_hp_path.exists():
                    raise FileNotFoundError(
                        f"Arquivo de hiperparâmetros não encontrado: {best_hp_path}"
                    )

                with open(best_hp_path) as f:
                    best_hp = json.load(f)

                logger.info("\n📋 Hiperparâmetros carregados:")
                logger.info(f"   Learning Rate:       {best_hp.get('learning_rate')}")
                logger.info(f"   Dropout Rate:        {best_hp.get('dropout_rate')}")
                logger.info(
                    f"   Unfreeze Layers:     {best_hp.get('unfreeze_last_n_layers')}"
                )
                if "l2_regularization" in best_hp:
                    logger.info(
                        f"   L2 Regularization:   {best_hp.get('l2_regularization')}"
                    )

                hp = kt.HyperParameters()
                hp.values = best_hp
                tuner.best_hp = hp

                logger.info("📈 Fase 2/3: Retreinando modelo final (retrain-only)...")
                best_model, history = tuner.retrain_best_model(epochs=final_epochs)

            logger.info("✅ Stage 4 completo!")

            stage4_result = {
                "success": True,
                "model": best_model,
                "history": history,
                "tuner": tuner,
            }

            if tuner.best_hp:
                mlflow.log_param("best_lr", tuner.best_hp.get("learning_rate"))
                mlflow.log_param("best_dropout", tuner.best_hp.get("dropout_rate"))
                mlflow.log_param(
                    "best_unfreeze", tuner.best_hp.get("unfreeze_last_n_layers")
                )
                mlflow.log_param("best_l2", tuner.best_hp.get("l2_regularization"))

            # ===== STAGE 5: Model Evaluation =====
            logger.info("\n🔄 === Stage 5: Model Evaluation (Detalhada) ===")
            try:
                eval_pipeline = ModelEvaluationPipeline(
                    model_config=model_config,
                    image_config=image_config,
                    data=stage2_result,
                )

                stage5_result = eval_pipeline.main(
                    validation_data=stage2_result["validation_data"],
                    model=best_model,
                    history=history,
                )

                if stage5_result["success"]:
                    logger.info("✅ Stage 5 completo!")
                    logger.info(
                        f"📊 Acurácia: {stage5_result['metrics'].get('accuracy', 0):.4f}"
                    )
                    logger.info(
                        f"🎯 F1-Score: {stage5_result['metrics'].get('f1_macro', 0):.4f}"
                    )
                    if (
                        "evaluation_result" in stage5_result
                        and "report_path" in stage5_result["evaluation_result"]
                    ):
                        stage5_result["report_path"] = stage5_result[
                            "evaluation_result"
                        ]["report_path"]
                else:
                    logger.warning(f"❌ Stage 5 falhou: {stage5_result['error']}")

            except Exception as e:
                logger.warning(f"❌ Stage 5 não executado: {e}")
                stage5_result = {"success": False, "error": str(e)}

            # ===== STAGE 6: Teste Final  =====
            logger.info("\n🔄 === Stage 6: Teste Final ===")
            try:
                test_data = stage2_result["test_data"]
                test_result = eval_pipeline.main(
                    validation_data=test_data,  # ← Usa teste em vez de validação
                    model=best_model,
                    history=history,
                )
                logger.info(
                    f"🧪 Teste Final - Acurácia: {test_result['metrics']['accuracy']:.4f}"
                    f"🧪 Teste Final - F1 score: {test_result['metrics']['f1_macro']:.4f}"
                )

                if test_result["success"]:
                    mlflow.log_metric(
                        "test_accuracy", test_result["metrics"].get("accuracy", 0)
                    )
                    mlflow.log_metric(
                        "test_f1_macro", test_result["metrics"].get("f1_macro", 0)
                    )

                    # Logar métricas por classe (teste)
                    class_report = test_result["metrics"].get(
                        "classification_report", {}
                    )
                    class_names = ["Caterpillar", "Diabrotica speciosa", "Healthy"]
                    for cls in class_names:
                        if cls in class_report:
                            cls_safe = cls.replace(" ", "_").lower()
                            mlflow.log_metric(
                                f"test_precision_{cls_safe}",
                                class_report[cls]["precision"],
                            )
                            mlflow.log_metric(
                                f"test_recall_{cls_safe}", class_report[cls]["recall"]
                            )
                            mlflow.log_metric(
                                f"test_f1_{cls_safe}", class_report[cls]["f1-score"]
                            )
                            mlflow.log_metric(
                                f"test_support_{cls_safe}", class_report[cls]["support"]
                            )
            except Exception as e:
                logger.warning(f"❌ Stage 6 não executado: {e}")
                stage5_result = {"success": False, "error": str(e)}

            # ===== RESUMO FINAL =====
            logger.info("\n" + "=" * 80)
            logger.info("✨ PIPELINE DE TUNING COMPLETO COM SUCESSO!")
            logger.info("=" * 80)
            logger.info(
                "📁 Modelo salvo: artifacts/models/mobilenetv3_keras_tuner_best.keras"
            )
            logger.info(f"📊 Relatório: {stage5_result.get('report_path', 'N/A')}")
            logger.info("=" * 80)

            if stage5_result["success"]:
                mlflow.log_metric(
                    "val_accuracy", stage5_result["metrics"].get("accuracy", 0)
                )
                mlflow.log_metric(
                    "val_f1_macro", stage5_result["metrics"].get("f1_macro", 0)
                )

                # Logar métricas por classe (validação)
                class_report = stage5_result["metrics"].get("classification_report", {})
                class_names = ["Caterpillar", "Diabrotica speciosa", "Healthy"]
                for cls in class_names:
                    if cls in class_report:
                        cls_safe = cls.replace(" ", "_").lower()
                        mlflow.log_metric(
                            f"val_precision_{cls_safe}", class_report[cls]["precision"]
                        )
                        mlflow.log_metric(
                            f"val_recall_{cls_safe}", class_report[cls]["recall"]
                        )
                        mlflow.log_metric(
                            f"val_f1_{cls_safe}", class_report[cls]["f1-score"]
                        )
                        mlflow.log_metric(
                            f"val_support_{cls_safe}", class_report[cls]["support"]
                        )

            evaluation_dir = stage5_result.get("evaluation_dir")
            if evaluation_dir and Path(evaluation_dir).exists():
                mlflow.log_artifacts(evaluation_dir, artifact_path="evaluation")

            keras_path = (
                Path("artifacts")
                / "models"
                / "mobile"
                / f"{model_config.model_name}_keras_tuner_best.keras"
            )
            if keras_path.exists():
                mlflow.log_artifact(keras_path, artifact_path="models")

            h5_path = (
                Path("artifacts")
                / "models"
                / "mobile"
                / f"{model_config.model_name}_keras_tuner_best.h5"
            )
            if h5_path.exists():
                mlflow.log_artifact(h5_path, artifact_path="models")

            tflite_path = (
                Path("artifacts")
                / "models"
                / "mobile"
                / f"{model_config.model_name}_keras_tuner_best.tflite"
            )
            if tflite_path.exists():
                mlflow.log_artifact(tflite_path, artifact_path="models")

            mlflow.keras.log_model(
                best_model,
                "model",
                registered_model_name=f"{model_config.model_name}_classifier",
            )

            return {
                "config": model_config,
                "stage1": stage1_result,
                "stage2": stage2_result,
                "stage3": {"skipped": True},
                "stage4": stage4_result,
                "stage5": stage5_result,
                "stage6": test_result,
                "success": True,
            }

    except Exception as e:
        logger.exception(f"💥 Pipeline de tuning falhou: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    typer.run(main)
