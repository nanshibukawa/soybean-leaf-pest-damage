import logging
import os
from pathlib import Path
import warnings
import json
import keras_tuner as kt
import typer

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

# 🔇 CONFIGURAÇÕES TF NO INÍCIO
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Só erros críticos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", ".*Invalid SOS parameters.*")


# 🔇 CONFIGURAÇÕES TF NO INÍCIO (antes de importar TensorFlow)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Só erros críticos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", ".*Invalid SOS parameters.*")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

logger = configure_logger(__name__)


def main(
    mode: str = typer.Option(
        "tune",
        help="Modo de execução: 'tune' (busca + retreino) ou 'retrain' (somente retreino com HPs salvos)",
    ),
    best_hp_path: Path = typer.Option(
        Path("artifacts/tuning/best_hyperparameters.json"),
        help="Caminho para o JSON com os melhores hiperparâmetros",
    ),
    max_trials: int = typer.Option(30, help="Número máximo de trials na busca"),
    epochs_per_trial: int = typer.Option(30, help="Épocas por trial na busca"),
    final_epochs: int = typer.Option(100, help="Épocas no retreino final"),
):
    logger.info("=" * 80)
    logger.info("🚀 PIPELINE DE TUNING - Keras Tuner com Bayesian Optimization")
    logger.info("=" * 80)

    try:

        # ===== STAGE 0: Load Configs =====
        logger.info("\n📋 === Stage 0: Carregando Configurações ===")
        model_config = ModelConfig.from_yaml(
            "model_params.yaml", experiment="mobilenet"
        )
        logger.info(f"✅ Configuração carregada: {model_config.model_name}")

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
        )

        data_splitter = DataSplitter(
            data_split_config=data_splitter_config,
            image_config=image_config,
            subset=DataSubsetType,
        )

        # Inicializar tuner
        logger.info("🔧 Inicializando Keras Tuner...")
        tuner = KerasTunerSearch(model_config, data_splitter)

        if mode == "tune":
            # 4a) Busca de hiperparâmetros
            logger.info("🔍 Fase 1/3: Busca de hiperparâmetros...")
            tuner.search(max_trials=max_trials, epochs_per_trial=epochs_per_trial)

            # 4b) Salvar melhores hiperparâmetros
            logger.info("💾 Salvando melhores hiperparâmetros...")
            tuner.save_best_hyperparameters()

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

        # ===== STAGE 5: Model Evaluation =====
        logger.info("\n🔄 === Stage 5: Model Evaluation (Detalhada) ===")
        try:
            eval_pipeline = ModelEvaluationPipeline(
                model_config=model_config, image_config=image_config, data=stage2_result
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
                    stage5_result["report_path"] = stage5_result["evaluation_result"][
                        "report_path"
                    ]
            else:
                logger.warning(f"❌ Stage 5 falhou: {stage5_result['error']}")

        except Exception as e:
            logger.warning(f"❌ Stage 5 não executado: {e}")
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

        return {
            "config": model_config,
            "stage1": stage1_result,
            "stage2": stage2_result,
            "stage3": {"skipped": True},
            "stage4": stage4_result,
            "stage5": stage5_result,
            "success": True,
        }

    except Exception as e:
        logger.exception(f"💥 Pipeline de tuning falhou: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    typer.run(main)
