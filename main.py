import os
from pathlib import Path
import warnings


# 🔇 CONFIGURAÇÕES TF NO INÍCIO
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Só erros críticos
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", ".*Invalid SOS parameters.*")

from cnnClassifier.config.constants import DATA_SOURCE_DIR
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_data_splitting import DataSplittingPipeline
from cnnClassifier.pipeline.stage_03_prepare_model import PrepareModelPipeline
from cnnClassifier.pipeline.stage_04_model_training import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_05_model_evaluation import ModelEvaluationPipeline

from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.entity.config_entity import ImageConfig, ModelConfig

logger = configure_logger(__name__)

def main():
    logger.info("🚀 Iniciando pipeline de machine learning...")

    try:
        # ===== STAGE 0: Load Configs =====
        logger.info("📋 Carregando configurações...")
        # model_config = ModelConfig.from_yaml(config_path="model_params.yaml")
        # model_config = ModelConfig.from_yaml("model_params.yaml", experiment="vgg_transfer")
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
        logger.info("🔄 === Stage 1: Data Ingestion ===")
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

        # ===== STAGE 3: DATA PREPARATION =====
        logger.info("\n🔄 === Stage 3: Data Preparation ===")
        data_preparation = PrepareModelPipeline(
            model_config=model_config, image_config=image_config
        )
        stage3_result = data_preparation.main()

        if stage3_result["success"]:
            logger.info("✅ Stage 3 completo!")
        else:
            logger.error(f"❌ Stage 3 falhou: {stage3_result['error']}")
            return stage3_result

        # ===== STAGE 4: Model Training =====
        logger.info("\n🔄 === Stage 4: Model Training ===")
        model_training = ModelTrainingPipeline(config=model_config)
        stage4_result = model_training.main(
            stage2_result=stage2_result, stage3_result=stage3_result
        )

        if stage4_result["success"]:
            logger.info("✅ Stage 4 completo!")
        else:
            logger.error(f"❌ Stage 4 falhou: {stage4_result['error']}")
            return stage4_result

        # ===== STAGE 5: Model Evaluation =====
        logger.info("\n🔄 === Stage 5: Model Evaluation ===")
        try:

            eval_pipeline = ModelEvaluationPipeline(
                model_config=model_config, image_config=image_config, data=stage2_result
            )

            stage5_result = eval_pipeline.main(
                validation_data=stage2_result["validation_data"],
                model=stage4_result["model"],
                history=stage4_result.get("history"),
            )

            if stage5_result["success"]:
                logger.info("✅ Stage 5 completo!")
                logger.info(f"📊 Acurácia: {stage5_result['metrics']['accuracy']:.4f}")
                logger.info(f"🎯 F1-Score: {stage5_result['metrics']['f1_macro']:.4f}")
            else:
                logger.warning(f"❌ Stage 5 falhou: {stage5_result['error']}")
                return stage5_result
        except Exception as e:
            logger.warning(f"❌ Stage 5 não executado: {e}")

        logger.info("🏁 Pipeline completo com sucesso!")

        return {
            "config": model_config,
            "stage1": stage1_result,
            "stage2": stage2_result,
            "stage3": stage3_result,
            "stage4": stage4_result,
            "stage5": stage5_result,
            "success": True,
        }
    except Exception as e:
        logger.exception(f"💥 Pipeline falhou: {e}")
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    result = main()

    if result and result.get("success"):
        print("\n🎉 PIPELINE EXECUTADO COM SUCESSO! 🎉")
    else:
        print("\n💥 PIPELINE FALHOU!")
        exit(1)
