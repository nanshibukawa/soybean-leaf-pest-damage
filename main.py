# 📁 main.py
import os
from pathlib import Path

# 🔇 CONFIGURAÇÕES TF NO INÍCIO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from cnnClassifier.pipeline.stage_02_data_splitting import DataSplittingPipeline
from cnnClassifier.pipeline.stage_03_prepare_model import PrepareModelPipeline
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)

def main():
    logger.info("🚀 Iniciando pipeline de machine learning...")
    
    try:
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
        data_splitting = DataSplittingPipeline()
        stage2_result = data_splitting.main()

        if stage2_result["success"]:
            logger.info("✅ Stage 2 completo!")
        else:
            logger.error(f"❌ Stage 2 falhou: {stage2_result['error']}")
            return stage2_result

        # ===== STAGE 3: DATA PREPARATION =====
        logger.info("\n🔄 === Stage 3: Data Preparation ===")
        data_preparation = PrepareModelPipeline()
        stage3_result = data_preparation.main()

        if stage3_result["success"]:
            logger.info("✅ Stage 3 completo!")
        else:
            logger.error(f"❌ Stage 3 falhou: {stage3_result['error']}")
            return stage3_result

        logger.info("🏁 Pipeline completo com sucesso!")
        return {
            "stage1": stage1_result,
            "stage2": stage2_result,
            "stage3": stage3_result,
            "success": True
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