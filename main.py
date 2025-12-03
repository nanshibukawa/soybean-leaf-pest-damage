from src.cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)

def main():
    """Pipeline principal seguindo método KISS"""
    try:
        logger.info("🚀 Iniciando pipeline de machine learning...")
        
        # Stage 1: Data Ingestion
        logger.info("🔄 === Stage 1: Data Ingestion ===")
        ingestion_pipeline = DataIngestionPipeline()
        ingestion_results = ingestion_pipeline.main()
        logger.info(f"Resultados da Ingestão de Dados: {ingestion_results}")
        
    except Exception as e:
        logger.error(f"❌ Erro no pipeline principal: {e}")
        raise

if __name__ == "__main__":
    try:
        results = main()
        print("🏁 Pipeline finalizado!")
        print(f"📈 Resultados: {results}")
        
    except Exception as e:
        logger.exception(f"💥 Falha crítica no pipeline: {e}")
        print("❌ Pipeline falhou! Verifique os logs para detalhes.")