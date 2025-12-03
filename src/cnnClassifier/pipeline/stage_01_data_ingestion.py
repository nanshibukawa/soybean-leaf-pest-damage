from dataclasses import dataclass, field
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pathlib import Path
from cnnClassifier.utils.data_utils import download_from_gdrive, extract_zip, create_dirs
# from cnnClassifier.config.settings import DataConfig
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.logger import configure_logger
logger = configure_logger(__name__)
STAGE_NAME = "Data Ingestion"


@dataclass
class DataIngestionPipeline:
    config: DataIngestionConfig = field(default_factory=DataIngestionConfig)  # ← Factory function!

    # def __init__(self):
    #     self.config = DataIngestionConfig

    def main(self) -> Path:
        try:
            # 1. Cria diretórios
            create_dirs(self.config.root_dir)
            
            # 2. Baixa dados
            zip_path = download_from_gdrive(
                self.config.source_URL,
                self.config.local_datafile
            )
            # 3. Extrai dados
            data_path = extract_zip(zip_path, self.config.unzip_dir)
            logger.info(f"✅ Dados prontos em: {data_path}")
            return data_path
            
        except Exception as e:
            logger.error(f"Erro na ingestão: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = DataIngestionPipeline()
        data_path = pipeline.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.error(f"❌ stage {STAGE_NAME} failed: {e}"   )
        raise e