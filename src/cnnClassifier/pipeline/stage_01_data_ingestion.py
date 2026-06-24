from dataclasses import dataclass, field
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from pathlib import Path
from cnnClassifier.utils.data_utils import download_file, extract_zip, create_dirs
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
            
            # 2. Baixa e extrai DatasetPests
            logger.info("📦 Ingestão: Baixando DatasetPests...")
            zip_path = download_file(
                self.config.source_URL,
                self.config.local_datafile
            )
            data_path = extract_zip(zip_path, self.config.unzip_dir)
            
            # 3. Baixa e extrai iNaturalist Raw
            logger.info("📦 Ingestão: Baixando iNaturalist Raw (Google Drive)...")
            inat_zip_path = download_file(
                self.config.inat_raw_url,
                self.config.inat_raw_zip
            )
            extract_zip(inat_zip_path, self.config.inat_unzip_dir)
            
            # 4. Baixa e extrai INSECT12C
            logger.info("📦 Ingestão: Baixando INSECT12C Dataset (GitHub)...")
            insect_zip_path = download_file(
                self.config.insect12c_url,
                self.config.insect12c_zip
            )
            extract_zip(insect_zip_path, self.config.insect12c_unzip_dir)
            
            logger.info(f"✅ Todos os dados baixados e extraídos com sucesso!")
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