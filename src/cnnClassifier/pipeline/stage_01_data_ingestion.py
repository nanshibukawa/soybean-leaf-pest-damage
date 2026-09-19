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
            # Garante os datasets finais prontos para treino e benchmark (DatasetPests-split, INSECT12C-test, IP102)
            final_pests_dir = self.config.data_final_dir / "final" / "DatasetPests-split"
            ip102_cropped_dir = self.config.data_final_dir / "ip102_cropped"
            insect12c_test_dir = self.config.data_final_dir / "final" / "INSECT12C-test"
            
            if not final_pests_dir.exists() or not ip102_cropped_dir.exists() or not insect12c_test_dir.exists():
                logger.info("📦 Ingestão: Baixando Dataset Final e IP102 Cropped do Google Drive (~516 MB)...")
                create_dirs(self.config.data_final_dir)
                final_zip_path = download_file(
                    self.config.data_final_url,
                    self.config.data_final_zip
                )
                extract_zip(final_zip_path, self.config.data_final_dir)
                logger.info("✅ Dataset Final e IP102 Cropped extraídos com sucesso!")
            else:
                logger.info("✅ Dataset Final e IP102 Cropped já presentes localmente.")
            
            return final_pests_dir
            
        except Exception as e:
            logger.error(f"Erro na ingestão de dados: {e}")
            raise

    def download_raw_datasets(self):
        """Baixa os datasets brutos originais (apenas para pipelines de recorte/pré-processamento)."""
        try:
            create_dirs(self.config.root_dir)
            logger.info("📦 Ingestão: Baixando DatasetPests bruto...")
            download_file(self.config.source_URL, self.config.local_datafile)
            logger.info("📦 Ingestão: Baixando iNaturalist Raw...")
            download_file(self.config.inat_raw_url, self.config.inat_raw_zip)
            logger.info("📦 Ingestão: Baixando INSECT12C Dataset...")
            download_file(self.config.insect12c_url, self.config.insect12c_zip)
        except Exception as e:
            logger.error(f"Erro na ingestão de dados brutos: {e}")
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