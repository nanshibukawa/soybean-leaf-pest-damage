import os
from pathlib import Path
import sys
import gdown
import traceback
import zipfile

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


from dataclasses import dataclass
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.utils.logger import configure_logger



logger = configure_logger(logger_name = __name__)

@dataclass
class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_google_drive(self) -> str:
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = str(self.config.local_datafile)  # Converter Path para string
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.debug(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id, zip_download_dir)
            logger.debug(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return True
        except Exception as e:
            logger.error(f"Error occurred while downloading file. Error: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False

    def extract_zip_file(self):
        unzip_path = str(self.config.unzip_dir)
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(str(self.config.local_datafile), 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def run_complete_ingestion(self):
        download_success = self.download_google_drive()
        if download_success:
            self.extract_zip_file()
            logger.info("Data ingestion completed successfully!")
        else:
            logger.error("Data download failed, skipping extraction.")
        return self.config.unzip_dir


if __name__ == "__main__":
    
    config = DataIngestionConfig(
        root_dir=Path("artifacts/data_ingestion"),
        source_URL="https://drive.google.com/file/d/1t;Qc-PGonSVSDtfL4HEsd30klq0gXxjqP/view?usp=sharing",
        local_datafile=Path("artifacts/data_ingestio;n/data.zip"),
        unzip_dir=Path("artifacts/data_ingestion")
    )
    
    data_ingestion = DataIngestion(config)
    data_ingestion.run_complete_ingestion()


