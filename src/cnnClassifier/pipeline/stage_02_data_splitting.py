from pathlib import Path

from cnnClassifier.components.data_splitter import DataSplitter
from cnnClassifier.config.constants import DATA_SOURCE_DIR
from cnnClassifier.entity.config_entity import (
    DataSplitterConfig,
    DataSubsetType,
    ImageConfig,
    ModelConfig,
)
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)


STAGE_NAME = "Data Splitting"


class DataSplittingPipeline:
    def __init__(self, config: ModelConfig = None, image_config: ImageConfig = None):
        self.DATA_SOURCE_DIR = DATA_SOURCE_DIR
        self.config = config
        self.image_config = image_config

    def main(self):
        """
        Executa a divisão dos dados em train/val/test
        Returns:
            dict: Resultados da divisão dos dados
            - success (bool): Indica se a divisão foi bem-sucedida
            - train_data (list): Dados de treino
            - validation_data (list): Dados de validação

        """

        try:

            logger.info("Iniciando o DataSplitter...")

            data_split_config = DataSplitterConfig(
                batch_size=self.config.batch_size,
                random_seed=self.config.random_seed,
                train_ratio=self.config.train_ratio,
                val_ratio=self.config.val_ratio,
            )

            data_splitter = DataSplitter(
                data_split_config=data_split_config,
                image_config=self.image_config,
                subset=DataSubsetType,
            )

            train_data = data_splitter.load_train_data()
            validation_data = data_splitter.load_validation_data()

            logger.info("Dados de treino e validação carregados com sucesso.")

            return {
                "success": True,
                "train_data": train_data,
                "validation_data": validation_data,
            }
        except Exception as e:
            logger.error(f"❌ Erro no pipeline de divisão: {e}")
            return {"success": False, "error": str(e)}


if __name__ == "__main__":
    try:
        logger.info("📋 Carregando configurações...")
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

        logger.info(f"***" * 10)
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataSplittingPipeline(config=model_config, image_config=image_config)
        stats = obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        logger.info(f"📈 Estatísticas: {stats}")
    except Exception as e:
        logger.exception(e)
        raise e
