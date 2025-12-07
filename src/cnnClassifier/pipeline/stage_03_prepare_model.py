from pathlib import Path
from cnnClassifier.components.prepare_model import PrepareModel
from cnnClassifier.config.constants import MODELS_DIR

from cnnClassifier.entity.config_entity import ImageConfig, ModelConfig
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.utils.data_utils import create_dirs 

logger = configure_logger(__name__)
STAGE_NAME = "Prepare Model"

class PrepareModelPipeline:
    def __init__(
            self, 
            config_path: str="model_params.yaml",
            experiment: str = None
            ):
        """
        Inicializa o pipeline de preparação de modelo.
        
        Args:
            config_path: Caminho para o arquivo YAML de configuração
            experiment: Nome do experimento específico para carregar
        """
        self.model_dir = MODELS_DIR
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

        try:
            logger.info(f"Carregando configuração de: {config_path}")
            if experiment:
                logger.info(f"Usando experimento: {experiment}")
            self.model_config = ModelConfig.from_yaml(config_path, experiment)
            
            # Criar ImageConfig baseada nas configurações do YAML
            self.image_config = ImageConfig(
                altura=self.model_config.image_size[0],
                largura=self.model_config.image_size[1],
                canais=self.model_config.image_size[2],
                data_dir=Path("artifacts/data/")  # ou seu path padrão
            )
            logger.info(f"Configuração carregada: {self.model_config.model_name}")
        except Exception as e:
            logger.error(f"Erro ao carregar YAML: {e}.")


    def main(self):
        """Prepara o modelo de classificação CNN."""
        try:
            logger.info("Iniciando a preparação do modelo...")
            logger.info(f"Modelo: {self.model_config.model_name}")
            logger.info(f"Dimensões: {self.image_config.size_tuple}")
            logger.info(f"Classes: {self.model_config.num_classes}")
            
            # Cria diretório para modelos
            create_dirs(self.model_dir)

            # Usar tanto model_config quanto image_config
            preparador_modelo = PrepareModel(
                model_config=self.model_config,
                image_config=self.image_config
            )
            modelo = preparador_modelo.build_model()

            # Usar nome baseado na configuração
            modelo_filename = f"{self.model_config.model_name.lower()}_model.keras"
            modelo_path = self.model_dir / modelo_filename
            modelo.save(modelo_path)

            logger.info(f"✅ Modelo salvo em: {modelo_path}")
            return {
                "success": True,
                "model_path": modelo_path,
                "model_name": self.model_config.model_name,
                "image_size": self.image_config.size_tuple,
                "num_classes": self.model_config.num_classes,
                "message": f"Modelo {self.model_config.model_name} salvo com sucesso em {modelo_path}"
            }

        except Exception as e:
            logger.error(f"Erro na preparação do modelo: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Falha na preparação do modelo"
            }
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = PrepareModelPipeline(config_path="model_params.yaml")
        result = pipeline.main()
        
        if result["success"]:
            logger.info(f">>>>>> stage {STAGE_NAME} completed successfully <<<<<<")
        else:
            logger.error(f">>>>>> stage {STAGE_NAME} failed <<<<<<")
        
    except Exception as e:
        logger.exception(e)
        raise e