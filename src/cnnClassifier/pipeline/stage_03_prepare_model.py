from cnnClassifier.components.prepare_model import PrepareModel
from cnnClassifier.config.constants import MODELS_DIR
from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger
from cnnClassifier.utils.data_utils import create_dirs

logger = configure_logger(__name__)
STAGE_NAME = "Prepare Model"

class PrepareModelPipeline:
    def __init__(self):
        self.model_dir = MODELS_DIR
        self.model_config = ModelConfig()

    def main(self):
        """Prepara o modelo de classificação CNN."""
        try:
            logger.info("Iniciando a preparação do modelo...")
            
            # Cria diretório para modelos
            create_dirs(self.model_dir)

            preparador_modelo = PrepareModel(model_config=self.model_config)
            modelo = preparador_modelo.build_model()

            modelo_path = self.model_dir / "cnn_model.h5"
            modelo.save(modelo_path)

            logger.info(f"✅ Modelo salvo em: {modelo_path}")
            return {
                    "success": True,
                    "model_path": modelo_path,
                    "model_name": self.model_config.model_name,
                    "message": f"Modelo salvo com sucesso em {modelo_path}"
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
        
        pipeline = PrepareModelPipeline()
        model_path = pipeline.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e)
        raise e