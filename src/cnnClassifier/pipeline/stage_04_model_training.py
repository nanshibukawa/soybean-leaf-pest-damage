import tensorflow as tf
from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger


from cnnClassifier.components.model_training import ModelTraining


MODEL_STAGE_NAME = "Model Training"
logger = configure_logger(__name__)

class ModelTrainingPipeline:
    """
    Pipeline para treinamento do modelo CNN.

    Esta pipeline carrega um modelo preparado, treina-o nos conjuntos de dados fornecidos
    e retorna métricas e histórico de treinamento.
    """
    def __init__(self, config: ModelConfig):
        """
        Inicializa o pipeline de treinamento do modelo.
        
        Args:
            config: Configuração do modelo contendo parâmetros de treinamento
        """
        self.config = config
    
    def main(self, stage3_result: dict, stage2_result: dict):
        """Executa o treinamento do modelo"""
        try:
            # Carrega modelo salvo
            model_path = stage3_result['model_path']
            model = tf.keras.models.load_model(model_path)
            
            # Inicializa treinamento
            trainer = ModelTraining(model, self.config)
            
            # Treina modelo
            history = trainer.train_model(
                train_data=stage2_result['train_data'],
                validation_data=stage2_result['validation_data']
            )
            
            # Salva modelo treinado
            trained_model_path = self.config.model_dir / f"{self.config.model_name}_trained.keras"
            model.save(trained_model_path)
            logger.info(f"✅ Trained model saved to: {trained_model_path}")

            return {
                "success": True,
                "model": model,
                "history": history,
                "metrics": trainer.get_training_metrics()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return {"success": False, "error": str(e)}
        
if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {MODEL_STAGE_NAME} started <<<<<<")
        # Load configuration
        model_config = ModelConfig.from_yaml("model_params.yaml")
        pipeline = ModelTrainingPipeline(config=model_config)
        # Note: This requires previous stage results
        # result = pipeline.main(stage3_result, stage2_result)
        logger.info(f">>>>>> stage {MODEL_STAGE_NAME} requires stage 2 and 3 results <<<<<<")
    except Exception as e:
        logger.exception(e)
        raise e