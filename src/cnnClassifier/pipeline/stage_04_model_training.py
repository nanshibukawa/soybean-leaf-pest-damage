import tensorflow as tf
from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger


from cnnClassifier.components.model_training import ModelTraining

logger = configure_logger(__name__)

class ModelTrainingPipeline:
    def __init__(self, config: ModelConfig):
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
            
            return {
                "success": True,
                "model": model,
                "history": history,
                "metrics": trainer.get_training_metrics()
            }
            
        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            return {"success": False, "error": str(e)}
        
# if __name__ == "__main__":
#     main()