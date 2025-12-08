from typing import Any, Dict
import tensorflow as tf

from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)

class ModelTraining:
    def __init__ (
            self,
            model: tf.keras.Model,
            model_config: ModelConfig
            ):
        self.model_config = model_config
        self.model = model
        self.history = None
    
    def train_model(
            self, 
            train_data: tf.data.Dataset, 
            validation_data: tf.data.Dataset
            ):
        try:
            logger.info("Iniciando treinamento do modelo...")
            logger.info(f"Épocas: {self.model_config.epochs}")
            self._compile_model()
            
            self.history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=self.model_config.epochs,
                verbose=1
            )
            return self.history

        except Exception as e:
            logger.error(f"❌ Erro durante treinamento: {e}")
            raise

    def get_training_metrics(self) -> Dict[str, Any]:
        """Retorna métricas do treinamento"""
        if self.history is None:
            return {"trained": False}
        
        return {
            "trained": True,
            "epochs_completed": len(self.history.history['loss']),
            "final_train_loss": self.history.history['loss'][-1],
            "final_val_loss": self.history.history['val_loss'][-1],
            "final_train_accuracy": self.history.history['accuracy'][-1],
            "final_val_accuracy": self.history.history['val_accuracy'][-1]
        }

    def _compile_model(self):
        self.model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.model_config.learning_rate
            ),
            loss=self.model_config.loss_function,
            metrics=self.model_config.metrics
        )
        logger.info("Modelo recompilado com configurações de treinamento")

