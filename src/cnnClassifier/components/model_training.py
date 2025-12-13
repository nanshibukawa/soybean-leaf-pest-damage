from typing import Any, Dict
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from cnnClassifier.entity.config_entity import ModelConfig
from cnnClassifier.utils.logger import configure_logger

logger = configure_logger(__name__)

class ModelTraining:
    """
    Componente para treinamento de modelos CNN.
    """

    def __init__(
            self,
            model: tf.keras.Model,
            model_config: ModelConfig
            ):
        self.model_config = model_config
        self.model = model
        self.history = None
        """
        Inicializa o componente de treinamento do modelo.
        
        Args:
            model: Modelo Keras pré-construído para treinar
            model_config: Configuração contendo parâmetros de treinamento (épocas, taxa de aprendizado, etc.)
        """
    def train_model(
            self, 
            train_data: tf.data.Dataset, 
            validation_data: tf.data.Dataset
            ) -> tf.keras.callbacks.History:

        """
        Treina o modelo com os dados fornecidos.

        Args:
            train_data: Conjunto de dados de treinamento (tf.data.Dataset)
            validation_data: Conjunto de dados de validação para monitorar o treinamento

        Returns:
            tf.keras.callbacks.History: Histórico de treinamento contendo perda e métricas

        Raises:
            Exception: Se o treinamento falhar por qualquer motivo
        """
        try:
            logger.info("Iniciando treinamento do modelo...")
            logger.info(f"Épocas: {self.model_config.epochs}")
            self._compile_model()
            
            callbacks = [
                # Early Stopping - para quando validation loss não melhora
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1,
                    mode='min'
                ),
                
                # Reduce Learning Rate on Plateau - reduz LR quando estagna
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1,
                    mode='min'
                )
            ]
            
            self.history = self.model.fit(
                train_data,
                validation_data=validation_data,
                epochs=self.model_config.epochs,
                callbacks=callbacks,
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
        
        history_dict = self.history.history

        return {
            "trained": True,
            "epochs_completed": len(history_dict.get('loss', [])),
            "final_train_loss": history_dict.get('loss', [None])[-1] if 'loss' in history_dict else None,
            "final_val_loss": history_dict.get('val_loss', [None])[-1] if 'val_loss' in history_dict else None,
            "final_train_accuracy": history_dict.get('accuracy', [None])[-1] if 'accuracy' in history_dict else None,
            "final_val_accuracy": history_dict.get('val_accuracy', [None])[-1] if 'val_accuracy' in history_dict else None
        }

    def _compile_model(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model_config.learning_rate
            ),
            loss=self.model_config.loss_function,
            metrics=self.model_config.metrics
        )
        logger.info("Modelo recompilado com configurações de treinamento")

