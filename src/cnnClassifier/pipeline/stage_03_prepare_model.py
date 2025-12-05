from cnnClassifier.config.settings import ModelConfig, Paths
from cnnClassifier.utils.data_utils import create_dirs
from cnnClassifier import logger
import tensorflow as tf

STAGE_NAME = "Prepare Model"

class PrepareModelPipeline:
    def __init__(self):
        self.config = ModelConfig()
    
    def main(self):
        """Pipeline SIMPLES de preparação do modelo"""
        try:
            # Cria diretórios
            create_dirs(Paths.MODELS_DIR)
            
            # Carrega modelo base
            logger.info(f"🤖 Carregando {self.config.MODEL_NAME}...")
            base_model = tf.keras.applications.EfficientNetB0(
                input_shape=self.config.IMAGE_SIZE,
                weights='imagenet',
                include_top=False
            )
            
            # Constrói modelo completo
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(self.config.CLASSES, activation='softmax')
            ])
            
            # Compila
            model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config.LEARNING_RATE),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Salva
            model_path = Paths.MODELS_DIR / "base_model.keras"
            model.save(model_path)
            
            logger.info(f"✅ Modelo preparado: {model_path}")
            return model_path
            
        except Exception as e:
            logger.error(f"❌ Erro na preparação: {e}")
            raise

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = PrepareModelPipeline()
        model_path = pipeline.main()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<")
        
    except Exception as e:
        logger.exception(e)
        raise e